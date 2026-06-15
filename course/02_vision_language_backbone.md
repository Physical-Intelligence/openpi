# Module 02 — The Vision-Language Backbone (SigLIP + Gemma → token sequence)

## Learning objectives

- Explain how a 224×224 image becomes a sequence of `~256` token embeddings via SigLIP.
- Explain how text becomes token embeddings with the Gemma embedder.
- Understand how `embed_prefix` concatenates image + language (+ state) into one
  multimodal sequence, and how it builds the parallel `input_mask` and `ar_mask`.
- Understand *why* the prefix uses full (bidirectional) attention.

---

## 1. Why start from PaliGemma?

π₀ is built on **PaliGemma**, a 3B open VLM = **SigLIP-So400m** vision encoder +
**Gemma-2B** LLM. The bet: web-pretrained vision-language semantics transfer to
robotics far better than learning perception from a few thousand robot demos. The
robot-specific learning is then "just" connecting those semantics to motor actions.

In `openpi`, PaliGemma is assembled inside `Pi0.__init__`
([`pi0.py:70`](../src/openpi/models/pi0.py)):
- `self.PaliGemma.img` — the SigLIP ViT ([`siglip.py`](../src/openpi/models/siglip.py)),
  `variant="So400m/14"`, `pool_type="none"` (we want **per-patch tokens**, not a pooled
  vector).
- `self.PaliGemma.llm` — the Gemma transformer ([`gemma.py`](../src/openpi/models/gemma.py)),
  configured with **two experts** (Module 03). The `gemma_2b` expert *is* the
  PaliGemma LLM; its embedder turns text token ids into vectors.

The vision encoder outputs `num_classes = paligemma_config.width` channels so its tokens
land in the LLM's embedding space directly ([`pi0.py:82`](../src/openpi/models/pi0.py)).

---

## 2. Image → tokens: the SigLIP ViT

Refresher (you know ViTs): an image is cut into patches, each patch linearly projected
to a `width`-dim vector, position embeddings added, then a stack of transformer encoder
blocks. Read the actual forward in [`siglip.py:207`](../src/openpi/models/siglip.py):

1. **Patchify** with a strided Conv (`patch_size`, `padding="VALID"`)
   ([`siglip.py:216`](../src/openpi/models/siglip.py)). For So400m/14 on 224×224:
   `224/14 = 16` patches per side → `16×16 = 256` tokens.
2. **Flatten** to `[n, 256, width]` and add a learned 2D position embedding
   ([`siglip.py:229`](../src/openpi/models/siglip.py)).
3. **Encoder**: `depth` standard pre-LN blocks (MHSA + MLP, GELU)
   ([`siglip.py:75`](../src/openpi/models/siglip.py),
   [`siglip.py:111`](../src/openpi/models/siglip.py)), optionally `nn.scan`-ed for
   compile speed/memory ([`siglip.py:126`](../src/openpi/models/siglip.py)).
4. **`pool_type="none"`** ([`siglip.py:266`](../src/openpi/models/siglip.py)): we skip
   pooling and keep all 256 patch tokens. The `head` Dense (since `num_classes` is set)
   projects them to LLM width.

So each camera contributes **~256 tokens**. With three cameras that's ~768 image tokens
before any language. This is the bulk of the prefix.

> **Why per-patch tokens, not a single pooled vector?** Manipulation needs spatial
> grounding — *where* the cup is, not just *that* there's a cup. Keeping all patches
> lets the action expert attend to specific image regions. Pooling would throw away the
> spatial layout the policy depends on.

> **Precision detail worth noticing:** patch extraction + posemb are forced to float32
> for numerical safety, then cast to `dtype_mm` (bf16) before the transformer
> ([`siglip.py:213`](../src/openpi/models/siglip.py),
> [`siglip.py:239`](../src/openpi/models/siglip.py)). Mixed precision is everywhere in
> this codebase; watch for `.astype`.

---

## 3. Text → tokens: the Gemma embedder

Language enters as token ids (from the PaliGemma SentencePiece tokenizer, Module 06).
The embedder is `gemma.Embedder` ([`gemma.py:135`](../src/openpi/models/gemma.py)):

```python
def encode(self, x):
    x = self.input_embedding_table[(x,)]   # lookup [vocab, d] -> [..., d]
    x *= jnp.sqrt(self.embed_dim)          # Gemma's embedding scale-up
    return x
```

The `sqrt(embed_dim)` scaling is a Gemma convention — it keeps embedding magnitudes
comparable to the post-attention residual stream. The vocabulary is the full PaliGemma
vocab of 257,152 ([`gemma.py:41`](../src/openpi/models/gemma.py)); π₀.₅ reuses ordinary
vocab tokens to represent discretized state (Module 06).

In `Pi0`, text embedding is invoked via the NNX bridge with `method="embed"`
([`pi0.py:129`](../src/openpi/models/pi0.py)).

---

## 4. Assembling the prefix: `embed_prefix`

Now read [`pi0.py:105`](../src/openpi/models/pi0.py) carefully. It builds three parallel
lists and concatenates them along the sequence axis:

```python
tokens, input_mask, ar_mask = [], [], []

for name in obs.images:                       # each camera view
    image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
    tokens.append(image_tokens)               # [b, 256, emb]
    input_mask.append(repeat(obs.image_masks[name], "b -> b s", s=256))
    ar_mask += [False] * 256                  # image tokens attend bidirectionally

if obs.tokenized_prompt is not None:          # language (+ discrete state for pi05)
    tokenized = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
    tokens.append(tokenized)
    input_mask.append(obs.tokenized_prompt_mask)
    ar_mask += [False] * tokenized.shape[1]   # language attends bidirectionally too

tokens     = concat(tokens, axis=1)           # [b, S_prefix, emb]
input_mask = concat(input_mask, axis=1)       # [b, S_prefix]   True = real token
ar_mask    = jnp.array(ar_mask)               # [S_prefix]       (see Module 03)
```

Three things to understand deeply:

**(a) `input_mask` vs `ar_mask` are different masks.**
- `input_mask` [b, S]: is this token *real* or padding (or an absent camera)? Used to
  zero out padding in attention.
- `ar_mask` [S]: does this token start a new *autoregressive block*? `False` means "I
  share attention scope with the previous token" → bidirectional within a block. The
  whole prefix is `ar_mask = all False` → **one big bidirectional block**. (Module 03
  shows how `make_attn_mask` turns these into a boolean attention matrix.)

**(b) The prefix is bidirectional, the suffix is not.** Perception and language are
context — every image patch can attend to every word and vice versa, like an encoder.
The action tokens (suffix) are where causality/structure is imposed. This "prefix-LM"
attention pattern is inherited from PaliGemma.

**(c) Images come first, then language.** Position ids are just `cumsum(input_mask) - 1`
([`pi0.py:208`](../src/openpi/models/pi0.py)), so ordering is fixed and contiguous over
real tokens; padding doesn't consume positions.

The return signature — `(tokens, input_mask, ar_mask)` — is the contract the loss and
sampling code rely on. Hold onto it; you'll produce the same triple in the lab.

---

## 5. Where π₀.₅ differs (preview)

For π₀.₅, the **robot state is part of `tokenized_prompt`** (discretized into the
language string), so it flows through this same prefix path — no separate state token.
For π₀, state is instead a continuous token in the *suffix* (Module 04). That's why
`embed_prefix` itself is identical for both: the divergence is upstream (tokenizer) and
downstream (suffix).

---

## Self-check

1. How many tokens does one 224×224 image become under So400m/14, and why that number?
2. Why `pool_type="none"`? What would break if we used global average pooling?
3. What does multiplying embeddings by `sqrt(embed_dim)` accomplish?
4. Distinguish `input_mask` from `ar_mask` in one sentence each.
5. Why is the entire prefix given bidirectional attention?

## Lab 02

Open [`labs/lab02_prefix.py`](labs/lab02_prefix.py). You will:
1. Implement a *tiny* patch-embed + transformer "mini-SigLIP" that maps `[b,224,224,3]`
   → `[b, N, d]` (you may downsample to keep N small).
2. Implement `embed_prefix(obs)` returning `(tokens, input_mask, ar_mask)` with the
   correct shapes and mask semantics for an arbitrary set of camera views + a text
   embedding table.
3. Validate token counts and mask values against the real `Pi0.embed_prefix` run on a
   `dummy`-variant model with `fake_obs()`.

Next: [Module 03 — The MoE transformer](03_transformer_moe_experts.md).
