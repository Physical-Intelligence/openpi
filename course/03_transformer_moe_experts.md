# Module 03 — The Mixture-of-Experts Transformer (two experts, one attention stack)

This is the structural core of π₀/π₀.₅. If you understand this module, the rest is
plumbing. Read [`gemma.py`](../src/openpi/models/gemma.py) end-to-end alongside these
notes.

## Learning objectives

- Explain what "expert" means here (it is **not** standard token-routed MoE).
- Implement multi-expert attention: per-expert QKV projections, a single shared
  attention, per-expert output projections.
- Implement the blockwise attention mask from `(input_mask, ar_mask)`.
- Explain RoPE, RMSNorm, grouped-query attention as used here.
- Understand why the action expert can be tiny (`gemma_300m`) while the VLM is `gemma_2b`.

---

## 1. "Experts" here = per-token-group weights, shared attention

Forget switch/top-k MoE. In π₀, an **expert** is a *full set of transformer weights*
applied to a *contiguous group of tokens*, where all experts **participate in one joint
self-attention**. There are exactly two:

- **Expert 0 = PaliGemma / `gemma_2b`** — processes the prefix (images + language +
  state). 2048 width, 16,384 MLP. This is the big pretrained VLM.
- **Expert 1 = action expert / `gemma_300m`** — processes the suffix (the noisy action
  tokens, and for π₀ the state token). 1024 width, 4096 MLP. Small and trained from
  scratch.

Configs at [`gemma.py:69`](../src/openpi/models/gemma.py) (`gemma_300m`) and
[`gemma.py:79`](../src/openpi/models/gemma.py) (`gemma_2b`). Both have `depth=18`,
`num_heads=8`, `num_kv_heads=1`, `head_dim=256` — they **must** share head geometry so
their Q/K/V can attend to each other ([`gemma.py:166`](../src/openpi/models/gemma.py)).

> **Why this design?** Three reasons, and you should be able to recite them:
> 1. **Protect the VLM.** The action expert attends *into* the VLM's keys/values but has
>    its own weights, so learning to denoise actions doesn't overwrite web-scale
>    semantics. (This is the seed of "knowledge insulation," Module 06.)
> 2. **Modality-specific capacity.** Continuous action denoising and language modeling
>    want different weights; experts give that without two separate networks.
> 3. **Cost.** Action tokens go through a 300M expert, not the 2B VLM. At inference the
>    big prefix is computed once and cached; only the small suffix re-runs each
>    flow-integration step (Module 05). The 7×-smaller action expert is what makes
>    high-frequency control affordable.

The naming trick that makes loading clean: expert 0's params get **no suffix**
(`attn`, `mlp`) so they load directly from a PaliGemma checkpoint; expert 1's get `_1`
(`attn_1`, `mlp_1`) and are fresh-initialized ([`gemma.py:443`](../src/openpi/models/gemma.py)).

---

## 2. The block: gated residual, optional adaRMS

`Block.__call__` ([`gemma.py:292`](../src/openpi/models/gemma.py)) is a standard pre-norm
transformer block, but generalized over the list of experts `xs = [x_prefix, x_suffix]`
(either may be `None` when that expert isn't running):

```
for each expert i with tokens x_i (if not None):
    x_i, gate_i = RMSNorm_i(x_i, adarms_cond[i])      # pre-attention norm
post_attn = Attention(pre_attn_list)                  # ONE joint attention
xs = [ x_i + post_attn_i * gate_i ]                   # gated residual
for each expert i:
    y_i, gate_i = RMSNorm_i(xs_i, adarms_cond[i])     # pre-FFN norm
    y_i = FeedForward_i(y_i)
xs = [ xs_i + y_i * gate_i ]                           # gated residual
```

- **Pre-norm + residual**: classic. The twist is the **gate** from adaRMS (Module 06):
  with `adarms_cond=None` the norm returns `gate=None` and `_gated_residual`
  ([`gemma.py:453`](../src/openpi/models/gemma.py)) reduces to plain `x + y`. With adaRMS
  (π₀.₅ suffix), the timestep modulates scale/shift *and* a residual gate.
- The whole stack of 18 blocks is run with `nn.scan` + `nn.remat`
  ([`gemma.py:359`](../src/openpi/models/gemma.py)) — gradient checkpointing to fit
  memory, layers stacked along axis 0.

### RMSNorm (and adaptive RMSNorm)

[`gemma.py:112`](../src/openpi/models/gemma.py):
```python
var = mean(x.float32 ** 2, -1, keepdims=True)
normed = x * rsqrt(var + 1e-6)
if cond is None:                       # standard RMSNorm
    scale = param("scale", zeros)      # note: learned as (1 + scale)
    return normed * (1 + scale), None
# adaptive RMSNorm: condition (the time embedding) produces scale/shift/gate
scale, shift, gate = split(Dense(3*d, zeros_init)(cond), 3)
return normed * (1 + scale) + shift, gate
```

Standard RMSNorm normalizes by RMS (no mean subtraction, unlike LayerNorm) and applies a
learned per-channel `(1+scale)`. adaRMS replaces the static scale with a *function of the
conditioning vector*, plus a shift and an output gate — this is FiLM/DiT-style
conditioning. Computing variance in float32 then casting back (note the `.astype(dtype)`)
is the recurring mixed-precision pattern.

### FeedForward

[`gemma.py:252`](../src/openpi/models/gemma.py): GeGLU — `gelu(x·W_gate) * (x·W_up)` then
`·W_down`. Standard Gemma MLP.

---

## 3. Multi-expert attention

[`gemma.py:157`](../src/openpi/models/gemma.py). The algorithm:

1. **Per-expert QKV.** For each expert that's running, project *its own* tokens with
   *its own* einsum weights ([`gemma.py:173`](../src/openpi/models/gemma.py)). Since
   `num_kv_heads(1) != num_heads(8)`, it uses separate `q_einsum` and `kv_einsum`
   (grouped-query attention; see below).
2. **Concatenate along the sequence axis**: `q, k, v = concat over experts`
   ([`gemma.py:201`](../src/openpi/models/gemma.py)). Now prefix and suffix tokens live
   in one `[B, S_total, ...]` sequence.
3. **RoPE** on q and k ([`gemma.py:203`](../src/openpi/models/gemma.py)), scale q by
   `head_dim**-0.5`.
4. **(Inference) prepend KV cache** ([`gemma.py:211`](../src/openpi/models/gemma.py)) —
   Module 05.
5. **Attention** with GQA broadcasting: `logits = einsum("BTKGH,BSKH->BKGTS")`
   ([`gemma.py:217`](../src/openpi/models/gemma.py)), mask with a huge negative
   (`-2.38e38`, the Gemma constant) where `attn_mask` is False
   ([`gemma.py:226`](../src/openpi/models/gemma.py)), softmax in float32.
6. **Per-expert output projection.** Slice the attention output back into each expert's
   token range and project with *its own* `attn_vec_einsum`
   ([`gemma.py:233`](../src/openpi/models/gemma.py)).

The key idea in one line: **QKV and output projections are per-expert; the
softmax-attention in the middle is shared.** That shared middle is exactly how action
tokens read perception/language, and (during training) how prefix tokens can be made to
*not* read actions.

### Grouped-query attention (GQA)

`num_heads=8`, `num_kv_heads=1` → one K/V head shared by all 8 query heads (`G=8` query
heads per kv head). The rearrange `"B T (K G) H -> B T K G H"` with `K=1`
([`gemma.py:216`](../src/openpi/models/gemma.py)) sets this up. GQA slashes KV-cache size
(crucial for the cached prefix at inference) at minimal quality cost.

### RoPE

[`gemma.py:424`](../src/openpi/models/gemma.py): rotary position embeddings — rotate
pairs of q/k channels by an angle proportional to position. Relative positions fall out
of the q·k dot product. Computed in float32. Positions come from `cumsum(input_mask)-1`,
so padding doesn't advance position and the suffix positions continue after the prefix
([`pi0.py:259`](../src/openpi/models/pi0.py)).

---

## 4. The blockwise attention mask

[`make_attn_mask` at `pi0.py:19`](../src/openpi/models/pi0.py) is small but subtle.
Given `input_mask [B,N]` (real vs padding) and `ar_mask [N]` (does this token start a
new AR block):

```python
mask_ar = broadcast(mask_ar, input_mask.shape)
cumsum  = jnp.cumsum(mask_ar, axis=1)
attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]   # token i may attend to j if cumsum[j] <= cumsum[i]
valid     = input_mask[:, None, :] * input_mask[:, :, None]
return attn_mask & valid
```

Read the docstring examples ([`pi0.py:19`](../src/openpi/models/pi0.py)):
- `ar_mask = [0,0,0,0,...]` (all False) → all `cumsum` equal → **everyone attends to
  everyone**: pure bidirectional. That's the prefix.
- A `True` at position k bumps the cumsum, so tokens after k can attend to ≤k but tokens
  at/after k are not visible to tokens before k. Stringing `True`s creates **causal
  blocks**.

How π₀ uses it (combine prefix + suffix ar_masks at
[`pi0.py:206`](../src/openpi/models/pi0.py)):
- **Prefix** ar_mask: all `False` → bidirectional block (images+lang+state all see each
  other).
- **Suffix** ar_mask ([`pi0.py:182`](../src/openpi/models/pi0.py)): `[True, False, False, ...]`
  — the **first** action token starts a new block (so prefix tokens *cannot* attend
  forward into the actions), and the action tokens then form one bidirectional block
  among themselves. For π₀ there's also a leading state token with `ar_mask=[True]`
  ([`pi0.py:157`](../src/openpi/models/pi0.py)).

Net effect:
```
            attends to →   image/lang/state(prefix)   actions(suffix)
prefix tokens                     YES                      NO
action tokens                     YES                      YES (each other)
```
Actions read everything; perception/language never sees the (noisy, changing) actions.
This is what lets you **cache the prefix once** and re-run only the suffix during flow
integration. Beautiful, and load-bearing.

---

## 5. The `Module.__call__` interface

[`gemma.py:388`](../src/openpi/models/gemma.py): takes `embedded` (list, one per expert,
`None` to skip), `positions`, `mask`, `adarms_cond` (list), optional `kv_cache`. Returns
`(outputs_per_expert, new_kv_cache)`. The final per-expert RMSNorm is applied at
[`gemma.py:409`](../src/openpi/models/gemma.py). This is the exact function `Pi0` calls
three different ways: full forward (both experts) for loss, prefix-only (suffix=`None`)
to fill the cache, suffix-only (prefix=`None`) per integration step.

---

## Self-check

1. In what sense is this MoE, and in what sense is it *not* token-routed MoE?
2. Which parts of attention are per-expert and which are shared? Why does that split
   enable prefix caching?
3. Walk through `make_attn_mask` for `ar_mask=[F,F,T,F]` and `input_mask=[T,T,T,F]`.
   Draw the 4×4 boolean matrix.
4. Why must both experts share `head_dim`, `num_heads`, `num_kv_heads`?
5. What does adaRMS add over standard RMSNorm, and what is the `gate` used for?
6. Why can the action expert be ~7× smaller than the VLM?

## Lab 03

Open [`labs/lab03_transformer.py`](labs/lab03_transformer.py). You will:
1. Implement `make_attn_mask` and unit-test it against the docstring examples and the
   real `pi0.make_attn_mask`.
2. Implement a single multi-expert `Block` (RMSNorm → shared attention → gated residual
   → RMSNorm → per-expert FFN), using the `dummy` config, and confirm output shapes for
   a two-expert input `[x_prefix, x_suffix]`.
3. (Stretch) wire `nn.scan` over the block and reproduce the `dummy` Gemma `Module`
   forward on random tokens; compare to the real module's output shapes and that masking
   makes prefix outputs independent of suffix contents.

Next: [Module 04 — Flow-matching action head](04_flow_matching_action_head.md).
