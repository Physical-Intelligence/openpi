"""Lab 02 — Vision-language backbone & the prefix.

Build a tiny patch-embed "mini-SigLIP" and an embed_prefix that returns
(tokens, input_mask, ar_mask) with correct shapes and mask semantics.

Run:  uv run python course/labs/lab02_prefix.py
"""

import flax.linen as nn
import jax
import jax.numpy as jnp


# ----------------------------------------------------------------------------
# Part 1: mini-SigLIP image encoder (Module 02 §2)
# ----------------------------------------------------------------------------
class MiniSiglip(nn.Module):
    width: int = 128
    patch: int = 32          # bigger patch -> fewer tokens; 224/32=7 -> 49 tokens
    depth: int = 2
    num_heads: int = 2

    @nn.compact
    def __call__(self, image):  # image: [b, 224, 224, 3] in [-1,1]
        # TODO(you):
        # 1. strided Conv (patch, patch) -> [b, h, w, width]; reshape to [b, h*w, width]
        # 2. add a learned position embedding
        # 3. `depth` pre-LN transformer blocks (MHSA + MLP)
        # 4. return per-patch tokens [b, h*w, width]  (NO pooling -> pool_type="none")
        raise NotImplementedError


# ----------------------------------------------------------------------------
# Part 2: embed_prefix (Module 02 §4)
# ----------------------------------------------------------------------------
def embed_prefix(img_encoder, params, text_embed_table, images, image_masks,
                 tokenized_prompt, tokenized_prompt_mask):
    """Replicate Pi0.embed_prefix.

    Returns (tokens [b,S,d], input_mask [b,S], ar_mask [S]).
    - For each image view: encode -> tokens; input_mask = broadcast view mask;
      ar_mask extends with False per image token (bidirectional).
    - Then language: embed ids via lookup in text_embed_table; ar_mask False.
    Concatenate along seq axis.
    """
    # TODO(you): implement following Module 02 §4 pseudo-code.
    raise NotImplementedError


def check():
    rng = jax.random.key(0)
    enc = MiniSiglip()
    img = jnp.zeros((2, 224, 224, 3))
    params = enc.init(rng, img)
    tokens = enc.apply(params, img)
    print("image tokens:", tokens.shape)  # expect [2, 49, 128]
    assert tokens.ndim == 3

    # Two views + a 5-token prompt, with the 2nd view masked out for batch elem 1.
    d = enc.width
    images = {"base": img, "wrist": img}
    image_masks = {"base": jnp.array([True, True]), "wrist": jnp.array([True, False])}
    vocab = jax.random.normal(rng, (100, d))
    prompt = jnp.array([[1, 2, 3, 4, 5], [1, 2, 3, 0, 0]])
    prompt_mask = jnp.array([[True] * 5, [True, True, True, False, False]])

    toks, imask, armask = embed_prefix(
        enc, params, vocab, images, image_masks, prompt, prompt_mask
    )
    n_img = tokens.shape[1]
    expected_S = 2 * n_img + 5
    assert toks.shape == (2, expected_S, d), toks.shape
    assert imask.shape == (2, expected_S)
    assert armask.shape == (expected_S,)
    assert not bool(armask.any()), "prefix ar_mask must be all-False (bidirectional)"
    # masked-out wrist view for batch elem 1 should be input_mask=False
    assert not bool(imask[1, n_img:2 * n_img].any()), "masked view must be input_mask=0"
    print("embed_prefix OK: shapes + mask semantics correct.")


if __name__ == "__main__":
    check()
    print("\nCross-check: build a dummy-variant Pi0 and compare token COUNT per image "
          "(So400m/14 gives 256; your mini gives 49 — both 'pool_type=none' per-patch).")
