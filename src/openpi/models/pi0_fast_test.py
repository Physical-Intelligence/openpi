import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
import openpi.models.pi0_fast as pi0_fast
from openpi.shared import nnx_utils


class _FakeImageEncoder(nnx.Module):
    """Stands in for SigLIP so the test runs in seconds on CPU.

    Returns a fixed set of image tokens, independent of the image content. The decoding logic under test only cares
    that the image tokens occupy positions in the prefix, not what they contain.
    """

    def __init__(self, width: int, num_tokens: int = 4):
        self.tokens = nnx.Variable(jax.random.normal(jax.random.key(123), (num_tokens, width)) * 0.5)

    def __call__(self, images: jax.Array, train: bool = False):  # noqa: FBT001, FBT002
        return jnp.broadcast_to(self.tokens.value[None], (images.shape[0], *self.tokens.value.shape)), None


def _create_model(config: pi0_fast.Pi0FASTConfig, key: jax.Array, param_scale: float) -> pi0_fast.Pi0FAST:
    """Create a small model with random parameters.

    The token embedding table is zero-initialized, so a freshly created model produces identical logits for every
    token and every position. Random parameters make the logits depend on both, which is what the test needs.
    """
    model = config.create(key)
    model.PaliGemma.img = _FakeImageEncoder(model.PaliGemma.llm.module.width)

    graphdef, state = nnx.split(model)
    leaves, treedef = jax.tree.flatten(state)
    keys = jax.random.split(key, len(leaves))
    leaves = [
        jax.random.normal(k, leaf.shape, leaf.dtype) * param_scale if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf
        for k, leaf in zip(keys, leaves, strict=True)
    ]
    return nnx.merge(graphdef, jax.tree.unflatten(treedef, leaves))


def _make_observation(config: pi0_fast.Pi0FASTConfig, prompt_lens: list[int], key: jax.Array) -> _model.Observation:
    """Build an observation whose tokenized prompts have different lengths per batch element."""
    batch_size = len(prompt_lens)
    obs = config.fake_obs(batch_size)
    # Avoid token 0 (padding) and the EOS token so that decoding never stops early.
    tokens = jax.random.randint(key, (batch_size, config.max_token_len), 2, 1000, dtype=jnp.int32)
    valid = jnp.arange(config.max_token_len)[None, :] < jnp.asarray(prompt_lens)[:, None]
    return dataclasses.replace(
        obs,
        tokenized_prompt=jnp.where(valid, tokens, 0),
        tokenized_prompt_mask=valid,
        # The whole prompt is the prefix: tokens attend to each other bidirectionally.
        token_ar_mask=jnp.zeros_like(tokens),
        token_loss_mask=jnp.zeros_like(valid),
    )


def _reference_greedy_decode(
    model: pi0_fast.Pi0FAST, obs: _model.Observation, prompt_lens: list[int], num_steps: int
) -> jax.Array:
    """Greedy decoding without a KV cache.

    Every step re-runs the full forward pass used by `compute_loss`, i.e. with the default contiguous positions
    (`arange(seq_len)`). Because valid tokens are left-aligned, this is exactly what the model sees during training,
    which makes it the ground truth that the cached decoding loop in `sample_actions` must reproduce.
    """
    obs = _model.preprocess_observation(None, obs, train=False, image_keys=list(obs.images.keys()))
    batch_idx = jnp.arange(len(prompt_lens))
    lens = jnp.asarray(prompt_lens)

    @nnx.jit
    def next_token_logits(model, obs, token_index):
        embeddings, input_mask, ar_mask = model.embed_inputs(obs)
        attn_mask = pi0_fast.make_attn_mask(input_mask, ar_mask)
        logits, _, _ = model.PaliGemma.llm(embedded_prefix=embeddings, mask=attn_mask)
        num_image_tokens = embeddings.shape[1] - obs.tokenized_prompt.shape[1]
        return logits[batch_idx, num_image_tokens + token_index]

    tokens, mask, ar_mask = obs.tokenized_prompt, obs.tokenized_prompt_mask, obs.token_ar_mask
    generated = []
    for step in range(num_steps):
        current = dataclasses.replace(obs, tokenized_prompt=tokens, tokenized_prompt_mask=mask, token_ar_mask=ar_mask)
        # The last valid token predicts the next one.
        next_token = jnp.argmax(next_token_logits(model, current, lens + step - 1), axis=-1).astype(jnp.int32)
        generated.append(next_token)
        # Append the generated token as a causal (ar_mask=1) token, mirroring the training layout.
        slot = (batch_idx, lens + step)
        tokens = tokens.at[slot].set(next_token)
        mask = mask.at[slot].set(True)
        ar_mask = ar_mask.at[slot].set(1)
    return jnp.stack(generated, axis=1)


def test_sample_actions_matches_uncached_decoding():
    """The cached decoding loop must assign the same positions as training.

    Regression test for an off-by-one in the RoPE positions of generated tokens: the token sampled at `step` must get
    position `prefill_len + step`, directly after the previous token, not `prefill_len + step + 1`. The first sampled
    token is unaffected (it comes from the prefix logits), so several steps are decoded.
    """
    config = pi0_fast.Pi0FASTConfig(paligemma_variant="dummy", max_token_len=16)
    # Parameters large enough that attention is sharp, so a position shift changes the greedy tokens.
    model = _create_model(config, jax.random.key(0), param_scale=0.5)

    # Different prompt lengths exercise the right-alignment in `sample_actions`.
    prompt_lens = [5, 9]
    num_steps = 4
    obs = _make_observation(config, prompt_lens, jax.random.key(1))

    expected = _reference_greedy_decode(model, obs, prompt_lens, num_steps)
    sample_actions = nnx_utils.module_jit(model.sample_actions, static_argnames=("max_decoding_steps", "temperature"))
    actual = sample_actions(jax.random.key(2), obs, max_decoding_steps=num_steps, temperature=0.0)

    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
