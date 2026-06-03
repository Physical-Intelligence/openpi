"""Shared base for the TraceVLA model family.

Home of the module-level helpers and (in later batches) ``TraceVLABase``, the shared trunk/head
construction and embed/loss/sample machinery for:

  - Pi0TraceVLA          (trace stream = MoE, action stream = dense)
  - Pi0TraceVLAMoe       (both streams MoE)
  - Pi0TraceVLAActionMoe (action stream = MoE, trace stream = dense)
  - Pi0TargetVLAActionMoe (no trace stream; target conditions the action MoE)

These helpers were previously copy-pasted (identically, modulo docstrings) into each of the four
model files; they are pure functions and carry no parameters, so this move does not touch the
flax param tree.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from openpi.shared import array_typing as at


def make_attn_mask(input_mask, mask_ar):
    """Build a (B, S, S) attention mask from a token mask and an autoregressive mask.

    ``mask_ar`` marks tokens that may only attend to earlier tokens; runs of equal cumulative
    value form mutually-visible blocks. (Same construction as ``pi0_atomic.make_attn_mask``.)
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Sin/cos positional embedding for scalar positions (matches pi0/pi05)."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")
    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


def fourier_encode_2d(p: at.Float[at.Array, "*b 2"], num_freqs: int) -> at.Float[at.Array, "*b feat"]:
    """Map 2-D points (already in [0, 1]^2) to a Fourier feature vector.

    Returns ``(*b, 2 * 2 * num_freqs)`` features: for each of the 2 coords, sin and cos at
    ``num_freqs`` geometric frequencies. The trace/target configs all reuse this exact form so the
    AdaRMS conditioning signal matches across the family in both spectrum and scale.
    """
    freqs = 2.0 ** jnp.arange(num_freqs, dtype=jnp.float32)  # (num_freqs,)
    angles = p[..., None] * freqs * 2.0 * jnp.pi  # (..., 2, num_freqs)
    sins = jnp.sin(angles)
    coss = jnp.cos(angles)
    feat = jnp.concatenate([sins, coss], axis=-1)  # (..., 2, 2*num_freqs)
    new_shape = (*feat.shape[:-2], feat.shape[-2] * feat.shape[-1])
    return feat.reshape(new_shape)
