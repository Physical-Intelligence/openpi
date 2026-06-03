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

from typing import TYPE_CHECKING

import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp

from openpi.models import model as _model
import openpi.models.gemmoe as _gemma
import openpi.models.gemmoe_trace as _gemma_trace
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

if TYPE_CHECKING:
    from openpi.models.pi0_trace_vla_config import Pi0TraceVLAConfig


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


class TraceVLABase(_model.BaseModel):
    """Shared base for the TraceVLA model family.

    Currently holds the trunk + head construction for the canonical ``Pi0TraceVLA`` (3-stream:
    PaliGemma VLM + dense action expert + hard-routed trace MoE). Subsequent batches generalize
    this ``__init__`` (and add the shared embed/loss/sample methods) so the other variants
    (TraceVLAMoe / TraceVLAActionMoe / TargetVLAActionMoe) inherit it too.

    IMPORTANT: the submodule-construction order here is load-bearing — it fixes the order in which
    ``rngs`` is consumed, hence the random initialization of every head not loaded from pi05_base.
    Reordering changes those weights (and the loss). The attribute names are equally load-bearing:
    they are the flax param-tree paths the weight loaders and checkpoints match on.
    """

    def __init__(self, config: "Pi0TraceVLAConfig", rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.config = config
        self.pi05 = config.pi05
        if not self.pi05:
            raise ValueError("Pi0TraceVLA assumes pi05=True (adaRMS pathway).")

        self.num_trace_experts = int(config.num_trace_experts)
        self.trace_horizon = int(config.trace_horizon)
        self.trace_dim = int(config.trace_dim)
        # When the semantic-target anchor row is appended, the trace stream
        # internally carries ``trace_horizon + 1`` tokens (the extra one is
        # inpainting-clamped to ``p_tgt`` and masked from the flow loss). The
        # user-facing trace shape returned by ``sample_trace`` remains
        # ``(B, trace_horizon, 2)``.
        self.append_target_anchor = bool(getattr(config, "append_target_anchor", False))
        self.trace_seq_len = self.trace_horizon + (1 if self.append_target_anchor else 0)

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        trace_expert_config = _gemma_trace.get_trace_config(config.trace_expert_variant)
        if int(trace_expert_config.num_local_experts) != self.num_trace_experts:
            raise ValueError(
                f"trace_expert_variant has {trace_expert_config.num_local_experts} experts but "
                f"config requested {self.num_trace_experts}."
            )

        self.paligemma_width = int(paligemma_config.width)
        self.action_width = int(action_expert_config.width)
        self.trace_width = int(trace_expert_config.width)

        # ---------- Trunk: 3-stream Gemma with hard-routed trace MoE ----------
        llm = nnx_bridge.ToNNX(
            _gemma_trace.TraceModule(
                configs=[paligemma_config, action_expert_config, trace_expert_config],
                embed_dtype=config.dtype,
                # adaRMS is enabled (set per-stream below).
            )
        )
        # adaRMS on streams 1 (action) and 2 (trace).
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True, True])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)

        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        # ---------- Action expert I/O + time MLP ----------
        self.action_in_proj = nnx.Linear(config.action_dim, self.action_width, rngs=rngs)
        self.action_out_proj = nnx.Linear(self.action_width, config.action_dim, rngs=rngs)
        self.action_time_mlp_in = nnx.Linear(self.action_width, self.action_width, rngs=rngs)
        self.action_time_mlp_out = nnx.Linear(self.action_width, self.action_width, rngs=rngs)

        # ---------- Trace expert I/O ----------
        self.trace_in_proj = nnx.Linear(self.trace_dim, self.trace_width, rngs=rngs)
        self.trace_out_proj = nnx.Linear(self.trace_width, self.trace_dim, rngs=rngs)
        # Trace expert time MLP (separate parameters from the action expert's).
        self.trace_time_mlp_in = nnx.Linear(self.trace_width, self.trace_width, rngs=rngs)
        self.trace_time_mlp_out = nnx.Linear(self.trace_width, self.trace_width, rngs=rngs)

        # ---------- Semantic-target Fourier MLP for trace AdaRMS conditioning ----------
        target_fourier_dim = config.fourier_num_freqs * 2 * 2  # 2 coords * (sin+cos)
        self.target_mlp_in = nnx.Linear(target_fourier_dim, self.trace_width, rngs=rngs)
        self.target_mlp_out = nnx.Linear(self.trace_width, self.trace_width, rngs=rngs)

        # ---------- Completion head ----------
        # Shared compression from VLM hidden state width to a smaller dim.
        self.completion_shared_in = nnx.Linear(
            self.paligemma_width, config.completion_shared_dim, rngs=rngs
        )
        self.completion_shared_out = nnx.Linear(
            config.completion_shared_dim, config.completion_shared_dim, rngs=rngs
        )
        # Per-skill stacked MLP head: (K, shared_dim, hidden) -> (K, hidden, 1)
        K = self.num_trace_experts
        S = int(config.completion_shared_dim)
        H = int(config.completion_per_skill_hidden)
        rng1, rng2 = jax.random.split(rngs.params(), 2)
        std_in = (1.0 / S) ** 0.5
        std_out = (1.0 / H) ** 0.5
        self.cmp_w1 = nnx.Param(jax.random.normal(rng1, (K, S, H)) * std_in)
        self.cmp_b1 = nnx.Param(jnp.zeros((K, H)))
        self.cmp_w2 = nnx.Param(jax.random.normal(rng2, (K, H, 1)) * std_out)
        self.cmp_b2 = nnx.Param(jnp.zeros((K, 1)))

        self.deterministic = True
