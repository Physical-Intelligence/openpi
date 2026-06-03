"""Config for the Pi0TargetVLAActionMoe model (trace-free ablation).

This variant is the **trace-free counterpart** of ``Pi0TraceVLAActionMoe``: same
high-level information budget (skill + semantic target point + completion
prediction), but the trace generation pathway is completely removed. The
semantic-target keypoint that used to flow into the trace stream's AdaRMS now
modulates the action stream's AdaRMS instead.

Two streams:

  - Stream 0: PaliGemma 2B VLM (dense FFN, optional LoRA).
  - Stream 1: Action expert MoE (HardMoE FFN, K skill experts; full FT).
              Same shape as the action MoE in ``trace_vla_actionmoe`` /
              ``trace_vla_moe`` (``trace_moe_gemma_300m``: width=1024,
              mlp_dim=4096, depth=18, K=5).

There is **no Stream 2** (no trace expert). The trunk is built by reusing the
existing 3-stream-capable ``gemmoe_trace.TraceModule`` but instantiating it with
only the two configs above — the joint-attention code already supports an
arbitrary number of streams.

Conditioning on the action MoE's AdaRMS:

  ``adarms_cond_action = time_emb + tgt_emb``

where ``tgt_emb`` is the Fourier-encoded semantic-target point passed through a
2-layer MLP. This mirrors the trace stream's AdaRMS path used by
``Pi0TraceVLA`` / ``Pi0TraceVLAActionMoe`` / ``Pi0TraceVLAMoe``.

Losses (per step):

  1. Action flow-matching (single forward pass — no separate planning pass).
     - Targets ``actions`` (B, action_horizon, action_dim).
     - The action stream runs through a hard-routed MoE (skill_id -> one-hot
       broadcast over the action_horizon tokens).
  2. Completion progress regression.
     - Per-skill MLP head, mean-pooled VLM prefix hidden state — same head as
       in the TraceVLA family. Loss is masked by ``has_skill``.

There is **no trace loss** and **no inpainting clamp** because the model never
emits a trace.
"""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config as _pi0_config
from openpi.models import target_observation as _target_obs
import openpi.models.gemmoe as _gemmoe
import openpi.models.gemmoe_trace as _gemmoe_trace
from openpi.shared import array_typing as at

if TYPE_CHECKING:
    from openpi.models.pi0_target_vla_actionmoe import Pi0TargetVLAActionMoe


@dataclasses.dataclass(frozen=True)
class Pi0TargetVLAActionMoeConfig(_model.BaseModelConfig):
    """Configuration for the trace-free TargetVLA-ActionMoe ablation model.

    Two streams:
      - PaliGemma 2B (LoRA-able)
      - Action expert: 5-experts hard-routed MoE (same shape as trace_vla_actionmoe)

    The action expert receives an AdaRMS modulation that is the sum of:
      - time embedding (same Fourier sin/cos + 2-layer MLP as the action stream
        in TraceVLA / actionmoe),
      - Fourier-encoded semantic-target keypoint -> 2-layer MLP (the same
        target encoder used in TraceVLA, but its output now feeds the action
        stream instead of the trace stream).
    """

    dtype: str = "bfloat16"
    paligemma_variant: _gemmoe.Variant = "gemma_2b"
    # Action expert is an MoE (hard-routed by skill), drawn from gemmoe_trace.
    action_expert_variant: _gemmoe_trace.Variant = "trace_moe_gemma_300m"

    # Action chunk shape.
    action_dim: int = 32
    action_horizon: int = 10
    max_token_len: int = 200
    # Always pi05-style (state can live in the prompt via the tokenizer, time via adaRMS).
    pi05: bool = True
    discrete_state_input: bool = False  # match AtomicVLA / TraceVLA on libero

    # Number of skill-specific experts in the action MoE (and the completion head).
    # Must match ``num_local_experts`` of ``action_expert_variant``. Pinned to 5
    # (LIBERO skill atoms).
    num_action_experts: int = 5

    # Loss weights.
    action_loss_coeff: float = 1.0
    completion_loss_coeff: float = 0.1

    # Fourier-encoding for AdaRMS conditioning on the semantic target point.
    fourier_num_freqs: int = 8

    # Completion head: shared compression dim and per-skill hidden dim.
    completion_shared_dim: int = 256
    completion_per_skill_hidden: int = 64

    @property
    @override
    def model_type(self) -> _model.ModelType:
        # Reuse PI05 model_type for transform routing.
        return _model.ModelType.PI05

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0TargetVLAActionMoe":
        from openpi.models.pi0_target_vla_actionmoe import Pi0TargetVLAActionMoe  # noqa: PLC0415

        return Pi0TargetVLAActionMoe(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_target_obs.TargetObservation, _model.Actions]:
        # Trace-free observation: no overlay images, no trace fields, no
        # ``current_ee_xy``.
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)
        with at.disable_typechecking():
            observation_spec = _target_obs.TargetObservation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.bool_),
                token_ar_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                token_loss_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.bool_),
                atomic_token=jax.ShapeDtypeStruct([batch_size], jnp.float32),
                semantic_target_xy=jax.ShapeDtypeStruct([batch_size, 2], jnp.float32),
                has_skill=jax.ShapeDtypeStruct([batch_size], jnp.bool_),
                progress=jax.ShapeDtypeStruct([batch_size], jnp.float32),
                diffusion_loss_mask=jax.ShapeDtypeStruct([batch_size], jnp.bool_),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Freeze filter for the target_vla_actionmoe variant.

        Stream layout in the param tree:
          - paligemma (stream 0): ``llm/.../*_0`` paths (no suffix; ``_name(name, 0) == name``)
          - action expert (stream 1): ``llm/.../*_1`` paths (MoE: ``moe_1/expert_*``)

        Only two streams exist, so the freeze filter is a simplified version of
        ``Pi0TraceVLAActionMoeConfig.get_freeze_filter``:

          - For ``target_vla_actionmoe_lora`` (paligemma LoRA, action MoE full FT):
              freeze = (paligemma subtree) AND NOT (LoRA params)
                     = (all_llm AND NOT action_subtree) AND NOT lora

          - For ``target_vla_actionmoe`` (full FT everywhere): no freeze.
        """
        return _pi0_config.llm_freeze_filter(
            self.paligemma_variant, self.action_expert_variant, expert_suffixes=("_1",)
        )
