"""Config for the Pi0TraceVLAMoe model (combined MoE on both action and trace).

The "moe" variant of TraceVLA uses a hard-routed MoE on **both** non-VLM streams:

  - Stream 0: PaliGemma 2B VLM (dense FFN, optional LoRA).
  - Stream 1: Action expert MoE (HardMoE FFN, K skill experts; full FT).
              Defaults to ``trace_moe_gemma_300m`` — same size as
              ``trace_vla_actionmoe``'s action MoE.
  - Stream 2: Trace expert MoE  (HardMoE FFN, K skill experts; full FT).
              Defaults to ``trace_moe_small`` — Recipe C from
              ``traceVLA_moe_design.md`` (width=512, mlp_dim=2048, K=5).
              Shrunk along ``width``/``mlp_dim`` to keep total params manageable;
              cannot be warm-started from ``pi05_base`` because of the shape
              mismatch, so it is randomly initialized at train start.

Skill-routing is shared across both MoE streams: the dataset's
``atomic_token`` (∈ {0..4}) yields a one-hot over K=5 LIBERO skill atoms which
flows to **both** the action MoE and the trace MoE. Both K's are pinned to 5.

All other architecture, conditioning, training tricks, and losses are inherited
**unchanged** from the original ``Pi0TraceVLA`` / ``Pi0TraceVLAActionMoe``:

  - Trace flow-matching with AdaRMS conditioning on time + Fourier-encoded
    semantic target; row-0 inpainting clamp to current EE; optional appended
    semantic-target anchor row (``append_target_anchor=True`` by default).
  - Action flow-matching on the overlay-augmented image; only time conditions
    the action stream's AdaRMS.
  - Per-skill MLP completion head over mean-pooled VLM prefix hidden states.
"""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config as _pi0_config
from openpi.models import trace_observation as _trace_obs
import openpi.models.gemmoe as _gemmoe
import openpi.models.gemmoe_trace as _gemmoe_trace
from openpi.shared import array_typing as at

if TYPE_CHECKING:
    from openpi.models.pi0_trace_vla_moe import Pi0TraceVLAMoe


@dataclasses.dataclass(frozen=True)
class Pi0TraceVLAMoeConfig(_model.BaseModelConfig):
    """Configuration for the combined-MoE trace-augmented VLA model.

    Three streams:
      - PaliGemma 2B (LoRA-able)
      - Action expert MoE: 5-experts hard-routed; always full FT.
      - Trace expert  MoE: 5-experts hard-routed, shrunk-width; always full FT
        and randomly initialized (cannot be warm-started from ``pi05_base``).
    """

    dtype: str = "bfloat16"
    paligemma_variant: _gemmoe.Variant = "gemma_2b"
    # Action expert is the (full-size) MoE, drawn from gemmoe_trace.
    action_expert_variant: _gemmoe_trace.Variant = "trace_moe_gemma_300m"
    # Trace expert is the (shrunk) MoE, drawn from gemmoe_trace.
    trace_expert_variant: _gemmoe_trace.Variant = "trace_moe_small"

    # Action chunk shape.
    action_dim: int = 32
    action_horizon: int = 10
    max_token_len: int = 200
    # We always run pi05-style (state token in the prompt, time via adaRMS).
    pi05: bool = True
    discrete_state_input: bool = False  # match AtomicVLA/TraceVLA on libero

    # Original camera frame (height, width) before ``resize_with_pad`` letterboxes it
    # to the 224x224 model input. Only used by the train-time geometric augmentation in
    # ``preprocess_trace_observation`` to keep the image-space trace/keypoint targets
    # aligned with the letterboxed content. ``None`` (default) = square source / no
    # letterbox, i.e. the LIBERO behaviour. Set to e.g. ``(480, 640)`` for the
    # physical-robot table-tasks camera.
    image_source_hw: tuple[int, int] | None = None

    # Trace head shape: (N, 2). N is the number of waypoints.
    trace_horizon: int = 20
    trace_dim: int = 2
    # Number of skill-specific experts in each MoE (and the completion head).
    # Pinned to 5 (LIBERO skill atoms). Must match ``num_local_experts`` of
    # both ``action_expert_variant`` and ``trace_expert_variant``.
    num_action_experts: int = 5
    num_trace_experts: int = 5

    # When True, the trace stream is extended by one extra token whose value is
    # inpainting-clamped to the semantic target ``p_tgt`` (the same mechanism
    # already used for the current-EE clamp at row 0). Mirrors the latest
    # TraceVLA / TraceVLAActionMoe behaviour.
    append_target_anchor: bool = True

    # Loss weights.
    trace_loss_coeff: float = 1.0
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
        # Reuse the PI05 model_type for transform routing.
        return _model.ModelType.PI05

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0TraceVLAMoe":
        from openpi.models.pi0_trace_vla_moe import Pi0TraceVLAMoe  # noqa: PLC0415

        return Pi0TraceVLAMoe(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_trace_obs.TraceObservation, _model.Actions]:
        return _trace_obs.trace_inputs_spec(self, batch_size=batch_size)

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Freeze filter for the combined-MoE variant.

        Stream layout in the param tree:
          - paligemma (stream 0): ``llm/.../*_0`` paths (no suffix; ``_name(name, 0) == name``).
          - action expert (stream 1): ``llm/.../*_1`` paths (MoE: ``moe_1/expert_*``).
          - trace expert  (stream 2): ``llm/.../*_2`` paths (MoE: ``moe_2/expert_*``).

        For ``trace_vla_moe_lora`` (paligemma LoRA, both experts full FT):
            freeze = (paligemma subtree) AND NOT (LoRA params)
                   = (all_llm AND NOT action_subtree AND NOT trace_subtree) AND NOT lora

        For ``trace_vla_moe`` (full FT everywhere): no freeze.
        """
        return _pi0_config.llm_freeze_filter(
            self.paligemma_variant, self.action_expert_variant, expert_suffixes=("_1", "_2")
        )
