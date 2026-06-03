"""Config for the Pi0TraceVLA model."""
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
    from openpi.models.pi0_trace_vla import Pi0TraceVLA


@dataclasses.dataclass(frozen=True)
class Pi0TraceVLAConfig(_model.BaseModelConfig):
    """Configuration for the trace-augmented VLA model.

    Three streams:
      - PaliGemma 2B (LoRA-able)
      - Action expert gemma_300m (LoRA-able)
      - Trace expert (5-experts hard-routed MoE; full FT)

    The model jointly trains an action flow-matching head, a trace flow-matching head
    (over normalized 2-D pixel coords), and a per-skill MLP completion-progress head.
    """

    dtype: str = "bfloat16"
    paligemma_variant: _gemmoe.Variant = "gemma_2b"
    action_expert_variant: _gemmoe.Variant = "gemma_300m"
    trace_expert_variant: _gemmoe_trace.Variant = "trace_moe_gemma_300m"

    # Action chunk shape.
    action_dim: int = 32
    action_horizon: int = 10
    max_token_len: int = 200
    # We always run pi05-style (state token in the prompt, time via adaRMS).
    pi05: bool = True
    discrete_state_input: bool = False  # AtomicVLA uses False for libero; keep consistent.

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
    # Number of skill-specific experts in the trace MoE.
    num_trace_experts: int = 5

    # When True, the trace stream is extended by one extra token whose value is
    # inpainting-clamped to the semantic target ``p_tgt`` (the same mechanism
    # already used for the current-EE clamp at row 0). This gives the trace
    # generator a hard, spatial anchor for the target — complementary to the
    # AdaRMS modulation pathway. The supervised flow-matching target for the
    # extra row is constructed by appending ``p_tgt`` to the dataset's
    # ``future_trace_xy``; the flow loss is masked at this extra row (just as
    # it is at row 0).
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
        # Reuse existing PI05 model_type for transform routing; the new model has
        # its own training script so this is mostly cosmetic.
        return _model.ModelType.PI05

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0TraceVLA":
        from openpi.models.pi0_trace_vla import Pi0TraceVLA  # noqa: PLC0415

        return Pi0TraceVLA(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_trace_obs.TraceObservation, _model.Actions]:
        return _trace_obs.trace_inputs_spec(self, batch_size=batch_size)

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Same logic as Pi0AtomicConfig but considers our 3-stream layer naming.

        Streams in our model:
          - paligemma : VLM, llm.../*_0 paths
          - action exp: llm.../*_1 paths
          - trace exp : llm.../*_2 paths (HardMoE - we DO NOT LoRA-freeze this stream)

        For LoRA finetune (our `trace_vla_lora`):
          - VLM (paligemma) is LoRA-frozen (variant ``gemma_2b_lora``)
          - Action expert is LoRA-frozen (variant ``gemma_300m_lora``)
          - Trace expert is *fully trainable* — its FFN is the new MoE we add and we want
            to learn it from scratch / pi05 init.
        """
        return _pi0_config.llm_freeze_filter(
            self.paligemma_variant, self.action_expert_variant, expert_suffixes=("_1", "_2")
        )
