"""Config for the Pi0TraceVLAActionMoe model.

The "actionmoe" variant of TraceVLA swaps the location of the hard-routed MoE:

  - Stream 0: PaliGemma 2B VLM (dense FFN, optional LoRA).
  - Stream 1: Action expert MoE (HardMoE FFN, K skill experts, no shared expert; always full FT).
  - Stream 2: Trace expert single (dense FFN, like ``gemma_300m``; always full FT).

The MoE that used to denoise traces in TraceVLA now denoises actions; the single
dense head that used to denoise actions now denoises traces. All conditioning,
inpainting clamps, completion head, dataset, and training tricks are otherwise
inherited unchanged from the TraceVLA design.
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
    from openpi.models.pi0_trace_vla_actionmoe import Pi0TraceVLAActionMoe


@dataclasses.dataclass(frozen=True)
class Pi0TraceVLAActionMoeConfig(_model.BaseModelConfig):
    """Configuration for the trace-augmented VLA model with MoE action head.

    Three streams:
      - PaliGemma 2B (LoRA-able)
      - Action expert: 5-experts hard-routed MoE; always full FT
      - Trace expert: gemma_300m-shape single dense FFN; always full FT

    The model jointly trains an action flow-matching head (MoE), a trace
    flow-matching head (single dense, over normalized 2-D pixel coords), and
    a per-skill MLP completion-progress head.
    """

    dtype: str = "bfloat16"
    paligemma_variant: _gemmoe.Variant = "gemma_2b"
    # Action expert is now an MoE (hard-routed by skill), drawn from gemmoe_trace.
    action_expert_variant: _gemmoe_trace.Variant = "trace_moe_gemma_300m"
    # Trace expert is now a single dense FFN, drawn from gemmoe.
    trace_expert_variant: _gemmoe.Variant = "gemma_300m"

    # Action chunk shape.
    action_dim: int = 32
    action_horizon: int = 10
    max_token_len: int = 200
    # We always run pi05-style (state token in the prompt, time via adaRMS).
    pi05: bool = True
    discrete_state_input: bool = False  # match AtomicVLA/TraceVLA on libero

    # Trace head shape: (N, 2). N is the number of waypoints.
    trace_horizon: int = 20
    trace_dim: int = 2
    # Number of skill-specific experts in the action MoE (and the completion head).
    # Must match ``num_local_experts`` of ``action_expert_variant``.
    num_action_experts: int = 5

    # When True, the trace stream is extended by one extra token whose value is
    # inpainting-clamped to the semantic target ``p_tgt`` (the same mechanism
    # already used for the current-EE clamp at row 0). This gives the trace
    # generator a hard, spatial anchor for the target — complementary to the
    # AdaRMS modulation pathway. The supervised flow-matching target for the
    # extra row is constructed by appending ``p_tgt`` to the dataset's
    # ``future_trace_xy``; the flow loss is masked at this extra row (just as
    # it is at row 0). Same behavior as the latest TraceVLA.
    append_target_anchor: bool = True

    # Loss weights.
    trace_loss_coeff: float = 1.0
    action_loss_coeff: float = 1.0
    completion_loss_coeff: float = 0.1

    # Fourier-encoding for AdaRMS conditioning on the semantic target point
    # (applied to the trace stream, exactly as in TraceVLA).
    fourier_num_freqs: int = 8

    # Completion head: shared compression dim and per-skill hidden dim.
    completion_shared_dim: int = 256
    completion_per_skill_hidden: int = 64

    @property
    @override
    def model_type(self) -> _model.ModelType:
        # Reuse the existing PI05 model_type for transform routing.
        return _model.ModelType.PI05

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0TraceVLAActionMoe":
        from openpi.models.pi0_trace_vla_actionmoe import Pi0TraceVLAActionMoe  # noqa: PLC0415

        return Pi0TraceVLAActionMoe(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_trace_obs.TraceObservation, _model.Actions]:
        return _trace_obs.trace_inputs_spec(self, batch_size=batch_size)

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Freeze filter for the actionmoe variant.

        Stream layout in the param tree:
          - paligemma (stream 0): llm/.../*_0 paths (no suffix; ``_name(name, 0) == name``)
          - action expert (stream 1): llm/.../*_1 paths (now MoE: ``moe_1/expert_*``)
          - trace expert  (stream 2): llm/.../*_2 paths (now dense: ``mlp_2``)

        For ``trace_vla_actionmoe_lora`` (paligemma LoRA, others full FT):
          freeze = (paligemma subtree) AND NOT (LoRA params)
                 = (all_llm AND NOT action_subtree AND NOT trace_subtree) AND NOT lora

        For ``trace_vla_actionmoe`` (full FT everywhere): no freeze.
        """
        return _pi0_config.llm_freeze_filter(
            self.paligemma_variant, self.action_expert_variant, expert_suffixes=("_1", "_2")
        )
