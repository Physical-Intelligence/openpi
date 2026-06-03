"""Pi0TraceVLAActionMoe: trace-augmented VLA with MoE action head + single trace head.

Three-stream Gemma trunk:

  - Stream 0: PaliGemma 2B VLM (dense FFN, optional LoRA).
  - Stream 1: action expert (HardMoE FFN, K skill experts, no shared expert; full FT).
  - Stream 2: trace expert (dense FFN, ``gemma_300m``-shape; full FT).

This is the "actionmoe" mirror of ``Pi0TraceVLA``: same dataflow, same conditioning,
same training tricks, same losses (action, trace, completion) — the only architectural
change is *which* of the two non-VLM streams carries the MoE FFN. The hard-routing
key is unchanged (skill id -> one-hot over K experts), and so is the per-skill
completion head.

Three losses per step:

  1. Trace flow-matching (planning forward pass: clean image, no trace overlay).
     - Targets ``future_trace_xy`` (normalized [0,1] pixel coords, (B, N, 2)).
     - Conditioning:
         · time embedding via adaRMS (existing pi05 pathway, on the trace stream).
         · Fourier-encoded semantic target point added to the same adaRMS cond.
         · Hard inpainting clamp on ``x_t[:, 0, :]`` to the current EE pixel.
         · When ``append_target_anchor`` is True, an extra row is appended whose
           value is inpainting-clamped to the semantic target point.
     - Stream 2 here is a *single dense FFN* (no MoE routing on the trace stream).

  2. Action flow-matching (execution forward pass: image already overlaid with the
     GT trace).
     - Targets ``actions`` (B, action_horizon, action_dim).
     - The action stream now runs through a hard-routed MoE: ``skill_id -> one-hot``
       broadcast over ``action_horizon`` tokens; every block routes every token to
       the chosen expert FFN. No conditioning beyond time (adaRMS).

  3. Completion progress regression (execution forward pass).
     - Same per-skill MLP head as TraceVLA, mean-pooled VLM prefix hidden state.
"""
from __future__ import annotations

from typing_extensions import override

import openpi.models.gemmoe as _gemma
import openpi.models.gemmoe_trace as _gemma_trace
from openpi.models.pi0_trace_vla_base import TraceVLABase


# ---------------------------------------------------------------------------
# Pi0TraceVLAActionMoe model
# ---------------------------------------------------------------------------

class Pi0TraceVLAActionMoe(TraceVLABase):
    """Trace-augmented VLA with an MoE action head and a single (dense) trace head.

    Inherits everything from ``TraceVLABase`` (trunk/head construction, embeds, forward passes,
    loss, and sampling); the only override is ``_build_expert_configs`` — action stream is the MoE,
    trace stream is a single dense FFN.
    """

    @override
    def _build_expert_configs(self, config):
        """Action expert is the MoE (``gemmoe_trace`` config); the trace expert is a single dense
        FFN (``gemma`` config, ``num_local_experts=1``). Only the action stream is skill-routed, so
        ``num_skills = num_action_experts``; the base builds the (3-stream) trunk + heads.
        """
        self.num_skills = int(config.num_action_experts)
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        # Action expert is the MoE; pulled from gemmoe_trace's variants.
        action_expert_config = _gemma_trace.get_trace_config(config.action_expert_variant)
        # Trace expert is a single dense FFN; pulled from gemmoe's variants.
        trace_expert_config = _gemma.get_config(config.trace_expert_variant)
        if int(action_expert_config.num_local_experts) != self.num_skills:
            raise ValueError(
                f"action_expert_variant has {action_expert_config.num_local_experts} experts but "
                f"config requested {self.num_skills}."
            )
        if int(getattr(trace_expert_config, "num_local_experts", 1)) > 1:
            raise ValueError(
                f"trace_expert_variant must be a single dense FFN (num_local_experts=1) for the "
                f"actionmoe variant; got num_local_experts={trace_expert_config.num_local_experts}."
            )
        return paligemma_config, action_expert_config, trace_expert_config

    # The forward passes and sampling endpoints are inherited from TraceVLABase. For this
    # variant trace_is_moe=False (dense trace stream -> dummy combine weights during planning /
    # sample_trace) and action_is_moe=True (real one-hot routing during execution / action
    # sampling); the base selects each path from those flags.
