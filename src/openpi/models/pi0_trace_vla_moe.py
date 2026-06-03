"""Pi0TraceVLAMoe: trace-augmented VLA with MoE on BOTH the action head and the trace head.

Three-stream Gemma trunk:

  - Stream 0: PaliGemma 2B VLM (dense FFN, optional LoRA).
  - Stream 1: action expert (HardMoE FFN, K skill experts, no shared expert; full FT).
              Same shape as ``trace_vla_actionmoe``'s action MoE (default
              ``trace_moe_gemma_300m``: width=1024, mlp_dim=4096, depth=18).
  - Stream 2: trace  expert (HardMoE FFN, K skill experts, no shared expert; full FT).
              Shrunk along ``width``/``mlp_dim`` (default ``trace_moe_small``:
              width=512, mlp_dim=2048, depth=18) — Recipe C from
              ``traceVLA_moe_design.md``. Randomly initialized (cannot be
              warm-started from ``pi05_base`` because of the shape mismatch).

This is the combined mirror of ``Pi0TraceVLA`` and ``Pi0TraceVLAActionMoe``: same
dataflow, same conditioning (semantic-target Fourier+AdaRMS, EE inpainting clamp,
optional appended target-anchor row), same dataset, same training tricks
(anchor-age augmentation, scene/overlay dropout, trace perturbation, image
augmentation), and the same three losses (action, trace, completion). The only
architectural change is that both non-VLM streams now carry hard-routed MoE
FFNs and route the same skill-id one-hot.

Per-pass MoE activity (only one MoE stream is ever active at a time):

  - ``_forward_planning``  : trace stream is active   → trace MoE consumes
                              ``combine_weights`` of shape ``(B, trace_seq_len, K)``.
  - ``_forward_execution`` : action stream is active  → action MoE consumes
                              ``combine_weights`` of shape ``(B, action_horizon, K)``.
  - prefix prefill (sampling endpoints): neither expert stream has tokens, so
    a small placeholder ``combine_weights`` of shape ``(B, 1, K)`` is enough.
"""
from __future__ import annotations

from typing_extensions import override

import openpi.models.gemmoe as _gemma
import openpi.models.gemmoe_trace as _gemma_trace
from openpi.models.pi0_trace_vla_base import TraceVLABase


# ---------------------------------------------------------------------------
# Pi0TraceVLAMoe model
# ---------------------------------------------------------------------------

class Pi0TraceVLAMoe(TraceVLABase):
    """Trace-augmented VLA with MoE on both the action and the trace heads.

    Inherits everything from ``TraceVLABase`` (trunk/head construction, embeds, forward passes,
    loss, and sampling); the only override is ``_build_expert_configs`` — both expert streams are
    MoE, routed by the same skill one-hot.
    """

    @override
    def _build_expert_configs(self, config):
        """Both the action and the trace stream are hard-routed MoE (``gemmoe_trace`` configs),
        routed by the *same* skill-id one-hot. (The base builds the identical trunk/head set; only
        the action-expert config source differs from TraceVLA.)

        Supported configuration: ``num_action_experts == num_trace_experts``. Because both streams
        share the same skill one-hot, we collapse them into a single ``num_skills``; the config's
        separate ``num_action_experts`` / ``num_trace_experts`` fields are expected to be equal. We
        don't assert that directly — instead the per-stream ``num_local_experts`` checks below
        require each expert config to carry exactly ``num_skills`` experts, so an unequal
        ``num_trace_experts`` is rejected there.
        """
        # Take the action-expert count as the shared skill count. The trace check below then
        # requires the trace MoE to carry the same number, so unequal action/trace counts are not
        # silently accepted.
        self.num_skills = int(config.num_action_experts)
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        # Both expert streams are MoE; both come from gemmoe_trace's variants.
        action_expert_config = _gemma_trace.get_trace_config(config.action_expert_variant)
        trace_expert_config = _gemma_trace.get_trace_config(config.trace_expert_variant)

        if int(action_expert_config.num_local_experts) != self.num_skills:
            raise ValueError(
                f"action_expert_variant has {action_expert_config.num_local_experts} experts but "
                f"config requested {self.num_skills}."
            )
        if int(trace_expert_config.num_local_experts) != self.num_skills:
            raise ValueError(
                f"trace_expert_variant has {trace_expert_config.num_local_experts} experts but "
                f"config requested {self.num_skills}."
            )
        return paligemma_config, action_expert_config, trace_expert_config
