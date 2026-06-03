"""Pi0TraceVLA: trace-augmented Vision-Language-Action model.

Three-stream Gemma trunk:

  - Stream 0: PaliGemma 2B VLM (dense FFN, optional LoRA).
  - Stream 1: gemma_300m action expert (dense FFN, optional LoRA, adaRMS time emb).
  - Stream 2: trace expert (HardMoE FFN, K skill experts, no shared expert; full FT).

The model trains three losses jointly per step:

  1. Trace flow-matching (planning forward pass: clean image, no trace overlay).
     - Targets `future_trace_xy` (normalized [0,1] pixel coords, (B, N, 2)).
     - Conditioning:
         · time embedding via adaRMS (existing pi05 pathway, on the trace stream).
         · Fourier-encoded semantic target point added to the same adaRMS cond.
         · Hard inpainting clamp on `x_t[:, 0, :]` to the current EE pixel.
     - Skill -> hard one-hot MoE routing (no learned router).

  2. Action flow-matching (execution forward pass: image already overlaid with the GT trace).
     - Targets `actions` (B, action_horizon, action_dim), like pi05/AtomicVLA.

  3. Completion progress regression (execution forward pass).
     - Mean-pool VLM prefix hidden states -> shared compression -> per-skill MLP -> sigmoid.
     - Target is `obs.progress` in [0, 1].

Both forward passes share the *same* prefix structure (images + language + state) but
differ in (a) which suffix stream is active and (b) whether the base image carries
the trace overlay.
"""
from __future__ import annotations

from openpi.models.pi0_trace_vla_base import TraceVLABase


# ---------------------------------------------------------------------------
# Pi0TraceVLA model
# ---------------------------------------------------------------------------

class Pi0TraceVLA(TraceVLABase):
    """Trace-augmented Vision-Language-Action model.

    See the module docstring for an architectural summary. This is the canonical variant —
    trace stream is the hard-routed MoE, action stream is a single dense FFN — so it adds
    nothing to ``TraceVLABase``: the base's default ``_build_expert_configs`` already builds
    exactly this configuration, and all embeds / forward passes / loss / sampling are inherited.
    """
