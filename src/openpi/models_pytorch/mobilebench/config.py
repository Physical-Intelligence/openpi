"""Configuration for the MobileBench v1.2 additions to pi0.5.

Defaults follow the "starting configuration" table (sec. 4.4) of
MobileBench_Method_v1_2_CurrentAffordance_DualMemory.md. They are a starting point to
be validated, not tuned values.
"""

import dataclasses

# Affordance query types. Kept as fixed indices so NAV and object points are never
# mixed into one untyped coordinate (sec. 6.2).
NAV = 0
OBJ = 1
AFFORDANCE_TYPES = ("nav", "object")


@dataclasses.dataclass(frozen=True)
class MobileBenchConfig:
    # --- widths -------------------------------------------------------------------
    # Width of pi0.5's VLM tokens H_t (PaliGemma gemma_2b).
    vlm_width: int = 2048
    # Shared width of every auxiliary module (memory, decoders, workspace). Does NOT
    # shrink the pi0.5 action expert (sec. 4.4).
    d_model: int = 256
    num_heads: int = 4
    ffn_mult: int = 4
    dropout: float = 0.0

    # --- dual-timescale memory (sec. 5) -------------------------------------------
    num_fast_slots: int = 32  # N_F
    num_slow_slots: int = 16  # N_S
    writer_layers: int = 2
    # Fixed fusion coefficients M_t = (1 - a) M_{t-1} + a * candidate. Fixed alpha is
    # not a fixed writer: the writers are trainable (sec. 5.1).
    fast_alpha: float = 0.5
    slow_alpha: float = 0.5
    # The slow group is written on every K-th policy update of an episode (sec. 4.4).
    slow_write_every: int = 4

    # --- proprioception / execution / embodiment inputs ---------------------------
    state_dim: int = 32  # padded raw state S_t fed to the state encoder
    state_tokens: int = 1
    exec_dim: int = 16  # executed increment xi_t: EEF / gripper / body motion + dt
    exec_tokens: int = 1
    gripper_dim: int = 8  # static gripper description e_g
    base_desc_dim: int = 8  # base/body interface description e_r^B

    # --- affordance decoders (sec. 6.3-6.4) ---------------------------------------
    affordance_layers: int = 2

    # --- EEF goal (sec. 8.3) --------------------------------------------------------
    goal_layers: int = 2
    num_end_effectors: int = 1  # one learned query per end effector; masked if absent

    # --- workspace / goal-workspace decoder (sec. 8.2, 8.4) -------------------------
    workspace_feat_dim: int = 20  # 3 + 6 + 3 + 6 + 1 + 1
    goal_workspace_layers: int = 2
    # Workspace tokens the action heads read. None = all N_W library tokens (the
    # design as written); an int compresses them with learned-query attention
    # pooling first, to bound the cost of the extra condition tokens.
    action_workspace_tokens: int | None = None

    # --- mode decoder (sec. 8.5) ----------------------------------------------------
    num_modes: int = 2  # NAV, MANIP

    # --- training-only readouts (sec. 6.1, 7) ---------------------------------------
    trace_lags: int = 4  # number of past-time queries Q_lag for the trace readout
    phase_text_dim: int = 512  # width of the frozen text-teacher embeddings

    # --- validity gating at inference -----------------------------------------------
    # Training uses soft gating by predicted validity so the gate stays differentiable;
    # at inference a prediction below this probability is replaced by its typed NULL.
    null_threshold: float = 0.5
