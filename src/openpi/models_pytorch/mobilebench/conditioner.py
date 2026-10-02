"""One MobileBench policy update, up to (not including) the action heads.

Follows the forward order of design doc sec. 4.1 / 9.7 / 10.4:

    H_t (from pi0.5)                                  -- current image, ORIGINAL instruction, state
    W_t, s_t, e_g, e_r^B = encoders(robot desc, state)
    A_obs   = CurrentAffordanceDecoder(H, s)          -- current evidence only
    M^F, M^S = DualMemory(...)                        -- fast every update, slow every K-th
    A_use   = JointAffordanceDecoder(H, M^F, M^S, s, A_obs)
    G       = EEFGoalDecoder(H, M^F, M^S, s, e_g, E_O(A_use.object))
    C^G     = GoalWorkspaceDecoder(G, W, H, M^F, M^S, s, e_g)
    m       = ModeDecoder(H, M^F, M^S, s, A_use, G, C^G)
    cond    = typed tokens [M^F ; M^S ; s ; W ; C^G ; E(A_use) ; E_G(G) ; E_m(m) ; e_g ; e_r^B]

Every online intermediate is a model PREDICTION; labels only enter the losses
(sec. 9.1). H_t is not re-emitted in `cond`: the action expert already attends to it
natively as pi0.5's prefix, which keeps the "direct H_t path" of sec. 4.2. The memory
writers never read A_obs (sec. 6.4: the current branch is not a precondition of the
memory write).

Robots without an arm pass ws_mask all False and ee_mask all False: workspace tokens
are then masked out and the goal / capability tokens become their typed NULLs.
"""

import dataclasses

import torch
from torch import Tensor
from torch import nn

from openpi.models_pytorch.mobilebench.affordance import AffordancePred
from openpi.models_pytorch.mobilebench.affordance import CurrentAffordanceDecoder
from openpi.models_pytorch.mobilebench.affordance import JointAffordanceDecoder
from openpi.models_pytorch.mobilebench.affordance import PointEncoder
from openpi.models_pytorch.mobilebench.config import OBJ
from openpi.models_pytorch.mobilebench.config import MobileBenchConfig
from openpi.models_pytorch.mobilebench.goal import EEFGoalDecoder
from openpi.models_pytorch.mobilebench.goal import GoalPred
from openpi.models_pytorch.mobilebench.goal import PoseEncoder
from openpi.models_pytorch.mobilebench.heads import ModeDecoder
from openpi.models_pytorch.mobilebench.heads import PhaseTextReadout
from openpi.models_pytorch.mobilebench.heads import TraceReadout
from openpi.models_pytorch.mobilebench.layers import QueryDecoder
from openpi.models_pytorch.mobilebench.layers import gelu_mlp_tokens
from openpi.models_pytorch.mobilebench.layers import mlp
from openpi.models_pytorch.mobilebench.memory import DualMemory
from openpi.models_pytorch.mobilebench.memory import MemoryState
from openpi.models_pytorch.mobilebench.workspace import GoalWorkspaceDecoder
from openpi.models_pytorch.mobilebench.workspace import WorkspaceEncoder
from openpi.models_pytorch.mobilebench.workspace import workspace_features

# Source ids for the typed condition-token stream the action heads read.
SOURCES = ("fast", "slow", "state", "workspace", "capability", "affordance", "goal", "mode", "gripper", "base")
SRC = {name: i for i, name in enumerate(SOURCES)}


@dataclasses.dataclass
class StepInputs:
    h: Tensor  # [B, N_H, vlm_width] pi0.5 VLM tokens of the CURRENT observation
    h_mask: Tensor  # [B, N_H] True = valid
    state: Tensor  # [B, state_dim] raw proprioception in a fixed per-robot layout
    state_mask: Tensor  # [B, state_dim] True = this slot exists on this robot
    exec_increment: Tensor  # [B, exec_dim] actually-executed motion since the last update (+ dt)
    exec_mask: Tensor  # [B, exec_dim]
    dt: Tensor  # [B] seconds since the previous policy update
    gripper_desc: Tensor  # [B, gripper_dim]  static gripper / TCP description
    base_desc: Tensor  # [B, base_desc_dim] body-motion interface description
    eef_pos: Tensor  # [B, 3]  current TCP pose in the body frame B_t
    eef_rot: Tensor  # [B, 3, 3]
    lib_pos: Tensor  # [B, N_W, 3]  reachable-pose library (fixed per robot)
    lib_rot: Tensor  # [B, N_W, 3, 3]
    lib_cost: Tensor  # [B, N_W]   c_i (see workspace.joint_space_cost)
    lib_margin: Tensor  # [B, N_W]  mu_i
    ws_mask: Tensor  # [B, N_W]  True = valid library sample (all False for arm-less robots)
    ee_mask: Tensor  # [B, E]    True = end effector exists
    new_episode: Tensor  # [B] reset this slot's memory before updating


@dataclasses.dataclass
class StepOutputs:
    memory: MemoryState
    slow_written: Tensor  # [B]
    a_obs: AffordancePred
    a_use: AffordancePred
    goal: GoalPred
    capability: Tensor  # C_t^G [B, E, d]
    mode_logits: Tensor  # [B, num_modes]
    cond_tokens: Tensor  # [B, N_c, d]  typed condition tokens for the action heads
    cond_mask: Tensor  # [B, N_c]
    cond_types: Tensor  # [N_c] source ids (SRC)
    trace: tuple[Tensor, Tensor, Tensor] | None = None  # training-only readouts
    phase_z: Tensor | None = None


class MobileBenchConditioner(nn.Module):
    def __init__(self, cfg: MobileBenchConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        # Input projections / encoders.
        self.p_h = nn.Sequential(nn.LayerNorm(cfg.vlm_width), nn.Linear(cfg.vlm_width, d))  # P_H
        self.state_enc = gelu_mlp_tokens(2 * cfg.state_dim, d, cfg.state_tokens)  # E_S: values + mask
        self.exec_enc = gelu_mlp_tokens(2 * cfg.exec_dim, d, cfg.exec_tokens)  # E_xi: values + mask
        self.gripper_enc = mlp(cfg.gripper_dim, d, d)  # E_g
        self.base_enc = mlp(cfg.base_desc_dim, d, d)  # e_r^B
        self.workspace_enc = WorkspaceEncoder(cfg)  # MLP_W
        # Source tags e_H, e_F, e_S, e_s used inside the decoders' contexts.
        self.ctx_src = nn.Parameter(torch.zeros(1, 4, 1, d))
        # Core modules.
        self.memory = DualMemory(cfg)
        self.aff_obs = CurrentAffordanceDecoder(cfg)
        self.aff_use = JointAffordanceDecoder(cfg)
        self.use_encoder = PointEncoder(d)  # E_N / E_O of the JOINT prediction
        self.goal_dec = EEFGoalDecoder(cfg)
        self.goal_enc = PoseEncoder(d, cfg.num_end_effectors)  # E_G
        self.goal_ws = GoalWorkspaceDecoder(cfg)
        self.mode_dec = ModeDecoder(cfg)
        self.mode_enc = nn.Linear(cfg.num_modes, d)  # E_m(mode probabilities)
        # Optional compression of W_t before it joins the action-head condition stream.
        self.ws_pool = None
        if cfg.action_workspace_tokens is not None:
            self.ws_pool_queries = nn.Parameter(torch.randn(1, cfg.action_workspace_tokens, d) * 0.02)
            self.ws_pool = QueryDecoder(d, cfg.num_heads, 1, cfg.ffn_mult, cfg.dropout)
        self.cond_type_emb = nn.Embedding(len(SOURCES), d)
        # Training-only readouts.
        self.trace_readout = TraceReadout(cfg)
        self.phase_readout = PhaseTextReadout(cfg)

    # -- helpers ------------------------------------------------------------------------------
    def _gate(self, logits: Tensor) -> Tensor:
        """Soft validity gate in training; hard NULL replacement at inference."""
        p = logits.sigmoid()
        return p if self.training else (p > self.cfg.null_threshold).to(p.dtype)

    def _tokens(self, enc: nn.Module, x: Tensor, mask: Tensor, n: int) -> Tensor:
        # Missing slots are zeroed AND flagged, so a zero value never implies an existing joint.
        inp = torch.cat([x * mask.to(x.dtype), mask.to(x.dtype)], dim=-1)
        return enc(inp).view(x.shape[0], n, self.cfg.d_model)

    def init_memory(self, batch: int, device: torch.device | None = None) -> MemoryState:
        return self.memory.init_state(batch, device)

    # -- one policy update ------------------------------------------------------------------
    def forward(self, inp: StepInputs, memory: MemoryState, *, with_readouts: bool = False) -> StepOutputs:
        cfg = self.cfg
        b = inp.h.shape[0]
        dev = inp.h.device

        # Encoders.
        h = self.p_h(inp.h)
        s_tok = self._tokens(self.state_enc, inp.state, inp.state_mask, cfg.state_tokens)
        x_tok = self._tokens(self.exec_enc, inp.exec_increment, inp.exec_mask, cfg.exec_tokens)
        g_tok = self.gripper_enc(inp.gripper_desc)[:, None]
        base_tok = self.base_enc(inp.base_desc)[:, None]
        ws_feats = workspace_features(inp.lib_pos, inp.lib_rot, inp.eef_pos, inp.eef_rot, inp.lib_cost, inp.lib_margin)
        w_tok = self.workspace_enc(ws_feats)

        # Current-observation affordance: H_t and state only.
        e_h, e_f, e_s, e_st = self.ctx_src.unbind(dim=1)
        h_tagged = h + e_h
        a_obs = self.aff_obs(h_tagged, inp.h_mask, s_tok + e_st)

        # Memory: reset finished episodes, then one write for this NEW observation.
        memory = self.memory.reset(memory, inp.new_episode)
        memory, slow_written = self.memory(memory, h_tagged, inp.h_mask, x_tok, s_tok, inp.dt)

        # Shared decoder context [H ; M^F ; M^S ; s_t], each with its source tag.
        ctx = torch.cat([h_tagged, memory.fast + e_f, memory.slow + e_s, s_tok + e_st], dim=1)
        ones = lambda n: torch.ones(b, n, dtype=torch.bool, device=dev)  # noqa: E731
        ctx_mask = torch.cat([inp.h_mask, ones(ctx.shape[1] - inp.h.shape[1])], dim=1)

        # Joint affordance -> what downstream actually uses.
        a_use = self.aff_use(ctx, ctx_mask, a_obs, self._gate(a_obs.valid_logits))
        use_tok = self.use_encoder(a_use.points, self._gate(a_use.valid_logits))  # [B, 2, d]

        # EEF goal: reads H, M^F, M^S, state, gripper and the JOINT object point.
        goal_ctx, goal_mask = ctx[:, : -s_tok.shape[1]], ctx_mask[:, : -s_tok.shape[1]]
        goal = self.goal_dec(goal_ctx, goal_mask, s_tok, g_tok, use_tok[:, OBJ : OBJ + 1], inp.ee_mask)
        goal_gate = self._gate(goal.valid_logits) * inp.ee_mask.to(goal.valid_logits.dtype)
        goal_tok = self.goal_enc(goal.pos, goal.rot6d, goal_gate)

        # Goal-conditioned capability.
        capability = self.goal_ws(
            goal.pos, goal.rot6d, goal.rot, goal_gate, g_tok, ctx, ctx_mask,
            w_tok, inp.ws_mask, inp.lib_pos, inp.lib_rot, inp.lib_cost, inp.lib_margin,
        )  # fmt: skip

        # Online mode: latest features + predictions, never the true phase.
        mode_ctx = torch.cat([ctx, use_tok, goal_tok, capability], dim=1)
        mode_mask = torch.cat([ctx_mask, ones(mode_ctx.shape[1] - ctx.shape[1])], dim=1)
        mode_logits = self.mode_dec(mode_ctx, mode_mask)
        mode_tok = self.mode_enc(mode_logits.softmax(-1))[:, None]

        # Typed condition stream for the action heads.
        if self.ws_pool is not None:
            w_act = self.ws_pool(self.ws_pool_queries.expand(b, -1, -1), w_tok, inp.ws_mask)
            # A robot with no library samples has nothing to pool: mask the pooled tokens too.
            w_act_mask = inp.ws_mask.any(dim=1, keepdim=True).expand(b, w_act.shape[1])
        else:
            w_act, w_act_mask = w_tok, inp.ws_mask
        e = cfg.num_end_effectors
        groups = [
            ("fast", memory.fast, ones(cfg.num_fast_slots)),
            ("slow", memory.slow, ones(cfg.num_slow_slots)),
            ("state", s_tok, ones(cfg.state_tokens)),
            ("workspace", w_act, w_act_mask),
            ("capability", capability, ones(e)),
            ("affordance", use_tok, ones(2)),
            ("goal", goal_tok, ones(e)),
            ("mode", mode_tok, ones(1)),
            ("gripper", g_tok, ones(1)),
            ("base", base_tok, ones(1)),
        ]
        types = torch.cat([torch.full((t.shape[1],), SRC[n], dtype=torch.long, device=dev) for n, t, _ in groups])
        cond = torch.cat([t for _, t, _ in groups], dim=1) + self.cond_type_emb(types)[None]
        cond_mask = torch.cat([m for _, _, m in groups], dim=1)

        out = StepOutputs(
            memory=memory,
            slow_written=slow_written,
            a_obs=a_obs,
            a_use=a_use,
            goal=goal,
            capability=capability,
            mode_logits=mode_logits,
            cond_tokens=cond,
            cond_mask=cond_mask,
            cond_types=types,
        )
        if with_readouts:
            out.trace = self.trace_readout(memory.fast)  # fast group ONLY
            out.phase_z = self.phase_readout(memory.slow)  # slow group ONLY
        return out
