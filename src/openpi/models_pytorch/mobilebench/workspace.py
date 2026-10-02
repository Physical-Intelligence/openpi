"""Embodiment capability: workspace tokens W_t and the Goal-Workspace Decoder (sec. 8.2, 8.4).

The reachable-pose library P_r = {(q_i, p_i, R_i)} is FIXED per robot (offline FK/limits/
self-collision sampling, base held fixed). What changes online is the relation of each
sample to the current EEF pose and joint configuration, encoded per sample as

    x_i^W = [p_i ; rho(R_i) ; p_i - p_E ; rho(R_E^T R_i) ; c_i ; mu_i]          (20 dims)
    c_i   = max_j |dq_ij| / qdot_max_j      (coarse cost to that sampled solution, not a time)
    mu_i  = normalised joint-limit margin of the sample's configuration

and mapped by a SHARED learnable encoder MLP_W (20 -> 256 -> 256, GELU, LayerNorm).
Sampling holes are not unreachability; the nearest library sample never becomes a goal.
"""

import math

import torch
from torch import Tensor
from torch import nn

from openpi.models_pytorch.mobilebench.config import MobileBenchConfig
from openpi.models_pytorch.mobilebench.layers import RelationalCrossAttention
from openpi.models_pytorch.mobilebench.layers import mlp
from openpi.models_pytorch.mobilebench.rotation import matrix_to_rot6d

REL_DIM = 3 + 6 + 1 + 1  # delta_i^G = [p_i - p_G ; rho(R_G^T R_i) ; c_i ; mu_i]


def joint_space_cost(
    lib_q: Tensor, cur_q: Tensor, qdot_max: Tensor, wraps: Tensor, joint_mask: Tensor | None = None
) -> Tensor:
    """c_i = max_j |dq_ij| / qdot_max_j.

    lib_q [B, N, J], cur_q [B, J], qdot_max [B, J] (> 0), wraps [J] bool: continuous
    revolute joints use the wrapped angle difference, prismatic / limited joints the plain
    difference. joint_mask [B, J] excludes joints this robot does not have.
    """
    dq = lib_q - cur_q[:, None]
    wrapped = (dq + math.pi) % (2 * math.pi) - math.pi
    dq = torch.where(wraps[None, None], wrapped, dq)
    ratio = dq.abs() / qdot_max[:, None].clamp_min(1e-6)
    if joint_mask is not None:
        ratio = ratio.masked_fill(~joint_mask[:, None], 0.0)
    return ratio.amax(dim=-1)


def workspace_features(
    lib_pos: Tensor, lib_rot: Tensor, eef_pos: Tensor, eef_rot: Tensor, cost: Tensor, margin: Tensor
) -> Tensor:
    """lib_pos [B,N,3], lib_rot [B,N,3,3], eef_pos [B,3], eef_rot [B,3,3], cost/margin [B,N] -> [B,N,20]."""
    rel_rot = eef_rot.transpose(-1, -2)[:, None] @ lib_rot
    return torch.cat(
        [
            lib_pos,
            matrix_to_rot6d(lib_rot),
            lib_pos - eef_pos[:, None],
            matrix_to_rot6d(rel_rot),
            cost[..., None],
            margin[..., None],
        ],
        dim=-1,
    )


class WorkspaceEncoder(nn.Module):
    """Shared across robots: MLP_W 20 -> d -> d with GELU and LayerNorm."""

    def __init__(self, cfg: MobileBenchConfig):
        super().__init__()
        self.net = mlp(cfg.workspace_feat_dim, cfg.d_model, cfg.d_model, norm=True)

    def forward(self, feats: Tensor) -> Tensor:
        return self.net(feats)


class _GoalWorkspaceLayer(nn.Module):
    """Two-step block: read task context, then query the capability tokens with relations."""

    def __init__(self, d: int, heads: int, ffn_mult: int):
        super().__init__()
        self.ln_ctx_q, self.ln_ctx_kv = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ctx_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln_ws_q, self.ln_ws_kv = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ws_attn = RelationalCrossAttention(d, heads, REL_DIM)
        self.ffn = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, ffn_mult * d), nn.GELU(), nn.Linear(ffn_mult * d, d))

    def forward(self, q, ctx, ctx_pad, ws, ws_mask, rel):
        kv = self.ln_ctx_kv(ctx)
        q = q + self.ctx_attn(self.ln_ctx_q(q), kv, kv, key_padding_mask=ctx_pad, need_weights=False)[0]
        q = q + self.ws_attn(self.ln_ws_q(q), self.ln_ws_kv(ws), rel, ws_mask)
        return q + self.ffn(q)


class GoalWorkspaceDecoder(nn.Module):
    """Goal-conditioned capability tokens C_t^G -- NOT another goal and NOT a yes/no reachability flag."""

    def __init__(self, cfg: MobileBenchConfig):
        super().__init__()
        d = cfg.d_model
        self.e_c = mlp(3 + 6 + 1 + d, d, d)  # E_C(p_G, rho(R_G), v_G, e_g)
        self.layers = nn.ModuleList(
            _GoalWorkspaceLayer(d, cfg.num_heads, cfg.ffn_mult) for _ in range(cfg.goal_workspace_layers)
        )
        self.out_norm = nn.LayerNorm(d)
        self.c_null = nn.Parameter(torch.randn(1, cfg.num_end_effectors, d) * 0.02)  # C^G_NULL

    def forward(
        self,
        goal_pos: Tensor,
        goal_rot6d: Tensor,
        goal_rot: Tensor,
        goal_gate: Tensor,
        gripper_token: Tensor,
        context: Tensor,
        context_mask: Tensor | None,
        ws_tokens: Tensor,
        ws_mask: Tensor | None,
        lib_pos: Tensor,
        lib_rot: Tensor,
        cost: Tensor,
        margin: Tensor,
    ) -> Tensor:
        """Returns C_t^G [B, E, d].

        goal_*:   [B, E, ...] predicted goal per end effector; goal_gate [B, E] validity in [0, 1]
        context:  source-tagged [H ; M^F ; M^S ; s_t] (e_g is appended here)
        ws_*:     W_t [B, N, d], its padding mask, and the raw library poses/cost/margin
        Where the goal is invalid the dedicated C_NULL is used instead of a geometric
        query against a placeholder pose (sec. 8.4).
        """
        b, e, _ = goal_pos.shape
        q = self.e_c(torch.cat([goal_pos, goal_rot6d, goal_gate[..., None], gripper_token.expand(b, e, -1)], dim=-1))
        ctx = torch.cat([context, gripper_token], dim=1)
        ctx_pad = None
        if context_mask is not None:
            ctx_pad = ~torch.cat([context_mask, torch.ones(b, 1, dtype=torch.bool, device=ctx.device)], dim=1)
        # delta_{i}^G for every (end effector, library sample) pair: [B, E, N, 11]
        rel = torch.cat(
            [
                lib_pos[:, None] - goal_pos[:, :, None],
                matrix_to_rot6d(goal_rot.transpose(-1, -2)[:, :, None] @ lib_rot[:, None]),
                cost[:, None, :, None].expand(b, e, -1, 1),
                margin[:, None, :, None].expand(b, e, -1, 1),
            ],
            dim=-1,
        )
        for layer in self.layers:
            q = layer(q, ctx, ctx_pad, ws_tokens, ws_mask, rel)
        c = self.out_norm(q)
        g = goal_gate[..., None]
        return g * c + (1.0 - g) * self.c_null
