"""EEF Goal Decoder (design doc sec. 8.3).

    K^G   = [P_H H + e_H ; P_F M^F + e_F ; P_S M^S + e_S ; s_t ; e_g ; E_O(p^O, v^O)]
    q_0^G = Q_EEF + P_g e_g + P_s Pool(s_t)
    per layer:  q <- q + MHA(LN q, LN K^G, LN K^G);  q <- q + FFN(LN q)      (x2)
    heads:      p^G in R^3,  r^G in R^6 (-> SO(3)),  v^G = sigmoid(logit)

The goal is the NEXT KEY TCP interaction pose for the given gripper, not an object
centre. It is never projected or clipped into the current workspace, and NAV frames do
not mask it: "known but out of reach" is a normal state that the Goal-Workspace decoder
explains (sec. 6.6, 8.3). E_O consumes only the JOINT branch's predicted object point.
"""

import dataclasses

import torch
from torch import Tensor
from torch import nn

from openpi.models_pytorch.mobilebench.config import MobileBenchConfig
from openpi.models_pytorch.mobilebench.layers import QueryDecoder
from openpi.models_pytorch.mobilebench.layers import masked_token_mean
from openpi.models_pytorch.mobilebench.layers import mlp
from openpi.models_pytorch.mobilebench.rotation import rot6d_to_matrix


@dataclasses.dataclass
class GoalPred:
    pos: Tensor  # [B, E, 3]   in the current body frame B_t
    rot6d: Tensor  # [B, E, 6]
    valid_logits: Tensor  # [B, E]
    latents: Tensor  # [B, E, d]
    ee_mask: Tensor  # [B, E] True = end effector exists on this robot

    @property
    def rot(self) -> Tensor:
        return rot6d_to_matrix(self.rot6d)

    @property
    def valid_prob(self) -> Tensor:
        return self.valid_logits.sigmoid() * self.ee_mask.to(self.valid_logits.dtype)


class PoseEncoder(nn.Module):
    """E_G: validity-gated pose token with a typed NULL (never an origin pose)."""

    def __init__(self, d: int, num_slots: int):
        super().__init__()
        self.slot_emb = nn.Parameter(torch.randn(1, num_slots, d) * 0.02)
        self.null_emb = nn.Parameter(torch.randn(1, num_slots, d) * 0.02)
        self.pose = mlp(3 + 6 + 1, d, d)

    def forward(self, pos: Tensor, rot6d: Tensor, gate: Tensor) -> Tensor:
        g = gate.unsqueeze(-1)
        code = self.pose(torch.cat([pos, rot6d, g], dim=-1))
        return self.slot_emb + g * code + (1.0 - g) * self.null_emb


class EEFGoalDecoder(nn.Module):
    def __init__(self, cfg: MobileBenchConfig):
        super().__init__()
        d, e = cfg.d_model, cfg.num_end_effectors
        self.q_eef = nn.Parameter(torch.randn(1, e, d) * 0.02)  # one learned query per end effector
        self.p_g = nn.Linear(d, d)  # P_g e_g
        self.p_s = nn.Linear(d, d)  # P_s Pool(s_t)
        # The doc's per-layer update is cross-attention + FFN only (no query self-attention).
        self.decoder = QueryDecoder(d, cfg.num_heads, cfg.goal_layers, cfg.ffn_mult, cfg.dropout, self_attn=False)
        self.f_p = mlp(d, d, 3)
        self.f_r = mlp(d, d, 6)
        self.f_v = mlp(d, d, 1)
        # Bias the 6D head toward identity so early predictions are well conditioned.
        with torch.no_grad():
            self.f_r[-1].bias.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def forward(
        self,
        context: Tensor,
        context_mask: Tensor | None,
        state_tokens: Tensor,
        gripper_token: Tensor,
        obj_token: Tensor,
        ee_mask: Tensor | None = None,
    ) -> GoalPred:
        """context: source-tagged [H ; M^F ; M^S]; gripper_token [B,1,d]; obj_token = E_O(A_use object) [B,1,d]."""
        b = context.shape[0]
        e = self.q_eef.shape[1]
        if ee_mask is None:
            ee_mask = torch.ones(b, e, dtype=torch.bool, device=context.device)
        k = torch.cat([context, state_tokens, gripper_token, obj_token], dim=1)
        k_mask = None
        if context_mask is not None:
            extra = torch.ones(b, k.shape[1] - context.shape[1], dtype=torch.bool, device=k.device)
            k_mask = torch.cat([context_mask, extra], dim=1)
        s_pool = masked_token_mean(state_tokens, torch.ones(state_tokens.shape[:2], device=k.device), dim=1)
        q0 = self.q_eef + self.p_g(gripper_token) + self.p_s(s_pool)[:, None]
        z = self.decoder(q0.expand(b, -1, -1), k, k_mask)
        return GoalPred(self.f_p(z), self.f_r(z), self.f_v(z).squeeze(-1), z, ee_mask)
