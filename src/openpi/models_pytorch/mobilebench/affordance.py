"""NAV / object affordance decoders (design doc sec. 6.3-6.5).

Two branches, deliberately different:

  A_obs = D_A^obs([Q_N; Q_O], [P_H H_t ; s_t])
      "what spatial evidence does the CURRENT input give?" Reads no memory, so it may
      legitimately be invalid when only history could locate the target.

  A_use = D_A^use([Q_N; Q_O], [H ; M^F ; M^S ; s_t ; E_obs(A_obs)])
      "given current input, history and progress, which target should be used now?"
      The prediction downstream goals and actions consume.

A_obs enters the joint decoder as typed, source-tagged evidence. The two point sets are
never coordinate-averaged and no consistency loss forces them equal (sec. 6.4, 9.3).
Points are 3D in the current body frame B_t.
"""

import dataclasses

import torch
from torch import Tensor
from torch import nn

from openpi.models_pytorch.mobilebench.config import AFFORDANCE_TYPES
from openpi.models_pytorch.mobilebench.config import MobileBenchConfig
from openpi.models_pytorch.mobilebench.layers import QueryDecoder
from openpi.models_pytorch.mobilebench.layers import mlp

NUM_TYPES = len(AFFORDANCE_TYPES)


@dataclasses.dataclass
class AffordancePred:
    points: Tensor  # [B, 2, 3]   index 0 = NAV, 1 = object (config.NAV / config.OBJ)
    valid_logits: Tensor  # [B, 2]
    latents: Tensor  # [B, 2, d]   decoder outputs z^j

    @property
    def valid_prob(self) -> Tensor:
        return self.valid_logits.sigmoid()


class _TypedPointHeads(nn.Module):
    """One position MLP and one validity MLP per affordance type (MLP_{p,j}, MLP_{v,j})."""

    def __init__(self, d: int):
        super().__init__()
        self.pos = nn.ModuleList(mlp(d, d, 3) for _ in range(NUM_TYPES))
        self.valid = nn.ModuleList(mlp(d, d, 1) for _ in range(NUM_TYPES))

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        pts = torch.stack([self.pos[j](z[:, j]) for j in range(NUM_TYPES)], dim=1)
        logits = torch.cat([self.valid[j](z[:, j]) for j in range(NUM_TYPES)], dim=-1)
        return pts, logits


class CurrentAffordanceDecoder(nn.Module):
    """H_t -> current-observation affordance. Queries carry no true coordinates, IDs or phase."""

    def __init__(self, cfg: MobileBenchConfig):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, NUM_TYPES, cfg.d_model) * 0.02)  # [Q_N^obs; Q_O^obs]
        self.decoder = QueryDecoder(cfg.d_model, cfg.num_heads, cfg.affordance_layers, cfg.ffn_mult, cfg.dropout)
        self.heads = _TypedPointHeads(cfg.d_model)

    def forward(self, h: Tensor, h_mask: Tensor | None, state_tokens: Tensor) -> AffordancePred:
        b = h.shape[0]
        ctx = torch.cat([h, state_tokens], dim=1)
        mask = None
        if h_mask is not None:
            mask = torch.cat([h_mask, torch.ones(b, state_tokens.shape[1], dtype=torch.bool, device=h.device)], 1)
        z = self.decoder(self.queries.expand(b, -1, -1), ctx, mask)
        pts, logits = self.heads(z)
        return AffordancePred(pts, logits, z)


class PointEncoder(nn.Module):
    """E(.) for predicted points: type embedding + validity-gated point code, else a typed NULL.

    An invalid prediction is replaced by a learned per-type NULL token -- never by a
    placeholder origin that downstream modules could mistake for a real point (sec. 6.4).
    `gate` is the validity probability: soft during training (differentiable), and the
    caller may binarise it at inference with config.null_threshold.
    """

    def __init__(self, d: int, num_types: int = NUM_TYPES):
        super().__init__()
        self.type_emb = nn.Parameter(torch.randn(1, num_types, d) * 0.02)
        self.null_emb = nn.Parameter(torch.randn(1, num_types, d) * 0.02)
        self.point = mlp(3 + 1, d, d)

    def forward(self, points: Tensor, gate: Tensor) -> Tensor:
        """points [B, T, 3], gate [B, T] in [0, 1] -> tokens [B, T, d]."""
        g = gate.unsqueeze(-1)
        code = self.point(torch.cat([points, g], dim=-1))
        return self.type_emb + g * code + (1.0 - g) * self.null_emb


class JointAffordanceDecoder(nn.Module):
    """[H ; M^F ; M^S ; s_t ; E_obs(A_obs)] -> A_use, the task affordance actually used downstream."""

    def __init__(self, cfg: MobileBenchConfig):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, NUM_TYPES, cfg.d_model) * 0.02)  # [Q_N^use; Q_O^use]
        self.obs_encoder = PointEncoder(cfg.d_model)  # E_obs
        self.decoder = QueryDecoder(cfg.d_model, cfg.num_heads, cfg.affordance_layers, cfg.ffn_mult, cfg.dropout)
        self.heads = _TypedPointHeads(cfg.d_model)

    def forward(
        self, context: Tensor, context_mask: Tensor | None, a_obs: AffordancePred, obs_gate: Tensor
    ) -> AffordancePred:
        """context: already source-tagged [H+e_H ; M^F+e_F ; M^S+e_S ; s_t]; obs_gate [B, 2]."""
        b = context.shape[0]
        obs_tok = self.obs_encoder(a_obs.points, obs_gate)
        ctx = torch.cat([context, obs_tok], dim=1)
        mask = None
        if context_mask is not None:
            mask = torch.cat([context_mask, torch.ones(b, NUM_TYPES, dtype=torch.bool, device=ctx.device)], 1)
        z = self.decoder(self.queries.expand(b, -1, -1), ctx, mask)
        pts, logits = self.heads(z)
        return AffordancePred(pts, logits, z)
