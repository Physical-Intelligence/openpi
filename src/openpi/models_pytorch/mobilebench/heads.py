"""Mode decoder and the training-only memory readouts.

Mode decoder (sec. 8.5): reads H, M^F, M^S, the predicted affordances / goal and the
capability tokens. The true phase only supervises it; it is never a hard routing input.

Training-only readouts are INTENTIONALLY input-restricted (sec. 4.3, table last rows):
  TraceReadout       reads ONLY M^F  -> past EEF pose + gripper at a few lags (sec. 6.1)
  PhaseTextReadout   reads ONLY M^S  -> embedding aligned to frozen phase-text teachers (sec. 7)
Wiring H_t into either would let a correct reconstruction come from the current image
instead of from the memory group being checked. Both can be dropped at deployment.
"""

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

from openpi.models_pytorch.mobilebench.config import MobileBenchConfig
from openpi.models_pytorch.mobilebench.layers import QueryDecoder
from openpi.models_pytorch.mobilebench.layers import mlp


class ModeDecoder(nn.Module):
    def __init__(self, cfg: MobileBenchConfig):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.decoder = QueryDecoder(cfg.d_model, cfg.num_heads, 2, cfg.ffn_mult, cfg.dropout)
        self.head = mlp(cfg.d_model, cfg.d_model, cfg.num_modes)

    def forward(self, context: Tensor, context_mask: Tensor | None) -> Tensor:
        """context: [H ; M^F ; M^S ; s_t ; affordance/goal/capability tokens] -> logits [B, num_modes]."""
        z = self.decoder(self.query.expand(context.shape[0], -1, -1), context, context_mask)
        return self.head(z[:, 0])


class TraceReadout(nn.Module):
    """D_T(M^F, Q_lag): recall the actual EEF pose and gripper at past lags from the fast group only."""

    def __init__(self, cfg: MobileBenchConfig):
        super().__init__()
        self.lag_queries = nn.Parameter(torch.randn(1, cfg.trace_lags, cfg.d_model) * 0.02)
        self.decoder = QueryDecoder(cfg.d_model, cfg.num_heads, 2, cfg.ffn_mult, cfg.dropout)
        self.f_p = mlp(cfg.d_model, cfg.d_model, 3)
        self.f_r = mlp(cfg.d_model, cfg.d_model, 6)
        self.f_g = mlp(cfg.d_model, cfg.d_model, 1)

    def forward(self, fast: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """fast [B, N_F, d] -> (pos [B,L,3], rot6d [B,L,6], gripper [B,L]) in the CURRENT body frame."""
        z = self.decoder(self.lag_queries.expand(fast.shape[0], -1, -1), fast)
        return self.f_p(z), self.f_r(z), self.f_g(z).squeeze(-1)


class PhaseTextReadout(nn.Module):
    """z^S = norm(P_S Read(M^S)), compared against frozen text-teacher embeddings in the loss."""

    def __init__(self, cfg: MobileBenchConfig):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.read = QueryDecoder(cfg.d_model, cfg.num_heads, 1, cfg.ffn_mult, cfg.dropout)
        self.p_s = nn.Linear(cfg.d_model, cfg.phase_text_dim)

    def forward(self, slow: Tensor) -> Tensor:
        z = self.read(self.query.expand(slow.shape[0], -1, -1), slow)[:, 0]
        return F.normalize(self.p_s(z), dim=-1)
