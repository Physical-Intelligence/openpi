"""Dual-timescale latent memory (design doc sec. 5).

Fast group  M^F: visual-motor interaction state, written on every policy update.
Slow group  M^S: phase / object / destination context, written every K-th update.

    ~M^F_t = U_F(M^F_{t-1}, H_t, E_xi(xi_t), M^S_{t-1});  M^F_t = (1-a_F) M^F_{t-1} + a_F ~M^F_t
    ~M^S_t = U_S(M^S_{t-1}, H_t, M^F_t, s_t)            ;  M^S_t = (1-a_S) M^S_{t-1} + a_S ~M^S_t
    (M^S_t = M^S_{t-1} on non-write steps)

Information flow has no cycle: the OLD slow group guides this update's fast write, the
NEW fast group feeds the slow write, and goals/actions are decoded afterwards. Both
groups persist across NAV/MANIP switches and are reset only at a new episode (sec. 5.3).

Memories are latent tokens; no image cache, trajectory store or text log is kept.
"""

import dataclasses

import torch
from torch import Tensor
from torch import nn

from openpi.models_pytorch.mobilebench.config import MobileBenchConfig
from openpi.models_pytorch.mobilebench.layers import QueryDecoder
from openpi.models_pytorch.mobilebench.layers import sinusoidal


@dataclasses.dataclass
class MemoryState:
    """Per-batch-slot recurrent state. Each slot is an independent episode stream."""

    fast: Tensor  # [B, N_F, d]
    slow: Tensor  # [B, N_S, d]
    step: Tensor  # [B] long: policy updates since this slot's episode started
    since_slow: Tensor  # [B] float: real seconds since the last slow write

    def detach(self) -> "MemoryState":
        """TBPTT boundary: keep the values, cut the graph (sec. 9.6). Memory is NOT cleared."""
        return MemoryState(self.fast.detach(), self.slow.detach(), self.step.clone(), self.since_slow.clone())


class _Writer(nn.Module):
    """Old memory (+ slot embedding) as queries; residual attention/FFN over the inputs."""

    def __init__(self, cfg: MobileBenchConfig, num_slots: int):
        super().__init__()
        self.slot_emb = nn.Parameter(torch.randn(1, num_slots, cfg.d_model) * 0.02)
        self.decoder = QueryDecoder(cfg.d_model, cfg.num_heads, cfg.writer_layers, cfg.ffn_mult, cfg.dropout)

    def forward(self, old: Tensor, ctx: Tensor, ctx_mask: Tensor | None) -> Tensor:
        return self.decoder(old + self.slot_emb, ctx, ctx_mask)


class DualMemory(nn.Module):
    def __init__(self, cfg: MobileBenchConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.fast_init = nn.Parameter(torch.randn(1, cfg.num_fast_slots, d) * 0.02)
        self.slow_init = nn.Parameter(torch.randn(1, cfg.num_slow_slots, d) * 0.02)
        self.fast_writer = _Writer(cfg, cfg.num_fast_slots)
        self.slow_writer = _Writer(cfg, cfg.num_slow_slots)
        # Source tags so a writer can tell which group / input a context token came from.
        self.src_old_slow = nn.Parameter(torch.zeros(1, 1, d))
        self.src_new_fast = nn.Parameter(torch.zeros(1, 1, d))
        # The slow writer is told the real time elapsed since its previous write (sec. 4.4).
        self.slow_dt = nn.Linear(d, d)

    # -- state management ------------------------------------------------------------------
    def init_state(self, batch: int, device: torch.device | None = None) -> MemoryState:
        device = device or self.fast_init.device
        return MemoryState(
            fast=self.fast_init.expand(batch, -1, -1).to(device),
            slow=self.slow_init.expand(batch, -1, -1).to(device),
            step=torch.zeros(batch, dtype=torch.long, device=device),
            since_slow=torch.zeros(batch, device=device),
        )

    def reset(self, state: MemoryState, new_episode: Tensor) -> MemoryState:
        """Reset only the slots that start a new episode; the others keep their memory."""
        m = new_episode.to(torch.bool)
        b = m.shape[0]
        mf, ms = m[:, None, None], m[:, None, None]
        return MemoryState(
            fast=torch.where(mf, self.fast_init.expand(b, -1, -1), state.fast),
            slow=torch.where(ms, self.slow_init.expand(b, -1, -1), state.slow),
            step=torch.where(m, torch.zeros_like(state.step), state.step),
            since_slow=torch.where(m, torch.zeros_like(state.since_slow), state.since_slow),
        )

    # -- one policy update -------------------------------------------------------------------
    def forward(
        self,
        state: MemoryState,
        h: Tensor,
        h_mask: Tensor | None,
        exec_tokens: Tensor,
        state_tokens: Tensor,
        dt: Tensor,
    ) -> tuple[MemoryState, Tensor]:
        """Advance the memory by one NEW observation.

        h            [B, N_H, d]  projected (and source-tagged) current VLM tokens P_H H_t
        h_mask       [B, N_H]     True = valid token
        exec_tokens  [B, N_x, d]  E_xi(xi_t): executed EEF / gripper / body increments since the
                                  last update. Only what was actually executed, never planned actions.
        state_tokens [B, N_s, d]  s_t
        dt           [B]          seconds since the previous policy update
        Returns the new state and the per-slot boolean "slow group written this update".

        Call once per new environment observation -- NOT once per flow-matching denoising
        step (sec. 10.2: ten denoising steps are not ten experiences).
        """
        cfg = self.cfg
        b = h.shape[0]
        ones = lambda t: torch.ones(b, t.shape[1], dtype=torch.bool, device=h.device)  # noqa: E731
        hm = h_mask if h_mask is not None else ones(h)

        # Fast write: reads current tokens, executed increment and the OLD slow group.
        fast_ctx = torch.cat([h, exec_tokens, state.slow + self.src_old_slow], dim=1)
        fast_mask = torch.cat([hm, ones(exec_tokens), ones(state.slow)], dim=1)
        fast_cand = self.fast_writer(state.fast, fast_ctx, fast_mask)
        fast = (1.0 - cfg.fast_alpha) * state.fast + cfg.fast_alpha * fast_cand

        # Slow write on the K-th updates only; reads the NEW fast group and current state.
        since = state.since_slow + dt
        write_slow = (state.step % cfg.slow_write_every) == 0
        if write_slow.any():
            dt_tok = self.slow_dt(sinusoidal(since, cfg.d_model))[:, None]
            slow_ctx = torch.cat([h, fast + self.src_new_fast, state_tokens, dt_tok], dim=1)
            slow_mask = torch.cat([hm, ones(fast), ones(state_tokens), ones(dt_tok)], dim=1)
            slow_cand = self.slow_writer(state.slow, slow_ctx, slow_mask)
            slow_new = (1.0 - cfg.slow_alpha) * state.slow + cfg.slow_alpha * slow_cand
            slow = torch.where(write_slow[:, None, None], slow_new, state.slow)
        else:  # no slot writes this update: skip the slow writer entirely
            slow = state.slow

        new_state = MemoryState(
            fast=fast,
            slow=slow,
            step=state.step + 1,
            since_slow=torch.where(write_slow, torch.zeros_like(since), since),
        )
        return new_state, write_slow
