"""Shared building blocks for the MobileBench decoders and memory writers."""

import math

import torch
from torch import Tensor
from torch import nn


def mlp(in_dim: int, hidden: int, out_dim: int, *, norm: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim)]
    if norm:
        layers.append(nn.LayerNorm(out_dim))
    return nn.Sequential(*layers)


def key_padding(mask: Tensor | None) -> Tensor | None:
    """nn.MultiheadAttention wants True = ignore; our masks use True = valid.

    A row with NO valid key (e.g. the workspace of an arm-less robot) would make
    MultiheadAttention return NaN, which then poisons every token it touches. Such rows
    attend to everything instead; their outputs are masked or replaced by a NULL token
    downstream, so the values they read do not matter.
    """
    if mask is None:
        return None
    empty = ~mask.any(dim=-1, keepdim=True)
    return ~(mask | empty)


class QueryDecoderLayer(nn.Module):
    """Pre-LN block: query self-attention -> cross-attention to context -> FFN.

    This is the "small attention decoder" used throughout the design doc (sec. 6.3,
    8.3) and also the residual attention/FFN update of the memory writers (sec. 5.1).
    """

    def __init__(self, d: int, heads: int, ffn_mult: int = 4, dropout: float = 0.0, self_attn: bool = True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True) if self_attn else None
        self.ln_sa = nn.LayerNorm(d) if self_attn else None
        self.cross_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ln_q = nn.LayerNorm(d)
        self.ln_kv = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, ffn_mult * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_mult * d, d)
        )

    def forward(self, q: Tensor, ctx: Tensor, ctx_mask: Tensor | None = None) -> Tensor:
        if self.self_attn is not None:
            h = self.ln_sa(q)
            q = q + self.self_attn(h, h, h, need_weights=False)[0]
        kv = self.ln_kv(ctx)
        q = q + self.cross_attn(self.ln_q(q), kv, kv, key_padding_mask=key_padding(ctx_mask), need_weights=False)[0]
        return q + self.ffn(q)


class QueryDecoder(nn.Module):
    def __init__(
        self, d: int, heads: int, layers: int, ffn_mult: int = 4, dropout: float = 0.0, self_attn: bool = True
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            QueryDecoderLayer(d, heads, ffn_mult, dropout, self_attn=self_attn) for _ in range(layers)
        )
        self.out_norm = nn.LayerNorm(d)

    def forward(self, q: Tensor, ctx: Tensor, ctx_mask: Tensor | None = None) -> Tensor:
        for layer in self.layers:
            q = layer(q, ctx, ctx_mask)
        return self.out_norm(q)


class RelationalCrossAttention(nn.Module):
    """Multi-head cross-attention whose logits AND values carry pairwise relation features.

    Implements sec. 8.4 step 2:
        a_i^(h) = softmax_i( (W_Q q)^T (W_K w_i) / sqrt(d_h) + b_h(delta_i) )
        o^(h)   = sum_i a_i^(h) W_V [w_i ; E_delta(delta_i)]
    The relation enters the values too, so after softmax normalisation the output still
    knows how far the selected samples are from the goal, not only which is nearest.
    """

    def __init__(self, d: int, heads: int, rel_dim: int):
        super().__init__()
        if d % heads != 0:
            raise ValueError(f"d={d} not divisible by heads={heads}")
        self.heads, self.dh = heads, d // heads
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(2 * d, d)  # values read [w_i ; E_delta(delta_i)]
        self.rel_bias = mlp(rel_dim, d, heads)  # b_h(delta)
        self.rel_value = mlp(rel_dim, d, d)  # E_delta(delta)
        self.out = nn.Linear(d, d)

    def forward(self, q: Tensor, kv: Tensor, rel: Tensor, kv_mask: Tensor | None = None) -> Tensor:
        """q [B, Nq, d], kv [B, N, d], rel [B, Nq, N, rel_dim], kv_mask [B, N] (True=valid)."""
        b, nq, d = q.shape
        n = kv.shape[1]
        qh = self.q(q).view(b, nq, self.heads, self.dh).transpose(1, 2)  # [B, H, Nq, dh]
        kh = self.k(kv).view(b, n, self.heads, self.dh).transpose(1, 2)  # [B, H, N, dh]
        logits = qh @ kh.transpose(-1, -2) / math.sqrt(self.dh)  # [B, H, Nq, N]
        logits = logits + self.rel_bias(rel).permute(0, 3, 1, 2)
        if kv_mask is not None:
            logits = logits.masked_fill(~kv_mask[:, None, None, :], torch.finfo(logits.dtype).min)
        attn = logits.softmax(dim=-1)
        # Values depend on the (query, key) pair through delta, so they are per-query.
        vals = self.v(torch.cat([kv[:, None].expand(b, nq, n, d), self.rel_value(rel)], dim=-1))
        vals = vals.view(b, nq, n, self.heads, self.dh).permute(0, 3, 1, 2, 4)  # [B, H, Nq, N, dh]
        o = (attn.unsqueeze(-1) * vals).sum(dim=-2)  # [B, H, Nq, dh]
        return self.out(o.transpose(1, 2).reshape(b, nq, d))


def sinusoidal(x: Tensor, dim: int, max_period: float = 1e3) -> Tensor:
    """Sinusoidal features of a scalar per batch element, x [B] -> [B, dim]."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=x.device, dtype=torch.float32) / half)
    ang = x.float()[:, None] * freqs[None]
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


def masked_token_mean(x: Tensor, mask: Tensor, dim: int) -> Tensor:
    m = mask.to(x.dtype)
    return (x * m.unsqueeze(-1)).sum(dim) / m.sum(dim).clamp_min(1.0).unsqueeze(-1)


def gelu_mlp_tokens(in_dim: int, d: int, num_tokens: int) -> nn.Sequential:
    """`Linear -> GELU -> Linear` encoder emitting `num_tokens` tokens (state / exec encoders)."""
    return nn.Sequential(nn.Linear(in_dim, d), nn.GELU(), nn.Linear(d, num_tokens * d))


__all__ = [
    "QueryDecoder",
    "QueryDecoderLayer",
    "RelationalCrossAttention",
    "gelu_mlp_tokens",
    "key_padding",
    "masked_token_mean",
    "mlp",
    "sinusoidal",
]
