# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gemma adaptation for Pi, taken from big_vision.

We follow this einsum axis naming convention:
  B: batch
  T: query length
  S: k/v length
  N: num query heads
  K: num k/v heads
  G: num query heads per k/v head
  H: head dim
  D: d_model ("features")

This version integrates a GemMoE feed-forward option that can be enabled
per-config. If a config sets `num_local_experts > 1`, its FFN becomes a
Top-2 MoE block. Otherwise it uses the standard dense FFN.
"""

from collections.abc import Sequence
import dataclasses
from typing import Literal, TypeAlias

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp

import openpi.models.lora as lora
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding

# Shared transformer primitives live in gemma.py; re-export them here so this MoE module and its
# importers (e.g. gemmoe_trace) keep a single source of truth.
from openpi.models.gemma import (
    PALIGEMMA_VOCAB_SIZE,
    Embedder,
    FeedForward,
    RMSNorm,
    _apply_rope,
    _gated_residual,
    _name,
)


# =========================
# Config
# =========================
@dataclasses.dataclass
class Config:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    lora_configs: dict[str, lora.LoRAConfig] = dataclasses.field(default_factory=dict)

    # --- MoE (optional) ---
    num_local_experts: int = 1  # >1 to enable MoE FFN
    num_experts_per_tok: int = 2  # top-k (we implement top-2)
    router_aux_loss_coef: float = 0.0  # kept for future use (loss added in trainer)


Variant = Literal[
    "dummy",
    "gemma_300m",
    "gemma_2b",
    "gemma_2b_lora",
    "gemma_300m_lora",
    "moe_gemma",
    "moe_gemma_2",
    "moe_gemma_8",
]


def get_config(variant: Variant) -> Config:
    """Returns config for specified gemma variant."""
    if variant == "dummy":
        return Config(
            width=64,
            depth=4,
            mlp_dim=128,
            num_heads=8,
            num_kv_heads=1,
            head_dim=16,
        )
    if variant == "gemma_300m":
        # 311M params
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b_lora":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={
                "attn": lora.LoRAConfig(rank=16, alpha=16.0),
                "ffn": lora.LoRAConfig(rank=16, alpha=16.0),
            },
        )
    if variant == "gemma_300m_lora":
        # 311M params
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={
                "attn": lora.LoRAConfig(rank=32, alpha=32.0),
                "ffn": lora.LoRAConfig(rank=32, alpha=32.0),
            },
        )
    if variant == "moe_gemma":
        # A small GemMoE config intended to run alongside gemma_2b
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            num_local_experts=5,         # enable MoE
            num_experts_per_tok=1,       # top-2
            router_aux_loss_coef=1e-3,   # used in trainer when adding aux loss
        )
    if variant == "moe_gemma_2":
        # Table-tasks annotations use two atomic skill groups (PICKUP_FROM, PLACE_*).
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            num_local_experts=2,
            num_experts_per_tok=1,
            router_aux_loss_coef=1e-3,
        )
    if variant == "moe_gemma_8":
        # Calvin annotations use eight atomic skill groups.
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            num_local_experts=8,
            num_experts_per_tok=1,
            router_aux_loss_coef=1e-3,
        )
    raise ValueError(f"Unknown variant: {variant}")


# =========================
# Attention
# =========================
@at.typecheck
class Attention(nn.Module):
    """Attention module."""

    configs: Sequence[Config]

    @nn.compact
    def __call__(self, xs, positions, attn_mask, kv_cache):
        # all experts must share the same head dim, num heads, and num kv heads for self-attention to work
        assert all(config.head_dim == self.configs[0].head_dim for config in self.configs)
        assert all(config.num_heads == self.configs[0].num_heads for config in self.configs)
        assert all(config.num_kv_heads == self.configs[0].num_kv_heads for config in self.configs)

        dtype = next(x.dtype for x in xs if x is not None)  # original dtype, could be half-precision

        qkvs = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                continue
            if config.num_kv_heads == config.num_heads:
                qkv_einsum = lora.Einsum(
                    shape=(3, config.num_heads, config.width, config.head_dim),
                    name=_name("qkv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )
                qkvs.append(qkv_einsum("BSD,3KDH->3BSKH", x))
            else:
                q_einsum = lora.Einsum(
                    shape=(config.num_heads, config.width, config.head_dim),
                    name=_name("q_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                    lora_config=config.lora_configs.get("attn"),
                )
                q = q_einsum("BTD,NDH->BTNH", x)
                kv_einsum = lora.Einsum(
                    shape=(2, config.num_kv_heads, config.width, config.head_dim),
                    name=_name("kv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )
                k, v = kv_einsum("BSD,2KDH->2BSKH", x)
                qkvs.append((q, k, v))

        q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))

        q = _apply_rope(q, positions=positions)
        q *= self.configs[0].head_dim ** -0.5

        k = _apply_rope(k, positions=positions)

        # should still be half-precision here (if input was half-precision)
        assert q.dtype == k.dtype == v.dtype == dtype

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            k = jnp.concatenate([cache_k, k], axis=1)
            v = jnp.concatenate([cache_v, v], axis=1)

        q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=self.configs[0].num_kv_heads)
        logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)

        if attn_mask.shape != (q.shape[0], 1, q.shape[1], k.shape[1]):
            raise ValueError(
                f"Attention mask with shape {attn_mask.shape} but shapes for q and k are: {q.shape} and {k.shape}"
            )

        # big_neg = jnp.finfo(logits.dtype).min
        big_neg = -2.3819763e38  # See gemma/modules.py
        masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)

        probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)

        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")

        out = []
        start = 0
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                end = start + x.shape[1]
                out_einsum = lora.Einsum(
                    shape=(config.num_heads, config.head_dim, config.width),
                    name=_name("attn_vec_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                    lora_config=config.lora_configs.get("attn"),
                )
                out.append(out_einsum("BTNH,NHD->BTD", encoded[:, start:end]))
                start = end
            else:
                out.append(None)

        return out, (k, v)


# =========================
# Dense FFN (original)
# =========================

class GemmoeBlockSparseTop2MLP(nn.Module):
    features: int
    hidden_dim: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        # SwiGLU-style: silu(w1 x) * (w3 x) -> w2
        w1 = nn.Dense(self.hidden_dim, use_bias=False, name="w1")
        w3 = nn.Dense(self.hidden_dim, use_bias=False, name="w3")
        w2 = nn.Dense(self.features, use_bias=False, name="w2")
        a = w1(x)
        b = w3(x)
        y = w2(nn.silu(a) * b)
        return y


# =========================
# MoE FFN (top-2)
# =========================

def top1_routing(router_logits: jnp.ndarray):
    """Top-1 routing for MoE.

    Args:
      router_logits: [B, T, E]
    Returns:
      combine_weights: [B, T, E] (sparse; each token has exactly 1 non-zero expert)
      top_idx: [B, T, 1] selected expert indices
    """
    probs = jax.nn.softmax(router_logits, axis=-1)
    top_vals, top_idx = jax.lax.top_k(probs, k=1)

    B, T, E = router_logits.shape
    one_hot0 = jax.nn.one_hot(top_idx[..., 0], E, dtype=router_logits.dtype)
    combine = one_hot0 * top_vals
    return combine, top_idx


class Router(nn.Module):
    """Atomic skill router with scaled-identity-initialized kernel."""
    hidden_dim: int
    num_experts: int
    kernel_scale: float = 1.0

    def setup(self):
        scale = self.kernel_scale
        def kernel_init(key, shape, dtype=jnp.float32):
            if shape[0] < self.num_experts:
                raise ValueError("hidden_dim must be >= num_experts")
            eye = jnp.eye(self.num_experts, dtype=dtype) * scale
            if shape[0] > self.num_experts:
                eye = jnp.pad(eye, ((0, shape[0] - self.num_experts), (0, 0)))
            return eye
        self.route = nn.Dense(
            features=self.num_experts,
            use_bias=False,
            kernel_init=kernel_init,
        )

    @nn.compact
    def __call__(self, x):
        return self.route(x)


class GemmoeSparseMoeBlock(nn.Module):
    """Sparse MoE block with one shared expert and N atomic-skill experts."""
    config: Config

    def setup(self):
        self.num_extra_experts = getattr(self.config, "num_local_experts", 1)
        self.hidden_dim = self.config.mlp_dim
        self.num_experts = 1 + self.num_extra_experts

        # Expert 0: shared expert
        experts = [
            lora.FeedForward(
                features=1024,
                hidden_dim=4096,
                name="expert_0",
                lora_config=self.config.lora_configs.get("ffn"),
            )
        ]
        # Atomic-skill experts
        for e in range(1, self.num_experts):
            experts.append(
                GemmoeBlockSparseTop2MLP(
                    features=self.config.width,
                    hidden_dim=self.hidden_dim,
                    name=f"expert_{e}",
                )
            )
        self.experts = experts

    def __call__(self, x, combine_weights, deterministic=True):
        expert_outs = [expert(x) for expert in self.experts]
        expert_outs = jnp.stack(expert_outs, axis=2)  # [B, T, E, D]
        y = jnp.einsum("bte,bted->btd", combine_weights, expert_outs)
        return y.astype(x.dtype)


# =========================
# Transformer Block
# =========================
@at.typecheck
class Block(nn.Module):
    """Transformer block."""

    configs: tuple[Config, ...]

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()

    @nn.compact
    def __call__(self, xs, kv_cache, positions, attn_mask, adarms_cond, cond, deterministic=True):  # noqa: FBT002
        xs = sharding.activation_sharding_constraint(xs)
        drop = nn.Dropout(self.dropout, self.dropout_bdims) if self.dropout else (lambda x, _: x)

        attn = Attention(configs=self.configs, name="attn")

        pre_attn = []
        for i, x in enumerate(xs):
            if x is not None:
                x, gate = RMSNorm(name=_name("pre_attention_norm", i))(x, adarms_cond[i])  # noqa: PLW2901
            pre_attn.append(x)

        pre_attn = sharding.activation_sharding_constraint(pre_attn)
        post_attn, kv_cache = attn(pre_attn, positions, attn_mask, kv_cache)
        post_attn = jax.tree.map(lambda x: drop(x, deterministic), post_attn)
        post_attn = sharding.activation_sharding_constraint(post_attn)
        xs = jax.tree.map(lambda x, y: x + y, xs, post_attn)
        xs = sharding.activation_sharding_constraint(xs)

        out = []
        gates = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                x, gate = RMSNorm(name=_name("pre_ffw_norm", i))(x, adarms_cond[i])  # noqa: PLW2901
                if getattr(config, "num_local_experts", 1) > 1:
                    x = GemmoeSparseMoeBlock(config, name=_name("moe", i))(x, cond, deterministic=deterministic)
                else:
                    x = lora.FeedForward(  # noqa: PLW2901
                        features=config.width,
                        hidden_dim=config.mlp_dim,
                        name=_name("mlp", i),
                        lora_config=config.lora_configs.get("ffn"),
                    )(x)
            out.append(x)
            gates.append(gate if x is not None else None)

        out = sharding.activation_sharding_constraint(out)
        out = jax.tree.map(lambda x: drop(x, deterministic), out)
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, out, gates, strict=True)]
        xs = sharding.activation_sharding_constraint(xs)

        return xs, kv_cache

KVCache: TypeAlias = tuple[at.Float[at.Array, "l b _t _k _h"], at.Float[at.Array, "l b _t _v _h"]]


# =========================
# Top-level Module
# =========================
@at.typecheck
class Module(nn.Module):
    """Transformer model, supporting a mixture of different weights for different tokens."""

    # accepts multiple experts/configs (e.g., gemma_2b + moe_gemma)
    configs: Sequence[Config]  # list of configs, one for each expert
    embed_dtype: str

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()
    adarms: bool = False
    router_kernel_scale: float = 1.0

    def setup(self):
        assert all(config.depth == self.configs[0].depth for config in self.configs)

        self.embedder = Embedder(
            vocab_size=PALIGEMMA_VOCAB_SIZE,
            embed_dim=self.configs[0].width,
            name="embedder",
        )
        block_cls = nn.remat(
            Block,
            prevent_cse=False,
            static_argnums=(6,),
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(0, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast),
            length=self.configs[0].depth,
        )(
            configs=self.configs,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
        )
        self.num_extra_experts = getattr(self.configs[1], "num_local_experts", 1)
        self.router = Router(
            self.configs[1].width, self.num_extra_experts,
            kernel_scale=self.router_kernel_scale, name="router",
        )
        self.final_norms = [RMSNorm(name=_name("final_norm", i)) for i in range(len(self.configs))]

    @at.typecheck
    def embed(self, tokens: at.Int[at.Array, "b t"]) -> at.Float[at.Array, "b t d"]:
        return self.embedder.encode(tokens).astype(self.embed_dtype)

    @at.typecheck
    def embedder_decode(self, embedded: at.Float[at.Array, "b t d"]) -> at.Float[at.Array, "b t v"]:
        return self.embedder.decode(embedded)

    @at.typecheck
    def __call__(
        self,
        embedded: Sequence[at.Float[at.Array, "b _t _d"] | None],
        positions: at.Int[at.Array, "b t"],
        mask: at.Bool[at.Array, "b t s"],
        adarms_cond: Sequence[at.Float[at.Array, "b _d"] | None] | None,
        cond: at.Float[at.Array, "b _t _d"] | None,
        *,
        kv_cache: KVCache | None = None,
        deterministic: bool = True,
    ) -> tuple[Sequence[at.Float[at.Array, "b _t _d"] | None], KVCache]:
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
        mask = jnp.asarray(mask)[:, None, :, :]
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)

        if cond is not None:
            logits = self.router(cond)
            extra_weights, top_idx = top1_routing(logits)
            shared_w = 1.0 - jnp.sum(extra_weights, axis=-1, keepdims=True)
            combine_weights = jnp.concatenate([shared_w, extra_weights], axis=-1)
        else:
            extra_weights = None
            combine_weights = None

        embedded, kv_cache = self.layers(
            embedded, kv_cache, positions, mask, adarms_cond, combine_weights, deterministic
        )

        assert all(e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None)

        return [
            f(e, a)[0] if e is not None else e for f, e, a in zip(self.final_norms, embedded, adarms_cond, strict=True)
        ], kv_cache


    def init(self, use_adarms: Sequence[bool]):
        """Convenience method for initializing all parameters, necessary due to the quirks of linen."""
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))

        self(
            [jnp.zeros((1, 1, c.width)) for c in self.configs],
            jnp.zeros((1, len(self.configs)), dtype=jnp.int32),
            jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool),
            [jnp.zeros((1, c.width)) if u else None for u, c in zip(use_adarms, self.configs, strict=True)],
            jnp.zeros((1, 1, self.configs[1].width), dtype=jnp.bfloat16),
        )
