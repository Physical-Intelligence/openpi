# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Three-stream Gemma module with a *hard-routed* trace MoE expert.

Differences vs. ``gemmoe.py``:

  - Supports three streams in the same joint attention. Stream layout is:
        index 0 : PaliGemma VLM (dense FFN)
        index 1 : action expert  (dense FFN, like pi05 ``gemma_300m``)
        index 2 : trace expert   (hard-routed MoE FFN)

  - The trace stream's MoE has no shared expert and no learned router.
    Routing is provided directly by the caller as a sparse one-hot
    ``[B, T, K]`` ``hard_combine_weights`` tensor (K = number of trace experts).
    Skill-id → expert-id is decided outside the model (see
    :func:`openpi.models.tokenizer.embed_sigma`) and the resulting one-hot
    flows through the network at every layer.

  - We ALWAYS run only ONE non-VLM stream at a time (planning vs execution),
    so the third config entry can be omitted to keep parameter count down by
    not instantiating an MoE block when not needed. In practice we keep all
    three configs alive to keep code paths simple; calls just pass ``None``
    for the unused stream's tokens.

This file imports the reusable low-level pieces from ``gemmoe.py`` so we do
not duplicate Attention / RMSNorm / Embedder / etc.
"""

from collections.abc import Sequence
import dataclasses
from typing import Literal, TypeAlias

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp

from openpi.models.gemmoe import (
    Config,
    Embedder,
    FeedForward as DenseFeedForward,  # used for streams without MoE
    GemmoeBlockSparseTop2MLP,
    KVCache,
    PALIGEMMA_VOCAB_SIZE,
    RMSNorm,
    _apply_rope,
    _gated_residual,
    _name,
)
import openpi.models.lora as lora
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding


# ---------------------------------------------------------------------------
# Variant configs for the trace expert. We add a few new variants here that
# request "hard MoE" behavior. The Config dataclass itself is shared with
# gemmoe.py, but `num_local_experts` here means the *exact* number of
# experts (no shared expert is added).
# ---------------------------------------------------------------------------

Variant = Literal[
    "trace_moe_dummy",
    "trace_moe_gemma_300m",  # full FT trace expert (1024 width, 18 layers, 5 experts)
    "trace_moe_gemma_300m_lora",  # LoRA variant (LoRA on attn+ffn)
    "trace_moe_gemma_300m_2e",  # full FT action expert, 2 experts (table-tasks: PICKUP_FROM vs PLACE_*)
    "trace_moe_small",  # shrunk MoE for combined-MoE variant: 512 width, 2048 mlp, 5 experts
    "trace_moe_small_2e",  # shrunk trace MoE, 2 experts (table-tasks: PICKUP_FROM vs PLACE_*)
    "trace_moe_small_dummy",  # tiny version of trace_moe_small for sandbox tests
]


def get_trace_config(variant: Variant) -> Config:
    if variant == "trace_moe_dummy":
        return Config(
            width=64,
            depth=4,
            mlp_dim=128,
            num_heads=8,
            num_kv_heads=1,
            head_dim=16,
            num_local_experts=5,
            num_experts_per_tok=1,
        )
    if variant == "trace_moe_gemma_300m":
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            num_local_experts=5,
            num_experts_per_tok=1,
        )
    if variant == "trace_moe_gemma_300m_lora":
        # NOTE: per the design we keep the trace expert fully trainable even
        # in the "LoRA" variant, but we expose a LoRA-shaped variant for
        # completeness. The user's design uses full FT for the trace expert,
        # so this variant is currently unused but kept for symmetry.
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            num_local_experts=5,
            num_experts_per_tok=1,
            lora_configs={
                "attn": lora.LoRAConfig(rank=32, alpha=32.0),
                "ffn": lora.LoRAConfig(rank=32, alpha=32.0),
            },
        )
    if variant == "trace_moe_gemma_300m_2e":
        # Same shape as ``trace_moe_gemma_300m`` (full-size action MoE) but with only
        # 2 skill experts, for the physical-robot table-tasks dataset whose 3 skills
        # (PICKUP_FROM, PLACE_ON, PLACE_IN) route onto 2 experts per the skill mapping
        # in ``embed_sigma()`` / ``trace_utils.skill_to_expert_id`` (PICKUP_FROM -> 0,
        # PLACE_ON/PLACE_IN -> 1). Width/depth/head-shape are locked by the joint-
        # attention asserts against ``gemma_2b``.
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            num_local_experts=2,
            num_experts_per_tok=1,
        )
    if variant == "trace_moe_small":
        # Recipe C from traceVLA_moe_design.md: half-width, half-mlp_dim trace MoE
        # used in the combined-MoE variant (Pi0TraceVLAMoe). Both width and mlp_dim
        # are halved vs `trace_moe_gemma_300m`; depth, head shape, and K are locked
        # by the joint-attention asserts and the 5-skill routing semantics.
        # Trace MoE FFN params: 18 * 5 * 3 * 512 * 2048 ≈ 283 M (down from 1.13 B).
        return Config(
            width=512,
            depth=18,
            mlp_dim=2048,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            num_local_experts=5,
            num_experts_per_tok=1,
        )
    if variant == "trace_moe_small_2e":
        # 2-expert sibling of ``trace_moe_small`` (shrunk trace MoE) for the table-tasks
        # combined-MoE variant. Same width=512/mlp_dim=2048/depth=18 as ``trace_moe_small``;
        # only ``num_local_experts`` differs (2 instead of 5). Randomly initialized at
        # train start (shape mismatch vs pi05_base, identical to ``trace_moe_small``).
        return Config(
            width=512,
            depth=18,
            mlp_dim=2048,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            num_local_experts=2,
            num_experts_per_tok=1,
        )
    if variant == "trace_moe_small_dummy":
        # Sandbox-test dummy for `trace_moe_small`. Same shape contracts (different
        # `width`/`mlp_dim` from the action-stream `trace_moe_dummy`) so we can
        # exercise the joint-attention asserts with both streams MoE.
        return Config(
            width=32,
            depth=4,
            mlp_dim=64,
            num_heads=8,
            num_kv_heads=1,
            head_dim=16,
            num_local_experts=5,
            num_experts_per_tok=1,
        )
    raise ValueError(f"Unknown trace variant: {variant}")


# ---------------------------------------------------------------------------
# Attention (3-stream variant). Identical structure to gemmoe.Attention but
# tolerates a list of three streams instead of two. We re-implement here to
# avoid hidden coupling to the 2-stream version's debug asserts.
# ---------------------------------------------------------------------------

@at.typecheck
class Attention(nn.Module):
    """Joint attention over an arbitrary number of streams (typ. 3 here)."""

    configs: Sequence[Config]

    @nn.compact
    def __call__(self, xs, positions, attn_mask, kv_cache):
        # Sanity: shared head dim / num_heads / num_kv_heads across streams.
        assert all(c.head_dim == self.configs[0].head_dim for c in self.configs)
        assert all(c.num_heads == self.configs[0].num_heads for c in self.configs)
        assert all(c.num_kv_heads == self.configs[0].num_kv_heads for c in self.configs)

        dtype = next(x.dtype for x in xs if x is not None)

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
        big_neg = -2.3819763e38
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


# ---------------------------------------------------------------------------
# Hard-routed MoE FFN. No shared expert, exactly K experts. Combine weights
# come in pre-computed and one-hot.
# ---------------------------------------------------------------------------

class HardMoeBlock(nn.Module):
    """Sparse MoE block with K skill-specific experts, no shared expert.

    The combine_weights tensor is provided externally (one-hot per token)
    so that *no learned router parameters exist*. Each expert is a SwiGLU FFN.
    """

    config: Config

    def setup(self):
        self.num_experts = int(self.config.num_local_experts)
        if self.num_experts < 1:
            raise ValueError("HardMoeBlock requires num_local_experts >= 1")
        self.hidden_dim = int(self.config.mlp_dim)
        self.experts = [
            GemmoeBlockSparseTop2MLP(
                features=self.config.width,
                hidden_dim=self.hidden_dim,
                name=f"expert_{e}",
            )
            for e in range(self.num_experts)
        ]

    def __call__(self, x, combine_weights, deterministic=True):  # noqa: ARG002
        # x: [B, T, D]; combine_weights: [B, T, K] (one-hot expected for hard routing).
        expert_outs = [expert(x) for expert in self.experts]
        expert_outs = jnp.stack(expert_outs, axis=2)  # [B, T, K, D]
        y = jnp.einsum("btk,btkd->btd", combine_weights, expert_outs)
        return y.astype(x.dtype)


# ---------------------------------------------------------------------------
# Block: per-layer joint attention + per-stream FFN/MoE.
# ---------------------------------------------------------------------------

@at.typecheck
class TraceBlock(nn.Module):
    configs: tuple[Config, ...]
    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()

    @nn.compact
    def __call__(
        self,
        xs,
        kv_cache,
        positions,
        attn_mask,
        adarms_cond,
        hard_combine_weights,  # [B, T_trace, K] or None
        deterministic=True,  # noqa: FBT002
    ):
        xs = sharding.activation_sharding_constraint(xs)
        drop = nn.Dropout(self.dropout, self.dropout_bdims) if self.dropout else (lambda x, _: x)

        attn = Attention(configs=self.configs, name="attn")

        pre_attn = []
        for i, x in enumerate(xs):
            if x is not None:
                x, _gate_attn = RMSNorm(name=_name("pre_attention_norm", i))(x, adarms_cond[i])  # noqa: PLW2901
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
                if int(getattr(config, "num_local_experts", 1)) > 1:
                    # hard-routed MoE FFN, no shared expert
                    x = HardMoeBlock(config, name=_name("moe", i))(  # noqa: PLW2901
                        x,
                        hard_combine_weights,
                        deterministic=deterministic,
                    )
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


# ---------------------------------------------------------------------------
# Top-level Module: 3-stream Gemma with hard-routed trace MoE.
# ---------------------------------------------------------------------------

@at.typecheck
class TraceModule(nn.Module):
    """Gemma trunk over an arbitrary number of streams, with hard-routed MoE
    on streams whose config has ``num_local_experts > 1``.
    """

    configs: Sequence[Config]
    embed_dtype: str
    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()

    def setup(self):
        assert all(c.depth == self.configs[0].depth for c in self.configs)

        self.embedder = Embedder(
            vocab_size=PALIGEMMA_VOCAB_SIZE,
            embed_dim=self.configs[0].width,
            name="embedder",
        )
        block_cls = nn.remat(
            TraceBlock,
            prevent_cse=False,
            static_argnums=(6,),  # static `deterministic`
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(0, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast),
            length=self.configs[0].depth,
        )(
            configs=tuple(self.configs),
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
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
        hard_combine_weights: at.Float[at.Array, "b _t _k"] | None,
        *,
        kv_cache: KVCache | None = None,
        deterministic: bool = True,
    ) -> tuple[Sequence[at.Float[at.Array, "b _t _d"] | None], KVCache]:
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
        mask = jnp.asarray(mask)[:, None, :, :]
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)

        embedded, kv_cache = self.layers(
            embedded, kv_cache, positions, mask, adarms_cond, hard_combine_weights, deterministic
        )

        assert all(e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None)

        out = []
        for f, e, a in zip(self.final_norms, embedded, adarms_cond, strict=True):
            if e is None:
                out.append(None)
            else:
                out.append(f(e, a)[0])
        return out, kv_cache

    def init(self, use_adarms: Sequence[bool]):
        """Convenience init method (mirrors gemmoe.Module.init)."""
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))

        # Number of MoE experts on whichever stream uses MoE (we expect at most one).
        num_moe_experts = max((int(c.num_local_experts) for c in self.configs), default=1)
        # If no MoE: pass a tiny placeholder so shapes are well-defined.
        cond = jnp.zeros((1, 1, num_moe_experts), dtype=jnp.bfloat16)

        self(
            [jnp.zeros((1, 1, c.width)) for c in self.configs],
            jnp.zeros((1, len(self.configs)), dtype=jnp.int32),
            jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool),
            [jnp.zeros((1, c.width)) if u else None for u, c in zip(use_adarms, self.configs, strict=True)],
            cond,
        )
