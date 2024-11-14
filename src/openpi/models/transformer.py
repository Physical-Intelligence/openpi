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

"""gemma adaptation for Pi, taken from big_vision.

We follow this einsum axis naming convention:
  B: batch
  T: query length
  S: k/v length
  N: num query heads
  K: num k/v heads
  G: num query heads per k/v head
  H: head dim
  D: d_model ("features")

Notes:
- init seems to be variance(1.0, fan_in, trunc_normal) everywhere.
"""

import dataclasses
from typing import Sequence, Literal  # noqa: UP035

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp

import openpi.base.array_typing as at

PALIGEMMA_VOCAB_SIZE = 257_152


@dataclasses.dataclass
class Config:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int


def get_config(variant: Literal["gemma_300m", "gemma_2b"]) -> Config:
    """Returns config for specified gemma variant."""
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
    elif variant == "gemma_2b":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    raise ValueError(f"Unknown variant: {variant}")


@at.typecheck
class Einsum(nn.Module):
    shape: tuple[int, ...]
    init_fn: nn.initializers.Initializer

    @nn.compact
    def __call__(self, eqn, x):
        dtype = x.dtype  # original dtype, could be half-precision
        w = self.param("w", self.init_fn, self.shape).astype(dtype)
        return jnp.einsum(eqn, x, w)


@at.typecheck
class RMSNorm(nn.Module):
    @nn.compact
    def __call__(self, x):
        dtype = x.dtype  # original dtype, could be half-precision
        scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)  # compute variance in float32
        normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))  # compute normalization in float32
        normed_inputs = normed_inputs * (
            1 + scale
        )  # scale by learned parameter in float32 (matches Flax implementation)
        return normed_inputs.astype(dtype)  # return in original dtype


@at.typecheck
class Embedder(nn.Module):
    """Embedder module."""

    vocab_size: int
    embed_dim: int

    def setup(self):
        self.input_embedding_table = self.param(
            "input_embedding",
            nn.initializers.normal(),
            (self.vocab_size, self.embed_dim),
        )

    def encode(self, x):
        x = self.input_embedding_table[(x,)]
        x *= jnp.sqrt(self.embed_dim).astype(x.dtype)
        return x

    def decode(self, x):
        return jnp.dot(x, self.input_embedding_table.T)


@at.typecheck
class Attention(nn.Module):
    """Attention module."""

    configs: Sequence[Config]

    @nn.compact
    def __call__(self, xs, positions, attn_mask, decode: bool, deterministic: bool = True):
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
                qkvs.append(
                    Einsum(
                        shape=(3, config.num_heads, config.width, config.head_dim),
                        name=_name("qkv_einsum", i),
                        init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    )("BSD,3KDH->3BSKH", x)
                )
            else:
                q = Einsum(
                    shape=(config.num_heads, config.width, config.head_dim),
                    name=_name("q_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                )("BTD,NDH->BTNH", x)
                k, v = Einsum(
                    shape=(2, config.num_kv_heads, config.width, config.head_dim),
                    name=_name("kv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                )("BSD,2KDH->2BSKH", x)
                qkvs.append((q, k, v))

        q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))

        q = _apply_rope(q, positions=positions)
        q *= self.configs[0].head_dim ** -0.5

        k = _apply_rope(k, positions=positions)

        # should still be half-precision here (if input was half-precision)
        assert q.dtype == k.dtype == v.dtype == dtype

        if decode:
            if not self.has_variable("cache", "k_cache"):
                # initial prefill
                self.put_variable("cache", "k_cache", k)
                self.put_variable("cache", "v_cache", v)
            else:
                # decoding
                k = jnp.concatenate([self.get_variable("cache", "k_cache"), k], axis=1)
                v = jnp.concatenate([self.get_variable("cache", "v_cache"), v], axis=1)

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
                out.append(
                    Einsum(
                        shape=(config.num_heads, config.head_dim, config.width),
                        name=_name("attn_vec_einsum", i),
                        init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                    )("BTNH,NHD->BTD", encoded[:, start:end])
                )
                start = end
            else:
                out.append(None)

        return out


@at.typecheck
class FeedForward(nn.Module):
    """Feed forward module."""

    features: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        dtype = x.dtype  # original dtype, could be half-precision
        w_gating = self.param(
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            (2, self.features, self.hidden_dim),
        ).astype(dtype)
        ff_gate = jnp.dot(x, w_gating[0])
        gate_value = nn.gelu(ff_gate)

        ff1 = jnp.dot(x, w_gating[1])
        activations = gate_value * ff1

        w_linear = self.param(
            "linear",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.hidden_dim, self.features),
        ).astype(dtype)
        outputs = jnp.dot(activations, w_linear)
        assert outputs.dtype == dtype
        return outputs


@at.typecheck
class Block(nn.Module):
    """Transformer block."""

    configs: Sequence[Config]

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()

    @nn.compact
    def __call__(self, xs, unused_scan_arg, positions, attn_mask, decode, deterministic=True):
        drop = nn.Dropout(self.dropout, self.dropout_bdims) if self.dropout else lambda x, _: x

        attn = Attention(configs=self.configs, name="attn")

        pre_attn = []
        for i, x in enumerate(xs):
            if x is not None:
                x = RMSNorm(name=_name("pre_attention_norm", i))(x)  # noqa: PLW2901
            pre_attn.append(x)

        post_attn = attn(pre_attn, positions, attn_mask, decode, deterministic)
        post_attn = jax.tree.map(lambda x: drop(x, deterministic), post_attn)
        xs = jax.tree.map(lambda x, y: x + y, xs, post_attn)

        out = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                x = RMSNorm(name=_name("pre_ffw_norm", i))(x)  # noqa: PLW2901
                x = FeedForward(
                    features=config.width,
                    hidden_dim=config.mlp_dim,
                    name=_name("mlp", i),
                )(x)  # noqa: PLW2901
            out.append(x)

        out = jax.tree.map(lambda x: drop(x, deterministic), out)
        xs = jax.tree.map(lambda x, y: x + y, xs, out)

        return xs, unused_scan_arg


@at.typecheck
class Module(nn.Module):
    """Transformer model."""

    configs: Sequence[Config]
    embed_dtype: str

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()  # Every float is dropped independently.

    @nn.compact
    @at.typecheck
    def __call__(
        self,
        *,
        tokens: at.Int[at.Array, "b t"] | None,
        embedded: Sequence[at.Float[at.Array, "b _t _d"] | None] | None,
        positions: at.Int[at.Array, "b t"] | None = None,
        mask: at.Bool[at.Array, "b t s"] | None = None,
        decode: bool = False,
        deterministic: bool = True,
    ) -> at.Float[at.Array, "b t d"] | Sequence[at.Float[at.Array, "b _t _d"] | None]:
        # all experts must have the same depth
        assert all(config.depth == self.configs[0].depth for config in self.configs)

        # embedder for first expert only
        embedder = Embedder(
            vocab_size=PALIGEMMA_VOCAB_SIZE,
            embed_dim=self.configs[0].width,
            name="embedder",
        )

        if tokens is not None:
            # embed only
            assert embedded is None, "Cannot pass both tokens and embedded"
            return embedder.encode(tokens).astype(self.embed_dtype)

        assert embedded is not None and positions is not None and mask is not None

        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)

        mask = mask[:, None, :, :]

        block_cls = nn.remat(
            Block,
            prevent_cse=False,
            static_argnums=(5, 6),  # 0=self, 5=decode, 6=deterministic
            policy=jax.checkpoint_policies.nothing_saveable,
        )

        block = nn.scan(
            block_cls,
            # cache has axis 1 since we want leading dimension to be batch size.
            variable_axes={"params": 0, "cache": 1},
            split_rngs={"params": True, "dropout": True},
            in_axes=nn.broadcast,
            length=self.configs[0].depth,
        )(
            parent=self.scope.push("layers"),
            configs=self.configs,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
        )

        embedded, _ = block(embedded, (), positions, mask, decode, deterministic)

        assert all(e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None)

        return [RMSNorm(name=_name("final_norm", i))(e) if e is not None else e for i, e in enumerate(embedded)]


def _apply_rope(x, *, positions, max_wavelength=10_000):
    """Applies RoPE positions [B, L] to x [B, L, H, D]."""
    freq_exponents = (2.0 / x.shape[-1]) * jnp.arange(x.shape[-1] // 2, dtype=jnp.float32)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None] / timescale[None, None, :]
    radians = radians[..., None, :]
    assert radians.dtype == jnp.float32
    # radians.shape = [...,L,1,d=D/2]
    sin, cos = jnp.sin(radians), jnp.cos(radians)
    x1, x2 = jnp.split(x, 2, axis=-1)
    res = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    assert res.dtype == jnp.float32
    # The original bigvision impl allows RoPE to upcast to float32. It is then immediately downcast again to the cache
    # dtype when in inference mode (but not in training mode). I don't think any of this was intentional. Based on the
    # original DeepMind impl, as well as the widely-used transformers impl, it is ok to always downcast back to bfloat16
    # here.
    return res.astype(x.dtype)


def _name(name, i):
    if i == 0:
        return name
    return f"{name}_{i}"
