import math
import re

import flax.linen as nn
import flax.struct as struct
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

import openpi.shared.array_typing as at
from openpi.training import sharding as sharding_utils


@struct.dataclass
class LoRAConfig:
    """Configuration for LoRA."""

    # LoRA rank.
    rank: int
    # LoRA scaling factor.
    alpha: float = 1.0
    # Initialization function for LoRA parameters.
    init_fn: nn.initializers.Initializer = nn.initializers.normal(stddev=0.01)
    # Enable rank-stabilized LoRA: https://arxiv.org/pdf/2312.03732
    rslora: bool = False
    # Axes in the weight to apply LoRA to. Should typically be the last two axes.
    axes: tuple[int, int] = (-2, -1)
    # Axis label which is used by LoRA in einsum equations. Must not be present in the original equation.
    label: str = "L"

    @property
    def scaling_value(self) -> float:
        return self.alpha / math.sqrt(self.rank) if self.rslora else self.alpha / self.rank


class Einsum(nn.Module):
    """Einsum with LoRA support. Can be used as a drop-in replacement for the Gemma Einsum."""

    # Shape of the weight.
    shape: tuple[int, ...]
    # Initialization function for the weight.
    init_fn: nn.initializers.Initializer = nn.initializers.zeros
    # If not None, apply LoRA to the weight.
    lora_config: LoRAConfig = None

    def setup(self):
        self.w = self.param("w", self.init_fn, self.shape)

        if config := self.lora_config:
            # Setup LoRA parameters.
            shape_a, shape_b = list(self.shape), list(self.shape)
            shape_a[config.axes[1]] = config.rank
            shape_b[config.axes[0]] = config.rank
            self.w_a = self.param("lora_a", config.init_fn, shape_a)
            self.w_b = self.param("lora_b", config.init_fn, shape_b)

    @nn.compact
    def __call__(self, eqn: str, x):
        dtype = x.dtype  # original dtype, could be half-precision
        result = jnp.einsum(eqn, x, self.w.astype(dtype))

        if config := self.lora_config:
            eqn_a, eqn_b = self._make_lora_eqns(eqn)
            lora = jnp.einsum(eqn_a, x, self.w_a.astype(dtype))
            lora = jnp.einsum(eqn_b, lora, self.w_b.astype(dtype))
            result = result + lora * config.scaling_value

        return result

    def _make_lora_eqns(self, eqn: str) -> tuple[str, str]:
        if "L" in eqn:
            raise ValueError(f"L already in eqn: {eqn}")
        if not (m := re.match("(.*),(.*)->(.*)", eqn)):
            raise ValueError(f"Unsupported einsum eqn: {eqn}")
        lhs, rhs, out = m.groups()

        assert self.lora_config is not None
        a_label, b_label = (rhs[x] for x in self.lora_config.axes)
        label = self.lora_config.label

        a_rhs = rhs.replace(b_label, label)
        a_out = out.replace(b_label, label)
        eqn_a = f"{lhs},{a_rhs}->{a_out}"

        b_rhs = rhs.replace(a_label, label)
        eqn_b = f"{a_out},{b_rhs}->{out}"

        return eqn_a, eqn_b


class FeedForward(nn.Module):
    """Feed forward module."""

    features: int
    hidden_dim: int
    # If not None, apply LoRA to the weight.
    lora_config: LoRAConfig = None
    # Optional input sharding constraint function
    input_sharding_constraint: callable = None
    # Optional output sharding constraint function  
    output_sharding_constraint: callable = None

    def setup(self):
        self.w_gating = self.param(
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            (2, self.features, self.hidden_dim),
        )
        self.w_linear = self.param(
            "linear",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.hidden_dim, self.features),
        )
        self.w_gating_lora = None
        self.w_linear_lora = None
        if self.lora_config:
            # Setup LoRA parameters.
            # TODO: follow up with a simplified init_fn api.
            self.w_gating_lora = (
                self.param("gating_einsum_lora_a", self.lora_config.init_fn, (2, self.features, self.lora_config.rank)),
                self.param(
                    "gating_einsum_lora_b", self.lora_config.init_fn, (2, self.lora_config.rank, self.hidden_dim)
                ),
            )
            self.w_linear_lora = (
                self.param("linear_lora_a", self.lora_config.init_fn, (self.hidden_dim, self.lora_config.rank)),
                self.param("linear_lora_b", self.lora_config.init_fn, (self.lora_config.rank, self.features)),
            )

    @nn.compact
    def __call__(self, x):
        dtype = x.dtype  # original dtype, could be half-precision
        
        # Apply input sharding constraint if provided
        if self.input_sharding_constraint is not None:
            x = self.input_sharding_constraint(x)
        
        ff_gate = self._dot(
            x,
            self.w_gating[0],
            None if self.w_gating_lora is None else (self.w_gating_lora[0][0], self.w_gating_lora[1][0]),
        )
        gate_value = nn.gelu(ff_gate)

        ff1 = self._dot(
            x,
            self.w_gating[1],
            None if self.w_gating_lora is None else (self.w_gating_lora[0][1], self.w_gating_lora[1][1]),
        )
        activations = gate_value * ff1

        outputs = self._dot(activations, self.w_linear, self.w_linear_lora)
        
        # Apply output sharding constraint if provided
        if self.output_sharding_constraint is not None:
            outputs = self.output_sharding_constraint(outputs)
        
        assert outputs.dtype == dtype
        return outputs

    def _dot(self, x: at.Array, w: at.Array, lora_weights) -> at.Array:
        base = jnp.dot(x, w.astype(x.dtype))
        if lora_weights is None:
            return base
        return base + jnp.dot(jnp.dot(x, lora_weights[0].astype(x.dtype)), lora_weights[1].astype(x.dtype))
    
    def megatron_tensor_parallel_sharding_info(self):
        """Return parameter names for tensor parallel sharding.
        
        Returns:
            dict: {
                'column_parallel': list of parameter names that should be column-parallel sharded,
                'row_parallel': list of parameter names that should be row-parallel sharded
            }
        """
        assert self.lora_config == None, "Tensor parallel sharding not supported with LORA"
        info = {
            'column_parallel': ['gating_einsum'],  # Column parallel: shard hidden_dim (last axis)
            'row_parallel': ['linear']             # Row parallel: shard hidden_dim (first axis)
        }
        
        # This might work for LORA but need to test
        #if self.lora_config:
        #    info['column_parallel'].extend(['gating_einsum_lora_a', 'gating_einsum_lora_b'])
        #    info['row_parallel'].extend(['linear_lora_a', 'linear_lora_b'])
        
        return info
