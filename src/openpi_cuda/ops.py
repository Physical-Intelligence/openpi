"""
Custom op registration for torch.compile compatibility.

This module wraps the raw CUDA ops from openpi_cuda_lib and registers them
with torch.library so torch.compile can trace through them without graph breaks.

Usage:
    # Import this module to register the ops
    import openpi_cuda.ops

    # Then call via torch.ops
    output = torch.ops.openpi_cuda.fused_bias_gelu(input, bias)
"""

import openpi_cuda_lib as _cuda_ops
import torch
from torch.library import custom_op, register_fake

LIBRARY_NAME = "openpi_cuda"


def _ensure_compatible(t: torch.Tensor, ref_dtype: torch.dtype = None) -> torch.Tensor:
    """Ensure tensor is contiguous and optionally cast to reference dtype.

    This is critical for max-autotune compatibility:
    - contiguous() ensures consistent memory layout for CUDA graphs
    - dtype casting prevents AMP-induced type mismatches that cause NaNs
    """
    t = t.contiguous()
    if ref_dtype is not None and t.dtype != ref_dtype:
        t = t.to(dtype=ref_dtype)
    return t


# =============================================================================
# fused_bias_gelu: output = GELU(input + bias)
# =============================================================================
@custom_op(f"{LIBRARY_NAME}::fused_bias_gelu", mutates_args=())
def fused_bias_gelu(input: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    # Inductor might fuse a cast into the input, passing bf16 while bias is fp32.
    # We enforce input.dtype on the bias to prevent kernel bit-misinterpretation.
    input = input.contiguous()
    bias = _ensure_compatible(bias, ref_dtype=input.dtype)
    return _cuda_ops.fused_bias_gelu(input, bias)


@register_fake(f"{LIBRARY_NAME}::fused_bias_gelu")
def fused_bias_gelu_fake(input: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return input.new_empty(input.shape)


# =============================================================================
# fused_geglu: output = GELU(gate) * up
# =============================================================================
@custom_op(f"{LIBRARY_NAME}::fused_geglu", mutates_args=())
def fused_geglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    # Gate and Up projections must match dtypes and layout
    gate = gate.contiguous()
    up = _ensure_compatible(up, ref_dtype=gate.dtype)
    return _cuda_ops.fused_geglu(gate, up)


@register_fake(f"{LIBRARY_NAME}::fused_geglu")
def fused_geglu_fake(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return gate.new_empty(gate.shape)


# =============================================================================
# rmsnorm: RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * (1 + weight)
# =============================================================================
@custom_op(f"{LIBRARY_NAME}::rmsnorm", mutates_args=())
def rmsnorm(input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Weights in mixed-precision training are often FP32 while input is BF16.
    # Cast weight to input.dtype to ensure kernel compatibility.
    input = input.contiguous()
    weight = _ensure_compatible(weight, ref_dtype=input.dtype)
    return _cuda_ops.rmsnorm(input, weight, eps)


@register_fake(f"{LIBRARY_NAME}::rmsnorm")
def rmsnorm_fake(input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return input.new_empty(input.shape)


# =============================================================================
# fused_add_layernorm: output = LayerNorm(x + residual)
# Returns tuple: (normalized_output, sum_output)
# =============================================================================
@custom_op(f"{LIBRARY_NAME}::fused_add_layernorm", mutates_args=())
def fused_add_layernorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Critical: x and residual usually must match dtypes for add fusion
    x = x.contiguous()
    residual = _ensure_compatible(residual, ref_dtype=x.dtype)
    gamma = _ensure_compatible(gamma, ref_dtype=x.dtype)
    beta = _ensure_compatible(beta, ref_dtype=x.dtype)
    return _cuda_ops.fused_add_layernorm(x, residual, gamma, beta, eps)


@register_fake(f"{LIBRARY_NAME}::fused_add_layernorm")
def fused_add_layernorm_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    return x.new_empty(x.shape), x.new_empty(x.shape)
