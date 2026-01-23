"""Optimized Triton kernels for AMD MI350."""

import torch
import triton
import triton.language as tl


# ============================================================================
# RMSNorm Triton Kernel
# ============================================================================

@triton.jit
def _rms_norm_fwd_kernel(
    X_ptr, W_ptr, Y_ptr,
    stride_x_row, stride_y_row,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    """RMSNorm forward kernel."""
    row_idx = tl.program_id(0)
    
    # Compute row start pointers
    x_row_ptr = X_ptr + row_idx * stride_x_row
    y_row_ptr = Y_ptr + row_idx * stride_y_row
    
    # Load row and compute variance
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N
    
    x = tl.load(x_row_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Compute RMS
    variance = tl.sum(x * x, axis=0) / N
    rrms = 1.0 / tl.sqrt(variance + eps)
    
    # Normalize and scale
    y = x * rrms * w
    
    # Store result
    tl.store(y_row_ptr + col_offsets, y.to(tl.bfloat16), mask=mask)


def rms_norm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Triton optimized RMSNorm."""
    assert x.is_contiguous()
    
    # Flatten to 2D
    orig_shape = x.shape
    x_2d = x.view(-1, x.shape[-1])
    M, N = x_2d.shape
    
    # Output tensor
    y = torch.empty_like(x_2d)
    
    # Block size (must be power of 2 and >= N)
    BLOCK_SIZE = triton.next_power_of_2(N)
    
    # Launch kernel
    grid = (M,)
    _rms_norm_fwd_kernel[grid](
        x_2d, weight, y,
        x_2d.stride(0), y.stride(0),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return y.view(orig_shape)


# ============================================================================
# Fused GELU + Mul Triton Kernel
# ============================================================================

@triton.jit
def _gelu_tanh_and_mul_kernel(
    X_ptr, Y_ptr,
    stride_x_row, stride_y_row,
    N,  # hidden_size (half of input last dim)
    BLOCK_SIZE: tl.constexpr,
):
    """Fused GELU(tanh approx) + Mul kernel.
    
    Input: [*, 2*N] where first N is gate, second N is up
    Output: [*, N] = GELU(gate) * up
    """
    row_idx = tl.program_id(0)
    
    x_row_ptr = X_ptr + row_idx * stride_x_row
    y_row_ptr = Y_ptr + row_idx * stride_y_row
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N
    
    # Load gate and up
    gate = tl.load(x_row_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_row_ptr + N + col_offsets, mask=mask, other=0.0).to(tl.float32)
    
    # GELU tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    k = 0.7978845608028654  # sqrt(2/pi)
    gate_cubed = gate * gate * gate
    inner = k * (gate + 0.044715 * gate_cubed)
    # Compute tanh via exp
    exp_2x = tl.math.exp(2.0 * inner)
    tanh_val = (exp_2x - 1.0) / (exp_2x + 1.0)
    gelu_gate = 0.5 * gate * (1.0 + tanh_val)
    
    # Multiply
    y = gelu_gate * up
    
    tl.store(y_row_ptr + col_offsets, y.to(tl.bfloat16), mask=mask)


def gelu_tanh_and_mul_triton(x: torch.Tensor) -> torch.Tensor:
    """Triton optimized GELU tanh + mul."""
    assert x.is_contiguous()
    
    # Input shape: [*, 2*N], output shape: [*, N]
    orig_shape = x.shape
    N = x.shape[-1] // 2
    x_2d = x.view(-1, x.shape[-1])
    M = x_2d.shape[0]
    
    # Output tensor
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = triton.next_power_of_2(N)
    
    grid = (M,)
    _gelu_tanh_and_mul_kernel[grid](
        x_2d, y,
        x_2d.stride(0), y.stride(0),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return y.view(orig_shape[:-1] + (N,))


# ============================================================================
# Fused SiLU + Mul Triton Kernel
# ============================================================================

@triton.jit
def _silu_and_mul_kernel(
    X_ptr, Y_ptr,
    stride_x_row, stride_y_row,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused SiLU + Mul kernel."""
    row_idx = tl.program_id(0)
    
    x_row_ptr = X_ptr + row_idx * stride_x_row
    y_row_ptr = Y_ptr + row_idx * stride_y_row
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N
    
    gate = tl.load(x_row_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_row_ptr + N + col_offsets, mask=mask, other=0.0).to(tl.float32)
    
    # SiLU = x * sigmoid(x)
    silu_gate = gate * tl.sigmoid(gate)
    y = silu_gate * up
    
    tl.store(y_row_ptr + col_offsets, y.to(tl.bfloat16), mask=mask)


def silu_and_mul_triton(x: torch.Tensor) -> torch.Tensor:
    """Triton optimized SiLU + mul."""
    assert x.is_contiguous()
    
    orig_shape = x.shape
    N = x.shape[-1] // 2
    x_2d = x.view(-1, x.shape[-1])
    M = x_2d.shape[0]
    
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = triton.next_power_of_2(N)
    
    grid = (M,)
    _silu_and_mul_kernel[grid](
        x_2d, y,
        x_2d.stride(0), y.stride(0),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return y.view(orig_shape[:-1] + (N,))


# ============================================================================
# Fused Add + RMSNorm Triton Kernel
# ============================================================================

@triton.jit
def _fused_add_rms_norm_kernel(
    X_ptr, R_ptr, W_ptr, Y_ptr, RS_ptr,
    stride_x_row, stride_r_row, stride_y_row, stride_rs_row,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused Add + RMSNorm kernel.
    
    Y = RMSNorm(X + R, W)
    RS = X + R (residual sum for next layer)
    """
    row_idx = tl.program_id(0)
    
    x_row = X_ptr + row_idx * stride_x_row
    r_row = R_ptr + row_idx * stride_r_row
    y_row = Y_ptr + row_idx * stride_y_row
    rs_row = RS_ptr + row_idx * stride_rs_row
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N
    
    x = tl.load(x_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Add
    hidden = x + r
    
    # Store residual sum
    tl.store(rs_row + col_offsets, hidden.to(tl.bfloat16), mask=mask)
    
    # RMSNorm
    variance = tl.sum(hidden * hidden, axis=0) / N
    rrms = 1.0 / tl.sqrt(variance + eps)
    y = hidden * rrms * w
    
    tl.store(y_row + col_offsets, y.to(tl.bfloat16), mask=mask)


def fused_add_rms_norm_triton(
    x: torch.Tensor, 
    residual: torch.Tensor, 
    weight: torch.Tensor, 
    eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton optimized fused add + RMSNorm."""
    assert x.is_contiguous() and residual.is_contiguous()
    
    orig_shape = x.shape
    x_2d = x.view(-1, x.shape[-1])
    r_2d = residual.view(-1, residual.shape[-1])
    M, N = x_2d.shape
    
    y = torch.empty_like(x_2d)
    rs = torch.empty_like(x_2d)
    
    BLOCK_SIZE = triton.next_power_of_2(N)
    
    grid = (M,)
    _fused_add_rms_norm_kernel[grid](
        x_2d, r_2d, weight, y, rs,
        x_2d.stride(0), r_2d.stride(0), y.stride(0), rs.stride(0),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return y.view(orig_shape), rs.view(orig_shape)


# ============================================================================
# Benchmark
# ============================================================================

def benchmark_triton_ops():
    """Benchmark Triton kernels vs PyTorch."""
    import time
    import torch.nn.functional as F
    
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    
    print("=" * 60)
    print("TRITON OPS BENCHMARK")
    print("=" * 60)
    
    batch, seq, hidden = 8, 512, 2048
    
    def bench(fn, name, warmup=10, iters=100):
        for _ in range(warmup):
            _ = fn()
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(iters):
            _ = fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) / iters * 1000
    
    # RMSNorm
    print("\n1. RMSNorm")
    x = torch.randn(batch, seq, hidden, dtype=dtype, device=device)
    w = torch.ones(hidden, dtype=dtype, device=device)
    
    def eager_rms():
        var = x.float().pow(2).mean(-1, keepdim=True)
        return (x * torch.rsqrt(var + 1e-6) * w).to(dtype)
    
    eager_t = bench(eager_rms, "eager")
    triton_t = bench(lambda: rms_norm_triton(x, w), "triton")
    print(f"   Eager:  {eager_t:.3f} ms")
    print(f"   Triton: {triton_t:.3f} ms")
    print(f"   Speedup: {eager_t/triton_t:.2f}x")
    
    # GELU + Mul
    print("\n2. GELU + Mul")
    x2 = torch.randn(batch, seq, hidden * 2, dtype=dtype, device=device)
    
    def eager_gelu_mul():
        gate = x2[..., :hidden]
        up = x2[..., hidden:]
        return F.gelu(gate, approximate='tanh') * up
    
    eager_t = bench(eager_gelu_mul, "eager")
    triton_t = bench(lambda: gelu_tanh_and_mul_triton(x2), "triton")
    print(f"   Eager:  {eager_t:.3f} ms")
    print(f"   Triton: {triton_t:.3f} ms")
    print(f"   Speedup: {eager_t/triton_t:.2f}x")
    
    # SiLU + Mul
    print("\n3. SiLU + Mul")
    
    def eager_silu_mul():
        gate = x2[..., :hidden]
        up = x2[..., hidden:]
        return F.silu(gate) * up
    
    eager_t = bench(eager_silu_mul, "eager")
    triton_t = bench(lambda: silu_and_mul_triton(x2), "triton")
    print(f"   Eager:  {eager_t:.3f} ms")
    print(f"   Triton: {triton_t:.3f} ms")
    print(f"   Speedup: {eager_t/triton_t:.2f}x")
    
    # Fused Add + RMSNorm
    print("\n4. Fused Add + RMSNorm")
    r = torch.randn(batch, seq, hidden, dtype=dtype, device=device)
    
    def eager_add_rms():
        h = x + r
        var = h.float().pow(2).mean(-1, keepdim=True)
        return (h * torch.rsqrt(var + 1e-6) * w).to(dtype), h
    
    eager_t = bench(eager_add_rms, "eager")
    triton_t = bench(lambda: fused_add_rms_norm_triton(x, r, w), "triton")
    print(f"   Eager:  {eager_t:.3f} ms")
    print(f"   Triton: {triton_t:.3f} ms")
    print(f"   Speedup: {eager_t/triton_t:.2f}x")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    benchmark_triton_ops()
