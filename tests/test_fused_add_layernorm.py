"""Test fused Add + LayerNorm CUDA kernel.

uv run python tests/test_fused_add_layernorm.py
"""

import time

import torch
import torch.nn as nn


def test_correctness():
    """Test that fused operation matches PyTorch reference."""
    print("Testing Fused Add + LayerNorm correctness...")

    try:
        import openpi_cuda_lib as cuda_ops
    except ImportError:
        print("✗ openpi_cuda_lib not available, skipping test")
        return

    # Test dimensions (SiGLIP)
    batch_size = 1
    seq_len = 256
    hidden_size = 1152

    # Create inputs
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.bfloat16)

    # Create LayerNorm
    layernorm = nn.LayerNorm(hidden_size, eps=1e-6).cuda().to(torch.bfloat16)

    # Reference: PyTorch
    with torch.no_grad():
        sum_ref = x + residual
        output_ref = layernorm(sum_ref)

    # CUDA kernel
    with torch.no_grad():
        output_cuda, sum_cuda = cuda_ops.fused_add_layernorm(x, residual, layernorm.weight, layernorm.bias, 1e-6)

    # Compare results
    max_diff = torch.max(torch.abs(output_ref - output_cuda)).item()
    sum_diff = torch.max(torch.abs(sum_ref - sum_cuda)).item()
    mean_diff = torch.mean(torch.abs(output_ref - output_cuda)).item()
    relative_diff = torch.mean(torch.abs(output_ref - output_cuda) / (torch.abs(output_ref) + 1e-8)).item()

    print(f"\n{'=' * 60}")
    print("Correctness Results:")
    print(f"{'=' * 60}")
    print(f"Output shape: {output_cuda.shape}")
    print(f"Max absolute difference: {max_diff:.6e}")
    print(f"Max sum difference: {sum_diff:.6e}")
    print(f"Mean absolute difference: {mean_diff:.6e}")
    print(f"Mean relative difference: {relative_diff:.6e}")
    print(f"{'=' * 60}")

    # Looser tolerance for bfloat16
    tolerance_atol = 1e-2
    tolerance_rtol = 1e-2

    is_close_out = torch.allclose(output_ref, output_cuda, atol=tolerance_atol, rtol=tolerance_rtol)
    is_close_sum = torch.allclose(sum_ref, sum_cuda, atol=tolerance_atol, rtol=tolerance_rtol)

    if is_close_out and is_close_sum:
        print("✓ Correctness test PASSED!")
    else:
        print("✗ Correctness test FAILED!")


def benchmark():
    """Benchmark fused operation vs separate operations."""
    print("\n" + "=" * 60)
    print("Performance Benchmark")
    print("=" * 60)

    try:
        import openpi_cuda_lib as cuda_ops
    except ImportError:
        print("✗ openpi_cuda_lib not available, skipping benchmark")
        return

    # SiGLIP dimensions
    batch_size = 1
    seq_len = 256
    hidden_size = 1152

    num_warmup = 50
    num_iterations = 200

    # Test bfloat16 (what SiGLIP uses)
    dtype = torch.bfloat16

    print(f"\nBenchmarking {dtype}...")

    x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=dtype)
    residual = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=dtype)
    layernorm = nn.LayerNorm(hidden_size, eps=1e-6).cuda().to(dtype)

    # Warmup - PyTorch
    for _ in range(num_warmup):
        with torch.no_grad():
            sum_val = x + residual
            _ = layernorm(sum_val)
    torch.cuda.synchronize()

    # Benchmark - PyTorch
    start = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            sum_val = x + residual
            _ = layernorm(sum_val)
    torch.cuda.synchronize()
    time_pytorch = (time.time() - start) / num_iterations * 1000

    # Warmup - CUDA
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = cuda_ops.fused_add_layernorm(x, residual, layernorm.weight, layernorm.bias, 1e-6)
    torch.cuda.synchronize()

    # Benchmark - CUDA
    start = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = cuda_ops.fused_add_layernorm(x, residual, layernorm.weight, layernorm.bias, 1e-6)
    torch.cuda.synchronize()
    time_cuda = (time.time() - start) / num_iterations * 1000

    speedup = time_pytorch / time_cuda

    print(f"  PyTorch (add + LayerNorm): {time_pytorch:.4f} ms")
    print(f"  CUDA (fused):              {time_cuda:.4f} ms")
    print(f"  Speedup:                   {speedup:.2f}x")

    # Calculate impact across 54 calls (27 layers * 2 per layer)
    total_savings = (time_pytorch - time_cuda) * 54
    print(f"\n  Total savings per forward pass (54 calls): {total_savings:.2f} ms")
    print("=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("Fused Add + LayerNorm CUDA Kernel Tests")
    print("=" * 60)

    test_correctness()
    benchmark()

    print("\n✓ All tests completed!")
