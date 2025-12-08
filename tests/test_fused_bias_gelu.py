"""Test fused Bias + GELU CUDA kernel.

uv run python tests/test_fused_bias_gelu.py
"""

import time

import torch
import torch.nn as nn


def gelu_tanh_pytorch(x):
    """PyTorch reference implementation of GELU with tanh approximation."""
    return nn.functional.gelu(x, approximate="tanh")


def test_correctness():
    """Test that fused operation matches PyTorch reference."""
    print("Testing Fused Bias + GELU correctness...")

    try:
        import openpi_cuda_lib as cuda_ops
    except ImportError:
        print("✗ openpi_cuda_lib not available, skipping test")
        return

    # Test dimensions (similar to SiGLIP MLP)
    batch_size = 2
    seq_len = 256
    features = 4304  # intermediate_size in SiGLIP

    # Create input and bias
    torch.manual_seed(42)
    input_tensor = torch.randn(batch_size, seq_len, features, device="cuda", dtype=torch.float32)
    bias = torch.randn(features, device="cuda", dtype=torch.float32)

    # Reference: PyTorch
    with torch.no_grad():
        output_ref = gelu_tanh_pytorch(input_tensor + bias)

    # CUDA kernel
    with torch.no_grad():
        output_cuda = cuda_ops.fused_bias_gelu(input_tensor, bias)

    # Compare results
    max_diff = torch.max(torch.abs(output_ref - output_cuda)).item()
    mean_diff = torch.mean(torch.abs(output_ref - output_cuda)).item()
    relative_diff = torch.mean(torch.abs(output_ref - output_cuda) / (torch.abs(output_ref) + 1e-8)).item()

    print(f"\n{'='*60}")
    print("Correctness Results:")
    print(f"{'='*60}")
    print(f"Output shape: {output_cuda.shape}")
    print(f"Max absolute difference: {max_diff:.6e}")
    print(f"Mean absolute difference: {mean_diff:.6e}")
    print(f"Mean relative difference: {relative_diff:.6e}")
    print(f"{'='*60}")

    # Check if results are close
    tolerance_atol = 1e-5
    tolerance_rtol = 1e-4

    is_close = torch.allclose(output_ref, output_cuda, atol=tolerance_atol, rtol=tolerance_rtol)

    if is_close:
        print("✓ Correctness test PASSED!")
    else:
        print("✗ Correctness test FAILED!")
        print(f"  Tolerance: atol={tolerance_atol}, rtol={tolerance_rtol}")

    assert is_close, f"Test failed: max_diff={max_diff}, mean_diff={mean_diff}"


def test_different_dtypes():
    """Test with different dtypes."""
    print("\nTesting different dtypes...")

    try:
        import openpi_cuda_lib as cuda_ops
    except ImportError:
        print("✗ openpi_cuda_lib not available, skipping test")
        return

    batch_size = 2
    seq_len = 16
    features = 128

    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        torch.manual_seed(42)
        input_tensor = torch.randn(batch_size, seq_len, features, device="cuda", dtype=dtype)
        bias = torch.randn(features, device="cuda", dtype=dtype)

        with torch.no_grad():
            output_ref = gelu_tanh_pytorch(input_tensor + bias)
            output_cuda = cuda_ops.fused_bias_gelu(input_tensor, bias)

        # Use looser tolerance for half precision
        if dtype == torch.float32:
            atol, rtol = 1e-5, 1e-4
        else:
            atol, rtol = 1e-2, 1e-2

        is_close = torch.allclose(output_ref, output_cuda, atol=atol, rtol=rtol)

        if is_close:
            print(f"  ✓ {dtype} test passed")
        else:
            max_diff = torch.max(torch.abs(output_ref - output_cuda)).item()
            print(f"  ✗ {dtype} test failed (max_diff={max_diff:.6e})")


def benchmark():
    """Benchmark fused operation vs separate operations."""
    print("\n" + "="*60)
    print("Performance Benchmark")
    print("="*60)

    try:
        import openpi_cuda_lib as cuda_ops
    except ImportError:
        print("✗ openpi_cuda_lib not available, skipping benchmark")
        return

    # SiGLIP MLP dimensions
    batch_size = 4
    seq_len = 256  # 16x16 patches
    features = 4304  # intermediate_size

    num_warmup = 50
    num_iterations = 200

    # Test different dtypes
    for dtype in [torch.float32, torch.bfloat16]:
        print(f"\nBenchmarking {dtype}...")

        input_tensor = torch.randn(batch_size, seq_len, features, device="cuda", dtype=dtype)
        bias = torch.randn(features, device="cuda", dtype=dtype)

        # Warmup - PyTorch
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = gelu_tanh_pytorch(input_tensor + bias)
        torch.cuda.synchronize()

        # Benchmark - PyTorch
        start = time.time()
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = gelu_tanh_pytorch(input_tensor + bias)
        torch.cuda.synchronize()
        time_pytorch = (time.time() - start) / num_iterations * 1000

        # Warmup - CUDA
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = cuda_ops.fused_bias_gelu(input_tensor, bias)
        torch.cuda.synchronize()

        # Benchmark - CUDA
        start = time.time()
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = cuda_ops.fused_bias_gelu(input_tensor, bias)
        torch.cuda.synchronize()
        time_cuda = (time.time() - start) / num_iterations * 1000

        speedup = time_pytorch / time_cuda

        print(f"  PyTorch (bias + GELU): {time_pytorch:.4f} ms")
        print(f"  CUDA (fused):          {time_cuda:.4f} ms")
        print(f"  Speedup:               {speedup:.2f}x")


if __name__ == "__main__":
    print("="*60)
    print("Fused Bias + GELU CUDA Kernel Tests")
    print("="*60)

    test_correctness()
    test_different_dtypes()
    benchmark()

    print("\n✓ All tests completed!")
