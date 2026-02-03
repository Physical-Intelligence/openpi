"""Test RMSNorm CUDA kernel.

uv run python tests/test_rmsnorm.py
"""

import time

import torch


def rmsnorm_pytorch(x, weight, eps=1e-6):
    """PyTorch reference implementation of RMSNorm."""
    # Compute RMS
    variance = torch.mean(x.float() ** 2, dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    # Apply scale
    return x * (1.0 + weight.float())


def test_correctness():
    """Test that RMSNorm operation matches PyTorch reference."""
    print("Testing RMSNorm correctness...")

    try:
        import openpi_cuda_lib as cuda_ops
    except ImportError:
        print("✗ openpi_cuda_lib not available, skipping test")
        return

    # Test dimensions (Gemma hidden_size = 2048)
    batch_size = 2
    seq_len = 128
    hidden_size = 2048

    # Create input and weight
    torch.manual_seed(42)
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.float32)
    weight = torch.randn(hidden_size, device="cuda", dtype=torch.float32)
    eps = 1e-6

    # Reference: PyTorch
    with torch.no_grad():
        output_ref = rmsnorm_pytorch(input_tensor, weight, eps)

    # CUDA kernel
    with torch.no_grad():
        output_cuda = cuda_ops.rmsnorm(input_tensor, weight, eps)

    # Compare results
    max_diff = torch.max(torch.abs(output_ref - output_cuda)).item()
    mean_diff = torch.mean(torch.abs(output_ref - output_cuda)).item()
    relative_diff = torch.mean(torch.abs(output_ref - output_cuda) / (torch.abs(output_ref) + 1e-8)).item()

    print(f"\n{'=' * 60}")
    print("Correctness Results:")
    print(f"{'=' * 60}")
    print(f"Output shape: {output_cuda.shape}")
    print(f"Max absolute difference: {max_diff:.6e}")
    print(f"Mean absolute difference: {mean_diff:.6e}")
    print(f"Mean relative difference: {relative_diff:.6e}")
    print(f"{'=' * 60}")

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
    hidden_size = 2048

    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        torch.manual_seed(42)
        input_tensor = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
        eps = 1e-6

        with torch.no_grad():
            output_ref = rmsnorm_pytorch(input_tensor, weight, eps).to(dtype)
            output_cuda = cuda_ops.rmsnorm(input_tensor, weight, eps)

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


def test_different_shapes():
    """Test with different tensor shapes."""
    print("\nTesting different shapes...")

    try:
        import openpi_cuda_lib as cuda_ops
    except ImportError:
        print("✗ openpi_cuda_lib not available, skipping test")
        return

    hidden_size = 2048
    weight = torch.randn(hidden_size, device="cuda", dtype=torch.float32)
    eps = 1e-6

    test_shapes = [
        (1, 2048),  # 2D: single token
        (10, 2048),  # 2D: batch of tokens
        (2, 128, 2048),  # 3D: batch, seq_len, hidden
        (4, 64, 2048),  # 3D: different batch size
    ]

    for shape in test_shapes:
        torch.manual_seed(42)
        input_tensor = torch.randn(*shape, device="cuda", dtype=torch.float32)

        with torch.no_grad():
            output_ref = rmsnorm_pytorch(input_tensor, weight, eps)
            output_cuda = cuda_ops.rmsnorm(input_tensor, weight, eps)

        is_close = torch.allclose(output_ref, output_cuda, atol=1e-5, rtol=1e-4)

        if is_close:
            print(f"  ✓ Shape {shape} test passed")
        else:
            max_diff = torch.max(torch.abs(output_ref - output_cuda)).item()
            print(f"  ✗ Shape {shape} test failed (max_diff={max_diff:.6e})")


def benchmark():
    """Benchmark RMSNorm kernel vs PyTorch."""
    print("\n" + "=" * 60)
    print("Performance Benchmark")
    print("=" * 60)

    try:
        import openpi_cuda_lib as cuda_ops
    except ImportError:
        print("✗ openpi_cuda_lib not available, skipping benchmark")
        return

    # Gemma dimensions
    batch_size = 4
    seq_len = 128
    hidden_size = 2048

    num_warmup = 50
    num_iterations = 200

    # Test different dtypes
    for dtype in [torch.float32, torch.bfloat16]:
        print(f"\nBenchmarking {dtype}...")

        input_tensor = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=dtype)
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
        eps = 1e-6

        # Warmup - PyTorch
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = rmsnorm_pytorch(input_tensor, weight, eps)
        torch.cuda.synchronize()

        # Benchmark - PyTorch
        start = time.time()
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = rmsnorm_pytorch(input_tensor, weight, eps)
        torch.cuda.synchronize()
        time_pytorch = (time.time() - start) / num_iterations * 1000

        # Warmup - CUDA
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = cuda_ops.rmsnorm(input_tensor, weight, eps)
        torch.cuda.synchronize()

        # Benchmark - CUDA
        start = time.time()
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = cuda_ops.rmsnorm(input_tensor, weight, eps)
        torch.cuda.synchronize()
        time_cuda = (time.time() - start) / num_iterations * 1000

        speedup = time_pytorch / time_cuda

        print(f"  PyTorch RMSNorm: {time_pytorch:.4f} ms")
        print(f"  CUDA RMSNorm:    {time_cuda:.4f} ms")
        print(f"  Speedup:         {speedup:.2f}x")

        # Memory bandwidth analysis
        num_elements = batch_size * seq_len * hidden_size
        if dtype == torch.float32:  # noqa: SIM108
            bytes_per_element = 4
        else:  # bfloat16
            bytes_per_element = 2

        # PyTorch: reads input, computes variance, reads input again, reads weight, writes output
        # Roughly: 3 reads + 1 write
        pytorch_bytes = 4 * num_elements * bytes_per_element

        # CUDA: reads input once for reduction, reads input+weight for output, writes output
        # Roughly: 3 reads + 1 write (similar, but fused operations)
        cuda_bytes = 4 * num_elements * bytes_per_element

        pytorch_bandwidth = pytorch_bytes / (time_pytorch / 1000) / 1e9  # GB/s
        cuda_bandwidth = cuda_bytes / (time_cuda / 1000) / 1e9  # GB/s

        print(f"  PyTorch bandwidth: {pytorch_bandwidth:.1f} GB/s")
        print(f"  CUDA bandwidth:    {cuda_bandwidth:.1f} GB/s")


if __name__ == "__main__":
    print("=" * 60)
    print("RMSNorm CUDA Kernel Tests")
    print("=" * 60)

    test_correctness()
    test_different_dtypes()
    test_different_shapes()
    benchmark()

    print("\n✓ All tests completed!")
