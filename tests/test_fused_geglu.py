"""Test fused GeGLU CUDA kernel.

uv run python tests/test_fused_geglu.py
"""

import time

import torch
import torch.nn as nn


def geglu_pytorch(gate, up):
    """PyTorch reference implementation of GeGLU (GELU Gated Linear Unit)."""
    return nn.functional.gelu(gate, approximate="tanh") * up


def test_correctness():
    """Test that fused operation matches PyTorch reference."""
    print("Testing Fused GeGLU correctness...")

    try:
        import openpi_cuda_lib as cuda_ops
    except ImportError:
        print("✗ openpi_cuda_lib not available, skipping test")
        return

    # Test dimensions (similar to Gemma MLP)
    batch_size = 2
    seq_len = 128
    intermediate_size = 16384  # Gemma-2B intermediate size

    # Create input tensors (gate and up projections)
    torch.manual_seed(42)
    gate = torch.randn(batch_size, seq_len, intermediate_size, device="cuda", dtype=torch.float32)
    up = torch.randn(batch_size, seq_len, intermediate_size, device="cuda", dtype=torch.float32)

    # Reference: PyTorch
    with torch.no_grad():
        output_ref = geglu_pytorch(gate, up)

    # CUDA kernel
    with torch.no_grad():
        output_cuda = cuda_ops.fused_geglu(gate, up)

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
    features = 128

    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        torch.manual_seed(42)
        gate = torch.randn(batch_size, seq_len, features, device="cuda", dtype=dtype)
        up = torch.randn(batch_size, seq_len, features, device="cuda", dtype=dtype)

        with torch.no_grad():
            output_ref = geglu_pytorch(gate, up)
            output_cuda = cuda_ops.fused_geglu(gate, up)

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

    test_shapes = [
        (1, 100),  # 1D batch
        (10, 256),  # 2D
        (2, 128, 512),  # 3D
        (4, 64, 32, 128),  # 4D
    ]

    for shape in test_shapes:
        torch.manual_seed(42)
        gate = torch.randn(*shape, device="cuda", dtype=torch.float32)
        up = torch.randn(*shape, device="cuda", dtype=torch.float32)

        with torch.no_grad():
            output_ref = geglu_pytorch(gate, up)
            output_cuda = cuda_ops.fused_geglu(gate, up)

        is_close = torch.allclose(output_ref, output_cuda, atol=1e-5, rtol=1e-4)

        if is_close:
            print(f"  ✓ Shape {shape} test passed")
        else:
            max_diff = torch.max(torch.abs(output_ref - output_cuda)).item()
            print(f"  ✗ Shape {shape} test failed (max_diff={max_diff:.6e})")


def benchmark():
    """Benchmark fused GeGLU vs separate operations."""
    print("\n" + "=" * 60)
    print("Performance Benchmark")
    print("=" * 60)

    try:
        import openpi_cuda_lib as cuda_ops
    except ImportError:
        print("✗ openpi_cuda_lib not available, skipping benchmark")
        return

    # Gemma MLP dimensions
    batch_size = 4
    seq_len = 128
    intermediate_size = 16384  # Gemma-2B intermediate size

    num_warmup = 50
    num_iterations = 200

    # Test different dtypes
    for dtype in [torch.float32, torch.bfloat16]:
        print(f"\nBenchmarking {dtype}...")

        gate = torch.randn(batch_size, seq_len, intermediate_size, device="cuda", dtype=dtype)
        up = torch.randn(batch_size, seq_len, intermediate_size, device="cuda", dtype=dtype)

        # Warmup - PyTorch
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = geglu_pytorch(gate, up)
        torch.cuda.synchronize()

        # Benchmark - PyTorch (separate GELU + multiply)
        start = time.time()
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = geglu_pytorch(gate, up)
        torch.cuda.synchronize()
        time_pytorch = (time.time() - start) / num_iterations * 1000

        # Warmup - CUDA
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = cuda_ops.fused_geglu(gate, up)
        torch.cuda.synchronize()

        # Benchmark - CUDA (fused)
        start = time.time()
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = cuda_ops.fused_geglu(gate, up)
        torch.cuda.synchronize()
        time_cuda = (time.time() - start) / num_iterations * 1000

        speedup = time_pytorch / time_cuda

        print(f"  PyTorch (GELU + multiply): {time_pytorch:.4f} ms")
        print(f"  CUDA (fused):              {time_cuda:.4f} ms")
        print(f"  Speedup:                   {speedup:.2f}x")

        # Memory bandwidth analysis
        # Each operation reads gate and up, writes output (3 arrays)
        num_elements = batch_size * seq_len * intermediate_size
        if dtype == torch.float32:  # noqa: SIM108
            bytes_per_element = 4
        else:  # bfloat16
            bytes_per_element = 2

        # PyTorch: reads gate (GELU), writes temp, reads temp+up (multiply), writes output
        # = 4 reads + 2 writes = 6 * elements * bytes
        pytorch_bytes = 6 * num_elements * bytes_per_element

        # CUDA fused: reads gate+up, writes output
        # = 2 reads + 1 write = 3 * elements * bytes
        cuda_bytes = 3 * num_elements * bytes_per_element

        pytorch_bandwidth = pytorch_bytes / (time_pytorch / 1000) / 1e9  # GB/s
        cuda_bandwidth = cuda_bytes / (time_cuda / 1000) / 1e9  # GB/s

        print(f"  PyTorch bandwidth: {pytorch_bandwidth:.1f} GB/s")
        print(f"  CUDA bandwidth:    {cuda_bandwidth:.1f} GB/s")
        print(f"  Memory traffic saved: {100 * (1 - cuda_bytes / pytorch_bytes):.1f}%")


if __name__ == "__main__":
    print("=" * 60)
    print("Fused GeGLU CUDA Kernel Tests")
    print("=" * 60)

    test_correctness()
    test_different_dtypes()
    test_different_shapes()
    benchmark()

    print("\n✓ All tests completed!")
