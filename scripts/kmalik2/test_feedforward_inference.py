#!/usr/bin/env python3
"""
Test inference on a single FeedForward block from Gemma.
This script creates a minimal FeedForward block and runs forward pass without sharding.
"""

import os
import time
import psutil
import jax
import jax.numpy as jnp
from openpi.models import lora

# Set JAX to use CPU for testing
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_cpu_enable_fast_math=false"

# Memory limit in MB
MAX_MEMORY_MB = 1000  # 1GB limit


def get_memory_usage_mb():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def check_memory_limit(operation_name, max_memory_mb=MAX_MEMORY_MB):
    """Check if memory usage exceeds limit and raise error if so."""
    current_memory = get_memory_usage_mb()
    if current_memory > max_memory_mb:
        raise MemoryError(
            f"Memory limit exceeded in {operation_name}!\n"
            f"  Current memory: {current_memory:.1f} MB\n"
            f"  Limit: {max_memory_mb} MB\n"
            f"  Excess: {current_memory - max_memory_mb:.1f} MB"
        )
    print(f"  Memory usage: {current_memory:.1f} MB (limit: {max_memory_mb} MB)")


def estimate_memory_requirements(batch_size, seq_len, model_dim, hidden_dim, dtype=jnp.float32):
    """Estimate memory requirements for FeedForward block before creation."""
    print("Estimating memory requirements...")
    
    # Calculate parameter memory
    gating_params = model_dim * hidden_dim
    linear_params = hidden_dim * model_dim
    total_params = gating_params + linear_params
    
    # Memory per parameter (in bytes)
    param_size = jnp.dtype(dtype).itemsize
    param_memory = total_params * param_size
    
    # Calculate activation memory
    input_memory = batch_size * seq_len * model_dim * param_size
    hidden_memory = batch_size * seq_len * hidden_dim * param_size
    output_memory = batch_size * seq_len * model_dim * param_size
    activation_memory = input_memory + hidden_memory + output_memory
    
    # Total estimated memory
    total_estimated_memory = param_memory + activation_memory
    
    # Convert to MB
    param_memory_mb = param_memory / (1024 * 1024)
    activation_memory_mb = activation_memory / (1024 * 1024)
    total_estimated_memory_mb = total_estimated_memory / (1024 * 1024)
    
    print(f"  Parameter memory: {param_memory_mb:.1f} MB")
    print(f"  Activation memory: {activation_memory_mb:.1f} MB")
    print(f"  Total estimated: {total_estimated_memory_mb:.1f} MB")
    
    return {
        'param_memory_mb': param_memory_mb,
        'activation_memory_mb': activation_memory_mb,
        'total_estimated_memory_mb': total_estimated_memory_mb,
        'total_params': total_params
    }


def check_memory_requirements(batch_size, seq_len, model_dim, hidden_dim, max_memory_mb=MAX_MEMORY_MB):
    """Check if estimated memory requirements exceed limit before creating block."""
    current_memory = get_memory_usage_mb()
    available_memory = max_memory_mb - current_memory
    
    print(f"Current memory usage: {current_memory:.1f} MB")
    print(f"Available memory: {available_memory:.1f} MB")
    
    # Estimate memory requirements
    memory_est = estimate_memory_requirements(batch_size, seq_len, model_dim, hidden_dim)
    
    if memory_est['total_estimated_memory_mb'] > available_memory:
        raise MemoryError(
            f"Insufficient memory for FeedForward block!\n"
            f"  Required: {memory_est['total_estimated_memory_mb']:.1f} MB\n"
            f"  Available: {available_memory:.1f} MB\n"
            f"  Shortage: {memory_est['total_estimated_memory_mb'] - available_memory:.1f} MB\n"
            f"  Consider reducing batch_size ({batch_size}) or hidden_dim ({hidden_dim})"
        )
    
    print(f"Memory check: PASS (estimated {memory_est['total_estimated_memory_mb']:.1f} MB < {available_memory:.1f} MB available)")
    return memory_est


def create_dummy_input(batch_size=1, seq_len=10, model_dim=512):
    """Create dummy input for FeedForward block."""
    return jnp.ones((batch_size, seq_len, model_dim), dtype=jnp.float32)


def create_feedforward_block(model_dim=512, hidden_dim=2048, input_data=None):
    """Create a single FeedForward block from Gemma and compile it."""
    print(f"Creating FeedForward block: model_dim={model_dim}, hidden_dim={hidden_dim}")
    
    # Create FeedForward block without LoRA
    ff_block = lora.FeedForward(
        features=model_dim,
        hidden_dim=hidden_dim,
        lora_config=None
    )
    
    if input_data is not None:
        print("  Initializing and compiling...")
        start = time.time()
        
        # Initialize the block
        params = ff_block.init(jax.random.key(0), input_data)
        check_memory_limit("after initialization")
        
        # JIT compile the forward pass
        @jax.jit
        def forward_fn(params, x):
            return ff_block.apply(params, x)
        
        # First call (compilation)
        _ = forward_fn(params, input_data)
        check_memory_limit("after compilation")
        
        compile_time = time.time() - start
        print(f"  Compilation time: {compile_time:.3f}s")
        
        return ff_block, params, forward_fn
    else:
        return ff_block, None, None


def run_forward_pass(forward_fn, params, input_data):
    """Run forward pass on pre-compiled FeedForward block."""
    print("Running forward pass...")
    
    # Warmup run to ensure JIT compilation is complete
    print("  Warmup run...")
    _ = forward_fn(params, input_data).block_until_ready()
    
    # Actual timed run
    print("  Timed run...")
    start = time.time()
    output = forward_fn(params, input_data)
    output.block_until_ready()
    forward_time = time.time() - start
    print(f"  Forward pass time: {forward_time:.3f}s")
    
    return output, forward_time


def test_consistency(forward_fn, params, input_data, num_runs=100):
    """Test consistency across multiple runs."""
    print(f"Testing consistency across {num_runs} runs...")
    
    outputs = []
    
    # Time all runs together
    print("  Running all tests...")
    start_total = time.time()
    
    for i in range(num_runs):
        output = forward_fn(params, input_data).block_until_ready()
        outputs.append(output)
    
    total_time = time.time() - start_total
    avg_time = total_time / num_runs
    
    print(f"  Total time for {num_runs} runs: {total_time:.3f}s")
    print(f"  Average time per run: {avg_time:.3f}s")
    
    # Check consistency
    all_consistent = True
    for i in range(1, len(outputs)):
        max_diff = jnp.max(jnp.abs(outputs[0] - outputs[i]))
        if max_diff >= 1e-6:
            all_consistent = False
            break
    
    print(f"  Consistency: {'PASS' if all_consistent else 'FAIL'}")
    
    return all_consistent, avg_time


def main():
    print("FeedForward Block Inference Test")
    print("=" * 50)
    
    # Configuration
    batch_size = 64
    seq_len = 128
    model_dim = 512
    hidden_dim = 2048
    
    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Model dim: {model_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print()
    
    # Check memory requirements before creating anything
    print("Pre-flight memory check:")
    memory_est = check_memory_requirements(batch_size, seq_len, model_dim, hidden_dim)
    print()
    
    # Create dummy input
    input_data = create_dummy_input(batch_size, seq_len, model_dim)
    print(f"Input shape: {input_data.shape}")
    check_memory_limit("after creating input data")
    print()
    
    # Create FeedForward block with compilation
    ff_block, params, forward_fn = create_feedforward_block(model_dim, hidden_dim, input_data)
    print()
    
    # Run forward pass
    output, forward_time = run_forward_pass(forward_fn, params, input_data)
    check_memory_limit("after forward pass")
    print()
    
    # Show output
    print("Output:")
    print(f"  Shape: {output.shape}")
    print(f"  Dtype: {output.dtype}")
    print(f"  Norm: {jnp.linalg.norm(output):.6f}")
    print()
    
    # Test consistency
    consistent, avg_consistency_time = test_consistency(forward_fn, params, input_data, num_runs=10)
    print()
    
    # Performance summary
    print("Performance Summary:")
    print(f"  Single forward pass time: {forward_time:.3f}s")
    print(f"  Average consistency run time: {avg_consistency_time:.3f}s")
    print(f"  Consistency: {'PASS' if consistent else 'FAIL'}")
    print()
    
    print("FeedForward inference test completed!")


if __name__ == "__main__":
    main()