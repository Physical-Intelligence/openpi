#!/usr/bin/env python3
"""
Shared utilities for inference testing scripts.
Common functionality for testing sharded vs unsharded model blocks.
"""

import argparse
import os
import time
import jax
import jax.numpy as jnp
from jax import profiler
from jax.experimental import mesh_utils, pjit
from jax.sharding import Mesh, PartitionSpec as P
from openpi.training import sharding


def setup_jax_environment(num_devices, platform="cuda"):
    """Set up JAX environment variables BEFORE importing JAX.
    
    This ensures XLA flags and platform settings are properly applied.
    """
    # Check if JAX_PLATFORMS is already set (e.g., from command line)
    if "JAX_PLATFORMS" in os.environ:
        platform = os.environ["JAX_PLATFORMS"]
        print(f"Using JAX_PLATFORMS={platform} from environment")
    
    if platform == "cuda":
        os.environ["JAX_PLATFORMS"] = "cuda"
        # Automatically set CUDA_VISIBLE_DEVICES based on num_devices
        # Only override if not already set by user
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            cuda_devices = ",".join(str(i) for i in range(num_devices))
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
            print(f"Auto-setting CUDA_VISIBLE_DEVICES={cuda_devices} for {num_devices} devices")
        else:
            existing_devices = os.environ["CUDA_VISIBLE_DEVICES"]
            device_count = len([d for d in existing_devices.split(",") if d.strip()])
            if device_count != num_devices:
                print(f"WARNING: CUDA_VISIBLE_DEVICES={existing_devices} specifies {device_count} devices, but --num-shards={num_devices}")
        return
    else:  # cpu
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["XLA_FLAGS"] = f"--xla_cpu_enable_fast_math=false --xla_force_host_platform_device_count={num_devices}"


def create_base_argument_parser(description, block_type="block"):
    """Create base argument parser with common arguments for inference tests.
    
    Args:
        description: Description for the parser
        block_type: Type of block being tested (e.g., "FeedForward", "Attention")
    
    Returns:
        ArgumentParser with common arguments added
    """
    parser = argparse.ArgumentParser(description=description)
    
    # Common arguments for all inference tests
    parser.add_argument("--num-shards", type=int, default=8, 
                       help="Number of shards/devices to use (default: 8)")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size (default: 1)")
    parser.add_argument("--seq-len", type=int, default=128,
                       help="Sequence length (default: 128)")
    parser.add_argument("--model-dim", type=int, default=2048,
                       help="Model dimension (default: 2048)")
    parser.add_argument("--mode", type=str, choices=["debug", "timing", "profile", "profile_both"], default="debug",
                       help="Run mode: 'debug' for detailed shape info, 'timing' for performance timing only, 'profile' for communication analysis, 'profile_both' for both sharded and unsharded profiles (default: debug)")
    parser.add_argument("--trace-dir", type=str, default="/tmp/jax_traces",
                       help="Directory to save profiling traces (default: /tmp/jax_traces)")
    parser.add_argument("--disable-hlo-profile", action="store_true",
                       help="Disable detailed HLO profiling with per-operation timing (default: enabled)")
    parser.add_argument("--platform", type=str, choices=["cuda", "cpu"], default="cuda",
                       help="Platform to use: 'cuda' for NVIDIA GPUs, 'cpu' for CPU (default: cuda)")
    parser.add_argument("--no-jit", action="store_true",
                       help="Run forward passes without JIT compilation (useful for debugging)")
    parser.add_argument("--timing-runs", type=int, default=10,
                       help="Number of timing runs for performance measurement (default: 10)")
    
    return parser


def create_mesh(num_fsdp_devices=4):
    """Create a 2D device mesh for sharding.
    
    Args:
        num_fsdp_devices: Number of devices to use for FSDP dimension
        
    Returns:
        Mesh object with shape (batch=1, fsdp=num_fsdp_devices)
    """
    # Get available devices and use only the requested number
    available_devices = jax.devices()
    if len(available_devices) < num_fsdp_devices:
        raise ValueError(f"Requested {num_fsdp_devices} devices but only {len(available_devices)} available")
    
    # Use only the first num_fsdp_devices devices
    selected_devices = available_devices[:num_fsdp_devices]
    # Create 2D mesh: (batch_dim=1, fsdp_dim=num_fsdp_devices)
    # This matches the structure expected by sharding.activation_sharding_constraint
    dev_mesh = mesh_utils.create_device_mesh((1, num_fsdp_devices), devices=selected_devices)  # 2D (1, N)
    return Mesh(dev_mesh, ('batch', 'fsdp'))


def setup_devices_for_sharding():
    """Set up devices for sharding testing.
    
    Returns:
        int: Number of available devices
    """
    print("Setting up devices for sharding...")

    # Get device information
    device_count = jax.device_count()
    devices = jax.devices()
        
    print(f"  Available devices: {device_count}")
    print(f"  Device types: {[str(d) for d in devices]}")
        
    # Check if we're using GPU
    gpu_devices = [d for d in devices if 'gpu' in str(d).lower() or 'cuda' in str(d).lower()]
    cpu_devices = [d for d in devices if 'cpu' in str(d).lower()]
        
    if gpu_devices:
        print(f"  ✓ GPU devices detected: {len(gpu_devices)}")
        for i, device in enumerate(gpu_devices):
            print(f"    GPU {i}: {device}")
    else:
        print(f"  ⚠ No GPU devices detected - running on CPU")
        for i, device in enumerate(cpu_devices):
            print(f"    CPU {i}: {device}")
        
    # Check JAX backend
    try:
        backend = jax.extend.backend.get_backend()
        print(f"  JAX backend: {backend.platform}")
        if hasattr(backend, 'device_kind'):
            print(f"  Device kind: {backend.device_kind}")
    except Exception as e:
        print(f"  Backend info unavailable: {e}")
        
    if device_count < 2:
        print("  WARNING: Only 1 device available. Sharding will not work properly.")
        return 1
    else:
        print(f"  Using {device_count} devices for sharding.")
        return device_count


def test_device_computation():
    """Test basic device computation and print device information."""
    print("Testing device computation...")
    test_array = jnp.ones((2, 2))
    print(f"  Test array device: {test_array.device}")
    print(f"  Test array backend: {test_array.device.platform}")
    
    # Simple computation test
    result = jnp.dot(test_array, test_array)
    print(f"  Computation result device: {result.device}")
    print(f"  Computation result backend: {result.device.platform}")
    print()


def run_forward_pass_with_profiling(forward_fn, params, input_data, name="forward_pass", sharded=False, mode="profile", num_warmup=3, num_profile_runs=3, trace_dir_base="/tmp/jax_traces", output_extractor=None):
    """Run forward pass with JAX profiling to analyze communication patterns.
    
    Args:
        forward_fn: Function to run forward pass
        params: Model parameters
        input_data: Input data for the forward pass
        name: Name for the profiling run
        sharded: Whether this is a sharded run
        mode: Profiling mode
        num_warmup: Number of warmup runs
        num_profile_runs: Number of profiling runs
        trace_dir_base: Base directory for traces
        output_extractor: Function to extract output for block_until_ready (e.g., lambda x: x[0] for attention)
    
    Returns:
        Tuple of (output, dummy_time)
    """
    print(f"Running {name} with profiling...")
    
    # Create a fixed trace directory name (no timestamp)
    trace_name = f"{'sharded' if sharded else 'unsharded'}_{mode}"
    trace_dir = f"{trace_dir_base}/{trace_name}"
    
    # Enhanced warmup to ensure proper PJIT compilation and device synchronization
    for i in range(num_warmup):
        output = forward_fn(params, input_data, print_shapes=False)
        # Extract output for synchronization based on block type
        sync_output = output_extractor(output) if output_extractor else output
        sync_output.block_until_ready()
    
    # Additional synchronization step to ensure all devices are ready
    sync_output = output_extractor(output) if output_extractor else output
    jax.block_until_ready(sync_output)
    
    print(f"  Starting profiling...")
    # Start profiling after comprehensive warmup
    profiler.start_trace(trace_dir)
    
    try:
        # Run multiple forward passes with profiling for better trace data
        for i in range(num_profile_runs):
            output = forward_fn(params, input_data, print_shapes=False)
            sync_output = output_extractor(output) if output_extractor else output
            sync_output.block_until_ready()
        
        print(f"  {name} completed successfully")
        print(f"  Profiling trace saved to {trace_dir}")
        print(f"  To analyze: tensorboard --logdir={trace_dir}")
        return output, 0.0  # Return dummy time since we're focusing on profiling
        
    finally:
        # Stop profiling
        profiler.stop_trace()


def run_forward_pass(forward_fn, params, input_data, show_shapes=False, with_printing=True, num_warmup=3, num_timing_runs=10, output_extractor=None):
    """Run forward pass and measure time with proper warmup and multiple runs.
    
    Args:
        forward_fn: Function to run forward pass
        params: Model parameters
        input_data: Input data for the forward pass
        show_shapes: Whether to show tensor shapes
        with_printing: Whether to print detailed timing info
        num_warmup: Number of warmup runs
        num_timing_runs: Number of timing runs
        output_extractor: Function to extract output for block_until_ready (e.g., lambda x: x[0] for attention)
    
    Returns:
        Tuple of (output, forward_time)
    """
    if with_printing:
        print("Running forward pass...")
        
        if show_shapes:
            print("  Showing tensor shapes during forward pass:")
        
    else:
        print("Running forward pass (no printing)...")
    
    # Warmup runs to eliminate JAX compilation overhead and trigger PJIT compilation
    for i in range(num_warmup):
        # Only show debug output on the first warmup run
        debug_output = show_shapes and i == 0
        output = forward_fn(params, input_data, print_shapes=debug_output)
        sync_output = output_extractor(output) if output_extractor else output
        sync_output.block_until_ready()
    
    # Timing runs
    times = []
    for i in range(num_timing_runs):
        # Only show debug output on the first timing run
        debug_output = show_shapes and i == 0
        start = time.time()
        output = forward_fn(params, input_data, print_shapes=debug_output)
        sync_output = output_extractor(output) if output_extractor else output
        sync_output.block_until_ready()
        end = time.time()
        times.append(end - start)
    
    # Calculate statistics
    forward_time = sum(times) / len(times)  # Average time
    min_time = min(times)
    max_time = max(times)
    
    # Convert to milliseconds
    times_ms = [t * 1_000 for t in times]
    forward_time_ms = forward_time * 1_000
    min_time_ms = min_time * 1_000
    max_time_ms = max_time * 1_000
    
    if with_printing:
        print(f"  Forward pass time (avg): {forward_time_ms:.4f}ms")
        print(f"  Min time: {min_time_ms:.4f}ms")
        print(f"  Max time: {max_time_ms:.4f}ms")
        print(f"  Times (raw ms): {times_ms}")
    else:
        print(f"  Forward pass time (avg): {forward_time_ms:.4f}ms")
        print(f"  Times (raw ms): {times_ms}")
    
    return output, forward_time


def compare_outputs(output1, output2, name1="Output 1", name2="Output 2", output_extractor=None):
    """Compare two outputs and return whether they are identical.
    
    Args:
        output1, output2: Outputs to compare
        name1, name2: Names for the outputs
        output_extractor: Function to extract comparable output (e.g., lambda x: x[0] for attention)
    
    Returns:
        bool: Whether outputs are identical
    """
    # Extract outputs for comparison
    if output_extractor:
        out1 = output_extractor(output1)
        out2 = output_extractor(output2)
    else:
        out1 = output1
        out2 = output2
    
    print(f"Comparing {name1} vs {name2}:")
    print(f"  {name1} shape: {out1.shape}, dtype: {out1.dtype}")
    print(f"  {name2} shape: {out2.shape}, dtype: {out2.dtype}")
    
    # Calculate norms
    norm1 = jnp.linalg.norm(out1)
    norm2 = jnp.linalg.norm(out2)
    print(f"  {name1} norm: {norm1:.6f}")
    print(f"  {name2} norm: {norm2:.6f}")
    print(f"  Norm difference: {abs(norm1 - norm2):.2e}")
    
    # Calculate differences
    max_diff = jnp.max(jnp.abs(out1 - out2))
    mean_diff = jnp.mean(jnp.abs(out1 - out2))
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")
    
    # Check if identical
    identical = max_diff < 1e-6
    print(f"  Identical: {'YES' if identical else 'NO'}")
    
    return identical


def print_configuration(args, extra_config=None):
    """Print test configuration in a standard format.
    
    Args:
        args: Parsed command line arguments
        extra_config: Dictionary of additional configuration items to print
    """
    print(f"Configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Model dim: {args.model_dim}")
    
    # Print extra configuration if provided
    if extra_config:
        for key, value in extra_config.items():
            print(f"  {key}: {value}")
    
    print(f"  Number of shards: {args.num_shards}")
    print(f"  Platform: {args.platform}")
    print(f"  Mode: {args.mode}")
    print(f"  Sharding strategy: {args.sharding_strategy}")
    print(f"  JIT compilation: {'Disabled' if args.no_jit else 'Enabled'}")
    if args.mode == "timing":
        print(f"  Timing runs: {args.timing_runs}")
    if args.mode in ["profile", "profile_both"]:
        print(f"  Trace directory: {args.trace_dir}")
    print()


def print_performance_comparison(forward_time_unsharded, forward_time_sharded):
    """Print performance comparison in a standard format.
    
    Args:
        forward_time_unsharded: Unsharded forward time in seconds
        forward_time_sharded: Sharded forward time in seconds
    """
    print("=" * 30)
    print("PERFORMANCE COMPARISON")
    print("=" * 30)
    
    forward_time_unsharded_ms = forward_time_unsharded * 1_000
    forward_time_sharded_ms = forward_time_sharded * 1_000
    
    print(f"Unsharded forward time: {forward_time_unsharded_ms:.4f}ms")
    print(f"Sharded forward time: {forward_time_sharded_ms:.4f}ms")
    speedup = forward_time_unsharded / forward_time_sharded if forward_time_sharded > 0 else float('inf')
    print(f"Speedup: {speedup:.2f}x")
    print()


def print_final_summary(args, identical=None, forward_time_sharded=None):
    """Print final summary in a standard format.
    
    Args:
        args: Parsed command line arguments
        identical: Whether outputs are identical (None if not applicable)
        forward_time_sharded: Sharded forward time in seconds (None if not applicable)
    """
    print("=" * 30)
    print("FINAL SUMMARY")
    print("=" * 30)
    
    if args.mode == "debug" and identical is not None:
        print(f"Outputs identical: {'YES' if identical else 'NO'}")
    if args.mode == "timing" and forward_time_sharded is not None:
        forward_time_sharded_ms = forward_time_sharded * 1_000
        print(f"Sharded time: {forward_time_sharded_ms:.4f}ms")
    print()


def print_param_shapes(param_dict, prefix="", sharded=False):
    """Recursively print parameter shapes.
    
    Args:
        param_dict: Dictionary of parameters
        prefix: Prefix for parameter names
        sharded: Whether to show sharding information
    """
    for param_name, param in param_dict.items():
        if isinstance(param, dict):
            print_param_shapes(param, prefix + param_name + ".", sharded)
        else:
            if sharded and hasattr(param, 'sharding') and param.sharding is not None:
                shard_shape = param.sharding.shard_shape(param.shape)
                print(f"    {prefix}{param_name} local shard: {tuple(shard_shape)}")
            else:
                sharding_info = param.sharding if hasattr(param, 'sharding') else 'None'
                print(f"    {prefix}{param_name}: {param.shape}, sharding: {sharding_info}")
