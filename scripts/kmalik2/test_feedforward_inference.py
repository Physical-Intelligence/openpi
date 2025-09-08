#!/usr/bin/env python3
"""
Test inference on a single FeedForward block from Gemma.
This script compares unsharded vs sharded FeedForward blocks to ensure identical outputs.

To test with 2 devices:
  CUDA_VISIBLE_DEVICES=0,1 uv run scripts/kmalik2/test_feedforward_inference.py
  # or for CPU with 2 devices:
  JAX_PLATFORMS=cpu uv run scripts/kmalik2/test_feedforward_inference.py

Modes:
  --mode debug: Show detailed tensor shapes and sharding information (default)
  --mode timing: Show only performance timing information
  --mode profile: Generate JAX profiling traces for communication analysis
  --mode profile_both: Generate both sharded and unsharded profiles for comparison
"""

import argparse
import os
import time


# Set up JAX environment variables BEFORE importing JAX
# This ensures XLA flags and platform settings are properly applied
# --- in setup_jax_environment() ---
def setup_jax_environment(num_devices, platform="cuda"):
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
        
# Parse command line arguments to get device count early
parser = argparse.ArgumentParser(description="Test FeedForward block inference with sharding")
parser.add_argument("--num-shards", type=int, default=8, 
                   help="Number of shards/devices to use (default: 8)")
parser.add_argument("--batch-size", type=int, default=1,
                   help="Batch size (default: 1)")
parser.add_argument("--seq-len", type=int, default=128,
                   help="Sequence length (default: 128)")
parser.add_argument("--model-dim", type=int, default=2048,
                   help="Model dimension (default: 2048)")
parser.add_argument("--hidden-dim", type=int, default=512,
                   help="Hidden dimension (default: 512)")
parser.add_argument("--mode", type=str, choices=["debug", "timing", "profile", "profile_both"], default="debug",
                   help="Run mode: 'debug' for detailed shape info, 'timing' for performance timing only, 'profile' for communication analysis, 'profile_both' for both sharded and unsharded profiles (default: debug)")
parser.add_argument("--trace-dir", type=str, default="/tmp/jax_traces",
                   help="Directory to save profiling traces (default: /tmp/jax_traces)")
parser.add_argument("--disable-hlo-profile", action="store_true",
                   help="Disable detailed HLO profiling with per-operation timing (default: enabled)")
parser.add_argument("--platform", type=str, choices=["cuda", "cpu"], default="cuda",
                   help="Platform to use: 'cuda' for NVIDIA GPUs, 'cpu' for CPU (default: cuda)")
parser.add_argument("--sharding-strategy", type=str, choices=["default", "megatron"], default="default",
                   help="Sharding strategy: 'default' for FSDP-style sharding, 'megatron' for tensor parallel (default: default)")
parser.add_argument("--no-jit", action="store_true",
                   help="Run forward passes without JIT compilation (useful for debugging)")
parser.add_argument("--timing-runs", type=int, default=10,
                   help="Number of timing runs for performance measurement (default: 10)")

# Parse args early to get device count
args, _ = parser.parse_known_args()

# Set up JAX environment BEFORE importing JAX
setup_jax_environment(args.num_shards, platform=args.platform)

# Now import JAX and other dependencies
import jax
import jax.numpy as jnp
from jax import profiler
from jax.experimental import pjit
from openpi.models import lora
from openpi.training import sharding




def create_dummy_input(batch_size=1, seq_len=10, model_dim=512):
    """Create dummy input for FeedForward block."""
    return jnp.ones((batch_size, seq_len, model_dim), dtype=jnp.float32)


def get_gpu_memory_mb():
    """Get GPU memory usage in MB using JAX device memory stats"""
    device = jax.devices()[0]
    memory_stats = device.memory_stats()
    return {
        'used_mb': memory_stats['bytes_in_use'] / (1024**2),
        'peak_mb': memory_stats.get('peak_bytes_in_use', memory_stats['bytes_in_use']) / (1024**2),
        'limit_mb': memory_stats.get('bytes_limit', 0) / (1024**2)
    }


def log_memory_usage(stage_name):
    """Log current memory usage at a specific stage"""
    mem = get_gpu_memory_mb()
    print(f"  [MEMORY] {stage_name}: {mem['used_mb']:.1f}MB used, {mem['peak_mb']:.1f}MB peak")
    return mem


from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P

def create_mesh(num_fsdp_devices=4):
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


def megatron_sharding(params_shape, mesh, log=False):
    """Create megatron-style tensor parallel sharding for FeedForward parameters.
    
    In megatron sharding:
    - gating_einsum (shape: [2, model_dim, hidden_dim]) is sharded column-wise (last dim)
    - linear (shape: [hidden_dim, model_dim]) is sharded row-wise (first dim)
    - activations (x) are not sharded
    """
    def _shard_param(kp, param_shape):
        key_path = jax.tree_util.keystr(kp)
        
        if 'gating_einsum' in key_path:
            # Column-wise sharding: shard the hidden_dim (last dimension)
            if log:
                print(f"  Megatron sharding {key_path}: column-wise (last dim)")
            return jax.sharding.NamedSharding(mesh, P(None, None, 'fsdp'))
        elif 'linear' in key_path:
            # Row-wise sharding: shard the hidden_dim (first dimension)  
            if log:
                print(f"  Megatron sharding {key_path}: row-wise (first dim)")
            return jax.sharding.NamedSharding(mesh, P('fsdp', None))
        else:
            # Replicate everything else
            if log:
                print(f"  Megatron sharding {key_path}: replicated")
            return jax.sharding.NamedSharding(mesh, P())
    
    return jax.tree_util.tree_map_with_path(_shard_param, params_shape)
    

def create_feedforward_block(model_dim=512, hidden_dim=2048, input_data=None, sharded=False, mesh=None, sharding_strategy="default", use_jit=True):
    """Create a single FeedForward block from Gemma."""
    print(f"Creating FeedForward block: model_dim={model_dim}, hidden_dim={hidden_dim}, sharded={sharded}, strategy={sharding_strategy}")
    
    # Create FeedForward block without LoRA
    # Set up sharding constraint functions based on strategy
    if sharded and sharding_strategy == "megatron":
        # Megatron: use tensor parallel MLP constraints
        input_constraint = sharding.megatron_mlp_input_constraint
        output_constraint = sharding.megatron_mlp_output_constraint
    elif sharded and sharding_strategy == "default":
        # Default FSDP: use standard activation sharding for both
        input_constraint = sharding.activation_sharding_constraint
        output_constraint = sharding.activation_sharding_constraint
    else:
        # Unsharded: no constraints
        input_constraint = None
        output_constraint = None
    
    ff_block = lora.FeedForward(
        features=model_dim,
        hidden_dim=hidden_dim,
        lora_config=None,
        input_sharding_constraint=input_constraint,
        output_sharding_constraint=output_constraint
    )
    
    if sharded and mesh is not None:
        # Apply sharding to parameters using jitted init with out_shardings
        print(f"  Applying {sharding_strategy} sharding to parameters...")
        
        # Shape evaluation - mesh context is already active from the calling context
        params_shape = jax.eval_shape(ff_block.init, jax.random.key(0), input_data)
        
        if sharding_strategy == "megatron":
            # Get sharding info from the FeedForward block
            tp_info = ff_block.megatron_tensor_parallel_sharding_info()
            param_sharding = sharding.megatron_tensor_parallel_sharding(
                params_shape, 
                mesh, 
                column_parallel_names=tp_info['column_parallel'],
                row_parallel_names=tp_info['row_parallel'],
                log=True
            )
        else:  # default
            param_sharding = sharding.fsdp_sharding(params_shape, mesh, log=True)
            
        print("  Initializing parameters directly into sharded buffers...")
        # Parameter initialization - mesh context should already be active when sharded=True
        params = jax.jit(ff_block.init, out_shardings=param_sharding)(jax.random.key(0), input_data)
        
        # Create forward function based on use_jit flag
        if use_jit:
            # PJIT-compiled forward pass (FeedForward handles all internal sharding)
            def _forward_impl(params, x):
                return ff_block.apply(params, x)
            
            forward_fn_jit = pjit.pjit(
                _forward_impl,
                in_shardings=(param_sharding, None),  # No input sharding constraint, FeedForward handles it
                out_shardings=None,  # No output sharding constraint, FeedForward handles it
            )
        else:
            # Non-JIT version (FeedForward handles internal sharding)
            def forward_fn_no_jit(params, x):
                return ff_block.apply(params, x)
        
        # Wrapper function for debugging with shapes (not JIT-compiled)
        def forward_fn(params, x, print_shapes=False):
            if print_shapes:
                # Show only local shard shapes for weights
                fc1 = params['params']['gating_einsum']
                fc2 = params['params']['linear']
                num_shards = fc1.sharding.mesh.shape['fsdp']
                print(f"    [{sharding_strategy.upper()}] Weight shard shapes (across {num_shards} shards):")
                fc1_shard_shape = fc1.sharding.shard_shape(fc1.shape)
                fc2_shard_shape = fc2.sharding.shard_shape(fc2.shape)
                print(f"    FC1 local shard: {tuple(fc1_shard_shape)}")
                print(f"    FC2 local shard: {tuple(fc2_shard_shape)}")
                
                # Show input shape and sharding
                print(f"    [{sharding_strategy.upper()}] Input shape: {x.shape}, sharding: {x.sharding if hasattr(x, 'sharding') else 'None'}")
                if hasattr(x, 'sharding') and x.sharding is not None:
                    x_shard = tuple(x.sharding.shard_shape(x.shape))
                    print(f"    [{sharding_strategy.upper()}] Input x local shard: {x_shard}")
                
                # Forward pass - show only local shard shapes for activations
                print(f"    [{sharding_strategy.upper()}] Activation shard shapes:")
                
                # Use FeedForward block directly (it handles sharding internally)
                output = ff_block.apply(params, x)
                output_shard = tuple(output.sharding.shard_shape(output.shape))
                print(f"    Final output local shard: {output_shard}")
                
                return output
            else:
                # Use JIT or non-JIT version based on flag
                if use_jit:
                    return forward_fn_jit(params, x)
                else:
                    return forward_fn_no_jit(params, x)
    else:
        # Initialize the block without sharding
        params = ff_block.init(jax.random.key(0), input_data)
        
        # Create forward function based on use_jit flag
        if use_jit:
            # PJIT-compiled forward pass without sharding (for consistency)
            def _forward_impl(params, x):
                return ff_block.apply(params, x)
            forward_fn_jit = pjit.pjit(_forward_impl)
        else:
            # Non-JIT version
            def forward_fn_no_jit(params, x):
                return ff_block.apply(params, x)
        
        # Wrapper function for debugging with shapes (not JIT-compiled)
        def forward_fn(params, x, print_shapes=False):
            if print_shapes:
                # Get weights
                fc1 = params['params']['gating_einsum']
                fc2 = params['params']['linear']
                
                print(f"    [UNSHARDED] Input shape: {x.shape}, sharding: {x.sharding if hasattr(x, 'sharding') else 'None'}")
                if hasattr(x, 'sharding') and x.sharding is not None:
                    x_shard = tuple(x.sharding.shard_shape(x.shape))
                    print(f"    [UNSHARDED] Input x local shard: {x_shard}")
                print(f"    [UNSHARDED] FC1 weights: {fc1.shape}, sharding: {fc1.sharding if hasattr(fc1, 'sharding') else 'None'}")
                print(f"    [UNSHARDED] FC2 weights: {fc2.shape}, sharding: {fc2.sharding if hasattr(fc2, 'sharding') else 'None'}")
                print(f"    [UNSHARDED] Forward pass:")
                
                # Forward pass
                ff_gate = jnp.dot(x, fc1[0])
                gate_value = jax.nn.gelu(ff_gate)
                ff1 = jnp.dot(x, fc1[1])
                activations = gate_value * ff1
                outputs = jnp.dot(activations, fc2)
                
                print(f"    [UNSHARDED] All final_hidden_act on single device: {outputs.shape}")
                
                return outputs
            else:
                # Use JIT or non-JIT version based on flag
                if use_jit:
                    return forward_fn_jit(params, x)
                else:
                    return forward_fn_no_jit(params, x)
    
    # Note: Setup time calculation would need to be added at the beginning of the function
    
    return ff_block, params, forward_fn


def run_forward_pass_with_profiling(forward_fn, params, input_data, name="forward_pass", sharded=False, mode="profile", num_warmup=3, num_profile_runs=3, trace_dir_base="/tmp/jax_traces"):
    """Run forward pass with JAX profiling to analyze communication patterns."""
    print(f"Running {name} with profiling...")
    
    # Create a fixed trace directory name (no timestamp)
    trace_name = f"{'sharded' if sharded else 'unsharded'}_{mode}"
    trace_dir = f"{trace_dir_base}/{trace_name}"
    
    # Enhanced warmup to ensure proper PJIT compilation and device synchronization
    for i in range(num_warmup):
        output = forward_fn(params, input_data, print_shapes=False)
        output.block_until_ready()
    
    # Additional synchronization step to ensure all devices are ready
    jax.block_until_ready(output)
    
    print(f"  Starting profiling...")
    # Start profiling after comprehensive warmup
    profiler.start_trace(trace_dir)
    
    try:
        # Run multiple forward passes with profiling for better trace data
        for i in range(num_profile_runs):
            output = forward_fn(params, input_data, print_shapes=False)
            output.block_until_ready()
        
        print(f"  {name} completed successfully")
        print(f"  Profiling trace saved to {trace_dir}")
        print(f"  To analyze: tensorboard --logdir={trace_dir}")
        return output, 0.0  # Return dummy time since we're focusing on profiling
        
    finally:
        # Stop profiling
        profiler.stop_trace()


def run_forward_pass(forward_fn, params, input_data, show_shapes=False, with_printing=True, num_warmup=3, num_timing_runs=10):
    """Run forward pass and measure time with proper warmup and multiple runs"""
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
        output.block_until_ready()
    
    # Timing runs
    times = []
    for i in range(num_timing_runs):
        # Only show debug output on the first timing run
        debug_output = show_shapes and i == 0
        start = time.time()
        output = forward_fn(params, input_data, print_shapes=debug_output)
        output.block_until_ready()
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


def compare_outputs(output1, output2, name1="Output 1", name2="Output 2"):
    """Compare two outputs and return whether they are identical."""
    print(f"Comparing {name1} vs {name2}:")
    print(f"  {name1} shape: {output1.shape}, dtype: {output1.dtype}")
    print(f"  {name2} shape: {output2.shape}, dtype: {output2.dtype}")
    
    # Calculate norms
    norm1 = jnp.linalg.norm(output1)
    norm2 = jnp.linalg.norm(output2)
    print(f"  {name1} norm: {norm1:.6f}")
    print(f"  {name2} norm: {norm2:.6f}")
    print(f"  Norm difference: {abs(norm1 - norm2):.2e}")
    
    # Calculate differences
    max_diff = jnp.max(jnp.abs(output1 - output2))
    mean_diff = jnp.mean(jnp.abs(output1 - output2))
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")
    
    # Check if identical
    identical = max_diff < 1e-6
    print(f"  Identical: {'YES' if identical else 'NO'}")
    
    return identical


def setup_devices_for_sharding():
    """Set up devices for sharding testing."""
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



def main():
    """Main test function."""
    # Arguments are already parsed at module level
    # JAX environment is already set up before import
    
    print("FeedForward Block Inference Test (Unsharded vs Sharded)")
    print("=" * 60)
    
    # Configuration
    batch_size = args.batch_size
    seq_len = args.seq_len
    model_dim = args.model_dim
    hidden_dim = args.hidden_dim
    
    # Assert that batch size should be divisible by num_shards
    assert batch_size % args.num_shards == 0, f"Batch size ({batch_size}) must be divisible by num_shards ({args.num_shards})"
    
    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Model dim: {model_dim}")
    print(f"  Hidden dim: {hidden_dim}")
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
    
    # Set up devices for sharding
    device_count = setup_devices_for_sharding()
    print()
    
    # Memory tracking
    memory_snapshots = {}
    memory_snapshots['baseline'] = log_memory_usage("Baseline (before model creation)")
    print()
    
    # Create dummy input
    input_data = create_dummy_input(batch_size, seq_len, model_dim)
    print(f"Input shape: {input_data.shape}")
    
    # Test GPU computation
    print("Testing GPU computation...")
    test_array = jnp.ones((2, 2))
    print(f"  Test array device: {test_array.device}")
    print(f"  Test array backend: {test_array.device.platform}")
    
    # Simple computation test
    result = jnp.dot(test_array, test_array)
    print(f"  Computation result device: {result.device}")
    print(f"  Computation result backend: {result.device.platform}")
    print()
    
    # Create mesh for sharding
    mesh = create_mesh(args.num_shards)
    print(f"Mesh shape: {mesh.shape}")
    print()
    
    # Test 1: Unsharded version (only in debug mode)
    if args.mode == "debug":
        print("=" * 30)
        print("TEST 1: UNSHARDED VERSION")
        print("=" * 30)
        
        ff_block_unsharded, params_unsharded, forward_fn_unsharded = create_feedforward_block(
            model_dim, hidden_dim, input_data, sharded=False, use_jit=not args.no_jit
        )
        memory_snapshots['unsharded_model'] = log_memory_usage("After unsharded model creation")
        print()
        
        print("DEBUG RUN (showing shapes):")
        output_unsharded, forward_time_unsharded = run_forward_pass(forward_fn_unsharded, params_unsharded, input_data, show_shapes=True, with_printing=True, num_warmup=0, num_timing_runs=1)
        
        memory_snapshots['unsharded_forward'] = log_memory_usage("After unsharded forward pass")
        print()
    else:
        # For non-debug modes, create dummy values to avoid errors later
        memory_snapshots['unsharded_model'] = {'used_mb': 0, 'peak_mb': 0}
        memory_snapshots['unsharded_forward'] = {'used_mb': 0, 'peak_mb': 0}
        output_unsharded = None
        forward_time_unsharded = 0.0
    
    # Test 2: Sharded version
    print("=" * 30)
    print("TEST 2: SHARDED VERSION")
    print("=" * 30)
    
    with sharding.set_mesh(mesh):
        ff_block_sharded, params_sharded, forward_fn_sharded = create_feedforward_block(
            model_dim, hidden_dim, input_data, sharded=True, mesh=mesh, sharding_strategy=args.sharding_strategy, use_jit=not args.no_jit
        )
        memory_snapshots['sharded_model'] = log_memory_usage("After sharded model creation")
        print()
        
        # Run sharded based on mode
        if args.mode == "debug":
            print("DEBUG RUN (showing shard info):")
            output_sharded, forward_time_sharded = run_forward_pass(forward_fn_sharded, params_sharded, input_data, show_shapes=True, with_printing=True, num_warmup=0, num_timing_runs=1)
        elif args.mode == "profile":
            print("PROFILE RUN (analyzing communication):")
            output_sharded, forward_time_sharded = run_forward_pass_with_profiling(forward_fn_sharded, params_sharded, input_data, "sharded", sharded=True, mode=args.mode, num_warmup=3, num_profile_runs=3, trace_dir_base=args.trace_dir)
        elif args.mode == "profile_both":
            print("PROFILE RUN - SHARDED (analyzing communication):")
            output_sharded, forward_time_sharded = run_forward_pass_with_profiling(forward_fn_sharded, params_sharded, input_data, "sharded", sharded=True, mode="profile", num_warmup=3, num_profile_runs=3, trace_dir_base=args.trace_dir)
        else:  # timing mode
            print("TIMING RUN (no debug output):")
            output_sharded, forward_time_sharded = run_forward_pass(forward_fn_sharded, params_sharded, input_data, show_shapes=False, with_printing=False, num_warmup=3, num_timing_runs=args.timing_runs)
        
        memory_snapshots['sharded_forward'] = log_memory_usage("After sharded forward pass")
        print()
    
    # Compare outputs (only in debug mode)
    if args.mode == "debug":
        print("=" * 30)
        print("OUTPUT COMPARISON")
        print("=" * 30)
        
        identical = compare_outputs(output_unsharded, output_sharded, "Unsharded", "Sharded")
        print()
    else:
        identical = True  # No comparison needed for non-debug modes
    
    # Performance comparison (only in timing mode)
    if args.mode == "timing":
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
    
    # Memory comparison
    if args.mode == "debug":
        print("=" * 30)
        print("MEMORY USAGE COMPARISON")
        print("=" * 30)
        
        # Calculate memory differences
        unsharded_model_mem = memory_snapshots['unsharded_model']['used_mb'] - memory_snapshots['baseline']['used_mb']
        sharded_model_mem = memory_snapshots['sharded_model']['used_mb'] - memory_snapshots['baseline']['used_mb']
        unsharded_peak_mem = memory_snapshots['unsharded_forward']['peak_mb']
        sharded_peak_mem = memory_snapshots['sharded_forward']['peak_mb']
        
        print(f"{'Stage':<20} {'Unsharded':<12} {'Sharded':<12} {'Difference':<15}")
        print("-" * 60)
        print(f"{'Model params:':<20} {unsharded_model_mem:<12.1f} {sharded_model_mem:<12.1f} {sharded_model_mem - unsharded_model_mem:<+15.1f}")
        print(f"{'Peak forward:':<20} {unsharded_peak_mem:<12.1f} {sharded_peak_mem:<12.1f} {sharded_peak_mem - unsharded_peak_mem:<+15.1f}")
        print()
    else:
        # For non-debug modes, just show sharded memory usage
        print("=" * 30)
        print("SHARDED MEMORY USAGE")
        print("=" * 30)
        
        sharded_model_mem = memory_snapshots['sharded_model']['used_mb'] - memory_snapshots['baseline']['used_mb']
        sharded_peak_mem = memory_snapshots['sharded_forward']['peak_mb']
        
        print(f"Model params memory: {sharded_model_mem:.1f}MB")
        print(f"Peak forward memory: {sharded_peak_mem:.1f}MB")
        print()
    
    # Final summary
    print("=" * 30)
    print("FINAL SUMMARY")
    print("=" * 30)
    
    if args.mode == "debug":
        print(f"Outputs identical: {'YES' if identical else 'NO'}")
    if args.mode == "timing":
        forward_time_sharded_ms = forward_time_sharded * 1_000
        print(f"Sharded time: {forward_time_sharded_ms:.4f}ms")
    print()
    
    print("FeedForward inference test completed!")


if __name__ == "__main__":
    main()