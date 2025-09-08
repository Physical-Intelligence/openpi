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

# Import shared utilities
from inference_test_utils import (
    setup_jax_environment, create_base_argument_parser, create_mesh,
    setup_devices_for_sharding, test_device_computation,
    run_forward_pass_with_profiling, run_forward_pass, compare_outputs,
    print_configuration, print_performance_comparison, print_final_summary,
    print_param_shapes
)

# Set up argument parser with feedforward-specific arguments
parser = create_base_argument_parser("Test FeedForward block inference with sharding")
parser.add_argument("--hidden-dim", type=int, default=512,
                   help="Hidden dimension (default: 512)")
parser.add_argument("--sharding-strategy", type=str, choices=["default", "megatron"], default="default",
                   help="Sharding strategy: 'default' for FSDP-style sharding, 'megatron' for tensor parallel (default: default)")

# Parse args early to get device count
args, _ = parser.parse_known_args()

# Set up JAX environment BEFORE importing JAX
setup_jax_environment(args.num_shards, platform=args.platform)

# Now import JAX and other dependencies
import jax
import jax.numpy as jnp
from jax.experimental import pjit
from jax.sharding import PartitionSpec as P
from openpi.models import lora
from openpi.training import sharding




def create_dummy_feedforward_input(batch_size=1, seq_len=10, model_dim=512):
    """Create dummy input for FeedForward block."""
    return jnp.ones((batch_size, seq_len, model_dim), dtype=jnp.float32)




# Note: megatron_sharding function removed since we now use the shared
# sharding.megatron_tensor_parallel_sharding function from inference_test_utils
    

def create_feedforward_block(model_dim=512, hidden_dim=2048, input_data=None, sharded=False, mesh=None, sharding_strategy="default", use_jit=True):
    """Create a single FeedForward block from Gemma."""
    print(f"Creating FeedForward block: model_dim={model_dim}, hidden_dim={hidden_dim}, sharded={sharded}, strategy={sharding_strategy}")
    
    # Create FeedForward block without LoRA
    # Set up sharding constraint functions based on strategy
    if sharded and sharding_strategy == "megatron":
        # Megatron: use tensor parallel MLP constraints
        input_constraint = sharding.megatron_input_constraint
        output_constraint = sharding.megatron_output_constraint
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
            sharded_params_spec = ff_block.megatron_tensor_parallel_sharding_info()
            param_sharding = sharding.megatron_tensor_parallel_sharding(
                params_shape, 
                mesh, 
                sharded_params=sharded_params_spec,
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
                # Show parameter shapes and sharding info
                num_shards = mesh.shape['fsdp']
                print(f"    [{sharding_strategy.upper()}] Parameter shard shapes (across {num_shards} shards):")
                print_param_shapes(params['params'], sharded=True)
                
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
                print(f"    [UNSHARDED] Input shape: {x.shape}, sharding: {x.sharding if hasattr(x, 'sharding') else 'None'}")
                if hasattr(x, 'sharding') and x.sharding is not None:
                    x_shard = tuple(x.sharding.shard_shape(x.shape))
                    print(f"    [UNSHARDED] Input x local shard: {x_shard}")
                
                # Show parameter shapes
                print(f"    [UNSHARDED] Parameter shapes:")
                print_param_shapes(params['params'], sharded=False)
                
                print(f"    [UNSHARDED] Forward pass:")
                
                # Forward pass
                output = ff_block.apply(params, x)
                
                print(f"    [UNSHARDED] Output shape: {output.shape}")
                
                return output
            else:
                # Use JIT or non-JIT version based on flag
                if use_jit:
                    return forward_fn_jit(params, x)
                else:
                    return forward_fn_no_jit(params, x)
    
    # Note: Setup time calculation would need to be added at the beginning of the function
    
    return ff_block, params, forward_fn





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
    
    # Print configuration using shared utility
    extra_config = {
        "Hidden dim": hidden_dim,
    }
    print_configuration(args, extra_config)
    
    # Set up devices for sharding
    device_count = setup_devices_for_sharding()
    print()
    
    
    # Create dummy input
    input_data = create_dummy_feedforward_input(batch_size, seq_len, model_dim)
    print(f"Input shape: {input_data.shape}")
    
    # Test device computation
    test_device_computation()
    
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
        print()
        
        print("DEBUG RUN (showing shapes):")
        output_unsharded, forward_time_unsharded = run_forward_pass(forward_fn_unsharded, params_unsharded, input_data, show_shapes=True, with_printing=True, num_warmup=0, num_timing_runs=1)
        
        print()
    else:
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
        print_performance_comparison(forward_time_unsharded, forward_time_sharded)
    
    
    # Final summary
    print_final_summary(args, identical, forward_time_sharded)
    
    print("FeedForward inference test completed!")


if __name__ == "__main__":
    main()