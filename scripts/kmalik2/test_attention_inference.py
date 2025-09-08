#!/usr/bin/env python3
"""
Test inference on a single Attention block from Gemma.
This script compares unsharded vs sharded Attention blocks to ensure identical outputs.

To test with 2 devices:
  CUDA_VISIBLE_DEVICES=0,1 uv run scripts/kmalik2/test_attention_inference.py
  # or for CPU with 2 devices:
  JAX_PLATFORMS=cpu uv run scripts/kmalik2/test_attention_inference.py
"""

import jax
import jax.numpy as jnp
from jax.experimental import pjit
from openpi.models import gemma
from openpi.training import sharding

# Import shared utilities
from inference_test_utils import (
    setup_jax_environment, create_base_argument_parser, create_mesh,
    setup_devices_for_sharding, test_device_computation,
    run_forward_pass_with_profiling, run_forward_pass, compare_outputs,
    print_configuration, print_performance_comparison, print_final_summary,
    print_param_shapes
)




def create_dummy_attention_inputs(batch_size=1, seq_len=10, model_dim=512, num_heads=8, head_dim=64):
    """Create dummy inputs for Attention block."""
    # Input activations - list with single expert (non-expert case)
    x = jnp.ones((batch_size, seq_len, model_dim), dtype=jnp.float32)
    xs = [x]  # Single expert
    
    # Positions for RoPE
    positions = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
    
    # Attention mask - causal mask for autoregressive attention
    attn_mask = jnp.tril(jnp.ones((seq_len, seq_len)))[None, None, :, :]
    attn_mask = jnp.broadcast_to(attn_mask, (batch_size, 1, seq_len, seq_len))
    
    # No KV cache for now
    kv_cache = None
    
    return xs, positions, attn_mask, kv_cache


def create_attention_block(model_dim=512, num_heads=8, head_dim=64, input_data=None, sharded=False, mesh=None, sharding_strategy="default", use_jit=True):
    """Create a single Attention block from Gemma."""
    print(f"Creating Attention block: model_dim={model_dim}, num_heads={num_heads}, head_dim={head_dim}, sharded={sharded}, strategy={sharding_strategy}")
    
    # Create Gemma config for single expert
    config = gemma.Config(
        width=model_dim,
        depth=1,  # Not used for single attention block
        mlp_dim=model_dim * 4,  # Not used for attention
        num_heads=num_heads,
        num_kv_heads=num_heads,  # Use same number of kv heads as query heads
        head_dim=head_dim,
    )
    
    # Set up sharding constraint functions based on strategy
    if sharded and sharding_strategy == "megatron":
        # Megatron: use tensor parallel attention constraints
        input_constraint = sharding.megatron_attn_input_constraint
        output_constraint = sharding.megatron_attn_output_constraint
    elif sharded and sharding_strategy == "default":
        # Default FSDP: no constraints, handled by parameter sharding
        input_constraint = None
        output_constraint = None
    else:
        # Unsharded: no constraints
        input_constraint = None
        output_constraint = None

    # Create Attention block
    attn_block = gemma.Attention(
        configs=[config],
        input_sharding_constraint=input_constraint,
        output_sharding_constraint=output_constraint
    )
    
    # Unpack input data
    xs, positions, attn_mask, kv_cache = input_data
    
    if sharded and mesh is not None:
        # Apply sharding to parameters using jitted init with out_shardings
        print(f"  Applying {sharding_strategy} sharding to parameters...")
        
        # Shape evaluation - mesh context is already active from the calling context
        params_shape = jax.eval_shape(attn_block.init, jax.random.key(0), xs, positions, attn_mask, kv_cache)
        
        if sharding_strategy == "megatron":
            # Get sharding info from the Attention block
            tp_info = attn_block.megatron_tensor_parallel_sharding_info()
            param_sharding = sharding.megatron_tensor_parallel_sharding(
                params_shape, 
                mesh, 
                sharded_params=tp_info['sharded_params'],
                log=True
            )
        else:  # default
            param_sharding = sharding.fsdp_sharding(params_shape, mesh, log=True)
            
        print("  Initializing parameters directly into sharded buffers...")
        # Parameter initialization - mesh context should already be active when sharded=True
        params = jax.jit(attn_block.init, out_shardings=param_sharding)(jax.random.key(0), xs, positions, attn_mask, kv_cache)
        
        # Create forward function based on use_jit flag
        if use_jit:
            # PJIT-compiled forward pass
            def _forward_impl(params, xs, positions, attn_mask, kv_cache):
                return attn_block.apply(params, xs, positions, attn_mask, kv_cache)
            
            forward_fn_jit = pjit.pjit(
                _forward_impl,
                in_shardings=(param_sharding, None, None, None, None),
                out_shardings=None,
            )
        else:
            # Non-JIT version
            def forward_fn_no_jit(params, xs, positions, attn_mask, kv_cache):
                return attn_block.apply(params, xs, positions, attn_mask, kv_cache)
        
        # Wrapper function for debugging with shapes (not JIT-compiled)
        def forward_fn(params, input_data, print_shapes=False):
            xs, positions, attn_mask, kv_cache = input_data
            
            if print_shapes:
                # Show parameter shapes and sharding info
                num_shards = mesh.shape['fsdp']
                print(f"    [{sharding_strategy.upper()}] Parameter shard shapes (across {num_shards} shards):")
                print_param_shapes(params['params'], sharded=True)
                
                # Show input shapes and sharding
                print(f"    [{sharding_strategy.upper()}] Input shapes:")
                x = xs[0]  # First expert
                print(f"    Input x shape: {x.shape}, sharding: {x.sharding if hasattr(x, 'sharding') else 'None'}")
                print(f"    Positions shape: {positions.shape}")
                print(f"    Attention mask shape: {attn_mask.shape}")
                
                if hasattr(x, 'sharding') and x.sharding is not None:
                    x_shard = tuple(x.sharding.shard_shape(x.shape))
                    print(f"    [{sharding_strategy.upper()}] Input x local shard: {x_shard}")
                
                # Forward pass
                print(f"    [{sharding_strategy.upper()}] Running forward pass...")
                outputs, new_kv_cache = attn_block.apply(params, xs, positions, attn_mask, kv_cache)
                
                # Show output shapes
                output = outputs[0]  # First expert output
                if hasattr(output, 'sharding') and output.sharding is not None:
                    output_shard = tuple(output.sharding.shard_shape(output.shape))
                    print(f"    Final output local shard: {output_shard}")
                else:
                    print(f"    Final output shape: {output.shape}")
                
                return outputs, new_kv_cache
            else:
                # Use JIT or non-JIT version based on flag
                if use_jit:
                    return forward_fn_jit(params, xs, positions, attn_mask, kv_cache)
                else:
                    return forward_fn_no_jit(params, xs, positions, attn_mask, kv_cache)
    else:
        # Initialize the block without sharding
        params = attn_block.init(jax.random.key(0), xs, positions, attn_mask, kv_cache)
        
        # Create forward function based on use_jit flag
        if use_jit:
            # PJIT-compiled forward pass without sharding (for consistency)
            def _forward_impl(params, xs, positions, attn_mask, kv_cache):
                return attn_block.apply(params, xs, positions, attn_mask, kv_cache)
            forward_fn_jit = pjit.pjit(_forward_impl)
        else:
            # Non-JIT version
            def forward_fn_no_jit(params, xs, positions, attn_mask, kv_cache):
                return attn_block.apply(params, xs, positions, attn_mask, kv_cache)
        
        # Wrapper function for debugging with shapes (not JIT-compiled)
        def forward_fn(params, input_data, print_shapes=False):
            xs, positions, attn_mask, kv_cache = input_data
            
            if print_shapes:
                print(f"    [UNSHARDED] Input shapes:")
                x = xs[0]  # First expert
                print(f"    Input x shape: {x.shape}, sharding: {x.sharding if hasattr(x, 'sharding') else 'None'}")
                print(f"    Positions shape: {positions.shape}")
                print(f"    Attention mask shape: {attn_mask.shape}")
                
                # Show parameter shapes
                print(f"    [UNSHARDED] Parameter shapes:")
                print_param_shapes(params['params'], sharded=False)
                
                print(f"    [UNSHARDED] Forward pass:")
                
                # Forward pass
                outputs, new_kv_cache = attn_block.apply(params, xs, positions, attn_mask, kv_cache)
                
                output = outputs[0]  # First expert output
                print(f"    [UNSHARDED] Output shape: {output.shape}")
                
                return outputs, new_kv_cache
            else:
                # Use JIT or non-JIT version based on flag
                if use_jit:
                    return forward_fn_jit(params, xs, positions, attn_mask, kv_cache)
                else:
                    return forward_fn_no_jit(params, xs, positions, attn_mask, kv_cache)
    
    return attn_block, params, forward_fn





def main():
    """Main test function."""
    # Set up argument parser with attention-specific arguments
    parser = create_base_argument_parser("Test Attention block inference with sharding")
    parser.add_argument("--num-heads", type=int, default=None,
                       help="Number of attention heads (default: auto-calculated as model_dim // head_dim)")
    parser.add_argument("--head-dim", type=int, default=256,
                       help="Head dimension (default: 256)")
    parser.add_argument("--sharding-strategy", type=str, choices=["default", "megatron"], default="default",
                       help="Sharding strategy: 'default' for FSDP-style sharding, 'megatron' for tensor parallel (default: default)")
    
    # Parse args early to get device count
    args, _ = parser.parse_known_args()
    
    # Set up JAX environment BEFORE importing JAX
    setup_jax_environment(args.num_shards, platform=args.platform)
    
    print("Attention Block Inference Test (Unsharded vs Sharded)")
    print("=" * 60)
    
    # Configuration
    batch_size = args.batch_size
    seq_len = args.seq_len
    model_dim = args.model_dim
    head_dim = args.head_dim
    
    # Calculate num_heads if not provided
    if args.num_heads is None:
        if model_dim % head_dim != 0:
            raise ValueError(f"model_dim ({model_dim}) must be divisible by head_dim ({head_dim}) when num_heads is auto-calculated")
        num_heads = model_dim // head_dim
        print(f"Auto-calculated num_heads: {num_heads} (model_dim={model_dim} // head_dim={head_dim})")
    else:
        num_heads = args.num_heads
    
    # Assert that batch size should be divisible by num_shards
    assert batch_size % args.num_shards == 0, f"Batch size ({batch_size}) must be divisible by num_shards ({args.num_shards})"
    
    # Print configuration using shared utility
    extra_config = {
        "Number of heads": num_heads,
        "Head dim": head_dim,
    }
    print_configuration(args, extra_config)
    
    # Set up devices for sharding
    device_count = setup_devices_for_sharding()
    print()
    
    # Create dummy input
    input_data = create_dummy_attention_inputs(batch_size, seq_len, model_dim, num_heads, head_dim)
    xs, positions, attn_mask, kv_cache = input_data
    print(f"Input shapes:")
    print(f"  xs[0]: {xs[0].shape}")
    print(f"  positions: {positions.shape}")
    print(f"  attn_mask: {attn_mask.shape}")
    print(f"  kv_cache: {kv_cache}")
    
    # Test device computation
    test_device_computation()
    
    # Create mesh for sharding
    mesh = create_mesh(args.num_shards)
    print(f"Mesh shape: {mesh.shape}")
    print()
    
    # Output extractor for attention (returns first expert output)
    attention_output_extractor = lambda x: x[0][0]
    
    # Test 1: Unsharded version (only in debug mode)
    if args.mode == "debug":
        print("=" * 30)
        print("TEST 1: UNSHARDED VERSION")
        print("=" * 30)
        
        attn_block_unsharded, params_unsharded, forward_fn_unsharded = create_attention_block(
            model_dim, num_heads, head_dim, input_data, sharded=False, use_jit=not args.no_jit
        )
        
        print("DEBUG RUN (showing shapes):")
        output_unsharded, forward_time_unsharded = run_forward_pass(
            forward_fn_unsharded, params_unsharded, input_data, 
            show_shapes=True, with_printing=True, num_warmup=0, num_timing_runs=1,
            output_extractor=attention_output_extractor
        )
        
    else:
        # For non-debug modes, create dummy values to avoid errors later
        output_unsharded = None
        forward_time_unsharded = 0.0
    
    # Test 2: Sharded version
    print("=" * 30)
    print("TEST 2: SHARDED VERSION")
    print("=" * 30)
    
    with sharding.set_mesh(mesh):
        attn_block_sharded, params_sharded, forward_fn_sharded = create_attention_block(
            model_dim, num_heads, head_dim, input_data, sharded=True, mesh=mesh, sharding_strategy=args.sharding_strategy, use_jit=not args.no_jit
        )
        
        # Run sharded based on mode
        if args.mode == "debug":
            print("DEBUG RUN (showing shard info):")
            output_sharded, forward_time_sharded = run_forward_pass(
                forward_fn_sharded, params_sharded, input_data, 
                show_shapes=True, with_printing=True, num_warmup=0, num_timing_runs=1,
                output_extractor=attention_output_extractor
            )
        elif args.mode == "profile":
            print("PROFILE RUN (analyzing communication):")
            output_sharded, forward_time_sharded = run_forward_pass_with_profiling(
                forward_fn_sharded, params_sharded, input_data, "sharded", sharded=True, 
                mode=args.mode, num_warmup=3, num_profile_runs=3, trace_dir_base=args.trace_dir,
                output_extractor=attention_output_extractor
            )
        elif args.mode == "profile_both":
            print("PROFILE RUN - SHARDED (analyzing communication):")
            output_sharded, forward_time_sharded = run_forward_pass_with_profiling(
                forward_fn_sharded, params_sharded, input_data, "sharded", sharded=True, 
                mode="profile", num_warmup=3, num_profile_runs=3, trace_dir_base=args.trace_dir,
                output_extractor=attention_output_extractor
            )
        else:  # timing mode
            print("TIMING RUN (no debug output):")
            output_sharded, forward_time_sharded = run_forward_pass(
                forward_fn_sharded, params_sharded, input_data, 
                show_shapes=False, with_printing=False, num_warmup=3, num_timing_runs=args.timing_runs,
                output_extractor=attention_output_extractor
            )
    
    # Compare outputs (only in debug mode)
    if args.mode == "debug":
        print("=" * 30)
        print("OUTPUT COMPARISON")
        print("=" * 30)
        
        identical = compare_outputs(
            output_unsharded, output_sharded, "Unsharded", "Sharded",
            output_extractor=attention_output_extractor
        )
        print()
    else:
        identical = True  # No comparison needed for non-debug modes
    
    # Performance comparison (only in timing mode)
    if args.mode == "timing":
        print_performance_comparison(forward_time_unsharded, forward_time_sharded)
    
    # Final summary
    print_final_summary(args, identical, forward_time_sharded)
    
    print("Attention inference test completed!")


if __name__ == "__main__":
    main()
