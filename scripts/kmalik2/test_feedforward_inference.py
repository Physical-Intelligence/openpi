#!/usr/bin/env python3
"""
Test inference on a single FeedForward block from Gemma.
This script compares unsharded vs sharded FeedForward blocks to ensure identical outputs.

To test with 2 devices:
  CUDA_VISIBLE_DEVICES=0,1 uv run scripts/kmalik2/test_feedforward_inference.py
  # or for CPU with 2 devices:
  JAX_PLATFORMS=cpu uv run scripts/kmalik2/test_feedforward_inference.py
"""

import argparse
import os
import time
import jax
import jax.numpy as jnp
from openpi.models import lora
from openpi.training import sharding


def setup_jax_devices(num_devices):
    """Set up JAX to use the specified number of devices."""
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_FLAGS"] = f"--xla_cpu_enable_fast_math=false --xla_force_host_platform_device_count={num_devices}"
    jax.config.update("jax_platforms", "cpu")


def create_dummy_input(batch_size=1, seq_len=10, model_dim=512):
    """Create dummy input for FeedForward block."""
    return jnp.ones((batch_size, seq_len, model_dim), dtype=jnp.float32)


def create_mesh(batch_size=1, num_fsdp_devices=8):
    """Create a mesh for sharding with FSDP dimension only."""
    # Create a 1D mesh with FSDP dimension
    devices = jax.devices()[:num_fsdp_devices]
    return jax.sharding.Mesh(devices, ('fsdp',))


def create_feedforward_block(model_dim=512, hidden_dim=2048, input_data=None, sharded=False, mesh=None):
    """Create a single FeedForward block from Gemma."""
    print(f"Creating FeedForward block: model_dim={model_dim}, hidden_dim={hidden_dim}, sharded={sharded}")
    
    # Create FeedForward block without LoRA
    ff_block = lora.FeedForward(
        features=model_dim,
        hidden_dim=hidden_dim,
        lora_config=None
    )
    
    # Initialize the block
    params = ff_block.init(jax.random.key(0), input_data)
    
    if sharded and mesh is not None:
        # Apply FSDP sharding to parameters
        print("  Applying FSDP sharding to parameters...")
        param_sharding = sharding.fsdp_sharding(params, mesh, log=True)
        
        # Actually shard the parameters by putting them on devices with the sharding specs
        print("  Sharding parameters across devices...")
        sharded_params = {}
        for key, param in params.items():
            if hasattr(param, 'items'):  # nested dict
                sharded_params[key] = {}
                for subkey, subparam in param.items():
                    if hasattr(subparam, 'shape'):  # is an array
                        sharded_param = jax.device_put(subparam, param_sharding[key][subkey])
                        sharded_params[key][subkey] = sharded_param
                        print(f"    Sharded {key}.{subkey}: {subparam.shape} -> {sharded_param.shape}, sharding: {sharded_param.sharding}")
                    else:
                        sharded_params[key][subkey] = subparam
            elif hasattr(param, 'shape'):  # direct array
                sharded_param = jax.device_put(param, param_sharding[key])
                sharded_params[key] = sharded_param
                print(f"    Sharded {key}: {param.shape} -> {sharded_param.shape}, sharding: {sharded_param.sharding}")
            else:
                sharded_params[key] = param
        
        # Use FSDP-only activation sharding
        def activation_sharding_constraint(tensor):
            """Shard activations along FSDP dimension only"""
            return jax.lax.with_sharding_constraint(
                tensor, 
                jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, None, "fsdp"))
            )
        
        # Direct forward pass with sharding - no JIT, just print shapes
        def forward_fn(params, x, print_shapes=False):
            # Apply activation sharding constraint (both batch and FSDP dimensions)
            x = activation_sharding_constraint(x)
            
            # Get weights
            fc1 = params['params']['gating_einsum']
            fc2 = params['params']['linear']
            
            if print_shapes:
                # Show only local shard shapes for weights
                num_shards = fc1.sharding.mesh.shape['fsdp']
                print(f"    [SHARDED] Weight shard shapes (across {num_shards} shards):")
                fc1_shard_shape = fc1.sharding.shard_shape(fc1.shape)
                fc2_shard_shape = fc2.sharding.shard_shape(fc2.shape)
                print(f"    FC1 local shard: {tuple(fc1_shard_shape)}")
                print(f"    FC2 local shard: {tuple(fc2_shard_shape)}")
                
                # Forward pass - show only local shard shapes for activations
                print(f"    [SHARDED] Activation shard shapes:")
            
            ff_gate = jnp.dot(x, fc1[0])
            if print_shapes:
                ff_gate_shard = tuple(ff_gate.sharding.shard_shape(ff_gate.shape))
                print(f"    FF gate local shard: {ff_gate_shard}")
            
            gate_value = jax.nn.gelu(ff_gate)
            if print_shapes:
                gate_value_shard = tuple(gate_value.sharding.shard_shape(gate_value.shape))
                print(f"    Gate value local shard: {gate_value_shard}")
            
            ff1 = jnp.dot(x, fc1[1])
            if print_shapes:
                ff1_shard = tuple(ff1.sharding.shard_shape(ff1.shape))
                print(f"    FF1 local shard: {ff1_shard}")
            
            activations = gate_value * ff1
            if print_shapes:
                activations_shard = tuple(activations.sharding.shard_shape(activations.shape))
                print(f"    Activations local shard: {activations_shard}")
            
            outputs = jnp.dot(activations, fc2)
            if print_shapes:
                outputs_shard = tuple(outputs.sharding.shard_shape(outputs.shape))
                print(f"    Output local shard: {outputs_shard}")
            
            output = activation_sharding_constraint(outputs)
            if print_shapes:
                output_shard = tuple(output.sharding.shard_shape(output.shape))
                print(f"    Final output local shard: {output_shard}")
            
            return output
        
        # Use the sharded parameters
        params = sharded_params
    else:
        # Direct forward pass without sharding - no JIT, just print shapes
        def forward_fn(params, x, print_shapes=False):
            # Get weights
            fc1 = params['params']['gating_einsum']
            fc2 = params['params']['linear']
            
            if print_shapes:
                print(f"    [UNSHARDED] Input shape: {x.shape}, sharding: {x.sharding if hasattr(x, 'sharding') else 'None'}")
                print(f"    [UNSHARDED] FC1 weights: {fc1.shape}, sharding: {fc1.sharding if hasattr(fc1, 'sharding') else 'None'}")
                print(f"    [UNSHARDED] FC2 weights: {fc2.shape}, sharding: {fc2.sharding if hasattr(fc2, 'sharding') else 'None'}")
                print(f"    [UNSHARDED] Forward pass:")
            
            # Forward pass
            ff_gate = jnp.dot(x, fc1[0])
            gate_value = jax.nn.gelu(ff_gate)
            ff1 = jnp.dot(x, fc1[1])
            activations = gate_value * ff1
            outputs = jnp.dot(activations, fc2)
            
            if print_shapes:
                print(f"    [UNSHARDED] All activations on single device: {outputs.shape}")
            
            return outputs
    
    print(f"  Setup time: {time.time() - time.time():.3f}s")
    
    return ff_block, params, forward_fn


def run_forward_pass(forward_fn, params, input_data, show_shapes=False, with_printing=True):
    """Run forward pass and measure time"""
    if with_printing:
        print("Running forward pass...")
        
        if show_shapes:
            print("  Showing tensor shapes during forward pass:")
        
        # Timed run
        print("  Timed run...")
    else:
        print("Running forward pass (no printing)...")
    
    start = time.time()
    output = forward_fn(params, input_data, print_shapes=show_shapes)
    output.block_until_ready()
    forward_time = time.time() - start
    print(f"  Forward pass time: {forward_time:.3f}s")
    
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
    print(f"  Available devices: {jax.device_count()}")
    print(f"  Device types: {[str(d) for d in jax.devices()]}")
    
    if jax.device_count() < 2:
        print("  WARNING: Only 1 device available. Sharding will not work properly.")
        return 1
    else:
        print(f"  Using {jax.device_count()} devices for sharding.")
        return jax.device_count()


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test FeedForward block inference with sharding")
    parser.add_argument("--num-shards", type=int, default=8, 
                       help="Number of shards/devices to use (default: 2)")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size (default: 1)")
    parser.add_argument("--seq-len", type=int, default=128,
                       help="Sequence length (default: 128)")
    parser.add_argument("--model-dim", type=int, default=512,
                       help="Model dimension (default: 512)")
    parser.add_argument("--hidden-dim", type=int, default=2048,
                       help="Hidden dimension (default: 2048)")
    
    args = parser.parse_args()
    
    # Set up JAX with specified number of devices
    setup_jax_devices(args.num_shards)
    
    print("FeedForward Block Inference Test (Unsharded vs Sharded)")
    print("=" * 60)
    
    # Configuration
    batch_size = args.batch_size
    seq_len = args.seq_len
    model_dim = args.model_dim
    hidden_dim = args.hidden_dim
    
    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Model dim: {model_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Number of shards: {args.num_shards}")
    print()
    
    # Set up devices for sharding
    device_count = setup_devices_for_sharding()
    print()
    
    # Create dummy input
    input_data = create_dummy_input(batch_size, seq_len, model_dim)
    print(f"Input shape: {input_data.shape}")
    print()
    
    # Create mesh for sharding
    mesh = create_mesh(batch_size, args.num_shards)
    print(f"Mesh shape: {mesh.shape}")
    print()
    
    # Test 1: Unsharded version
    print("=" * 30)
    print("TEST 1: UNSHARDED VERSION")
    print("=" * 30)
    
    ff_block_unsharded, params_unsharded, forward_fn_unsharded = create_feedforward_block(
        model_dim, hidden_dim, input_data, sharded=False
    )
    print()
    
    # Run unsharded with debug printing
    print("DEBUG RUN (showing shapes):")
    output_unsharded, forward_time_unsharded_debug = run_forward_pass(forward_fn_unsharded, params_unsharded, input_data, show_shapes=True, with_printing=True)
    print()
    
    # Run unsharded without debug printing (for timing)
    print("TIMING RUN (no debug output):")
    output_unsharded_timing, forward_time_unsharded = run_forward_pass(forward_fn_unsharded, params_unsharded, input_data, show_shapes=False, with_printing=False)
    print()
    
    # Test 2: Sharded version
    print("=" * 30)
    print("TEST 2: SHARDED VERSION")
    print("=" * 30)
    
    with sharding.set_mesh(mesh):
        ff_block_sharded, params_sharded, forward_fn_sharded = create_feedforward_block(
            model_dim, hidden_dim, input_data, sharded=True, mesh=mesh
        )
        print()
        
        # Run sharded with debug printing
        print("DEBUG RUN (showing shard info):")
        output_sharded, forward_time_sharded_debug = run_forward_pass(forward_fn_sharded, params_sharded, input_data, show_shapes=True, with_printing=True)
        print()
        
        # Run sharded without debug printing (for timing)
        print("TIMING RUN (no debug output):")
        output_sharded_timing, forward_time_sharded = run_forward_pass(forward_fn_sharded, params_sharded, input_data, show_shapes=False, with_printing=False)
        print()
    
    # Compare outputs
    print("=" * 30)
    print("OUTPUT COMPARISON")
    print("=" * 30)
    
    identical = compare_outputs(output_unsharded, output_sharded, "Unsharded", "Sharded")
    print()
    
    # Performance comparison
    print("=" * 30)
    print("PERFORMANCE COMPARISON")
    print("=" * 30)
    
    print(f"Unsharded forward time: {forward_time_unsharded:.3f}s")
    print(f"Sharded forward time: {forward_time_sharded:.3f}s")
    speedup = forward_time_unsharded / forward_time_sharded if forward_time_sharded > 0 else float('inf')
    print(f"Speedup: {speedup:.2f}x")
    print()
    
    # Final summary
    print("=" * 30)
    print("FINAL SUMMARY")
    print("=" * 30)
    
    print(f"Outputs identical: {'YES' if identical else 'NO'}")
    print(f"Unsharded time: {forward_time_unsharded:.3f}s")
    print(f"Sharded time: {forward_time_sharded:.3f}s")
    print()
    
    print("FeedForward inference test completed!")


if __name__ == "__main__":
    main()