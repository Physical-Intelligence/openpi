"""Tests for FeedForward and Attention sharding constraint functionality."""

import pytest
import jax
import jax.numpy as jnp

from openpi.models import lora, gemma
from openpi.training import sharding


# Test configuration constants
MODEL_DIM = 512
HIDDEN_DIM = 2048
SEQ_LEN = 128
BATCH_SIZE = 2
NUM_HEADS = 8
HEAD_DIM = 64


def _create_test_data(batch_size=BATCH_SIZE, seq_len=SEQ_LEN, model_dim=MODEL_DIM):
    """Create test input data for feedforward."""
    key = jax.random.key(42)
    return jax.random.normal(key, (batch_size, seq_len, model_dim))


def _create_attention_test_data(batch_size=BATCH_SIZE, seq_len=SEQ_LEN, model_dim=MODEL_DIM, num_heads=NUM_HEADS, head_dim=HEAD_DIM):
    """Create test input data for attention."""
    key = jax.random.key(42)
    
    # Input activations - list with single expert (non-expert case)
    x = jax.random.normal(key, (batch_size, seq_len, model_dim))
    xs = [x]  # Single expert
    
    # Positions for RoPE
    positions = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
    
    # Attention mask - causal mask for autoregressive attention
    attn_mask = jnp.tril(jnp.ones((seq_len, seq_len)))[None, None, :, :]
    attn_mask = jnp.broadcast_to(attn_mask, (batch_size, 1, seq_len, seq_len))
    
    # No KV cache for testing
    kv_cache = None
    
    return xs, positions, attn_mask, kv_cache


def _create_multi_device_mesh():
    """Create mesh for multi-device testing, skip if insufficient devices."""
    available_devices = jax.device_count()
    if available_devices < 2:
        pytest.skip(f"Test requires at least 2 devices, but only {available_devices} available. "
                   f"Run with: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 to enable multi-GPU testing.")
    
    # Use up to 8 devices with reasonable mesh configuration
    num_devices = min(available_devices, 8)
    num_fsdp_devices = max(2, num_devices // 2)
    
    mesh = sharding.make_mesh(num_fsdp_devices=num_fsdp_devices)

    # Choose a batch size that works for both FSDP (P(("batch","fsdp"))) and
    # Megatron (P("batch", None, "fsdp")) activation sharding: make it a
    # multiple of (batch * fsdp) which equals num_devices. Use 2x for headroom.
    batch_size = 2 * num_devices

    return mesh, batch_size


@pytest.mark.skipif(jax.device_count() < 2, reason="Requires multiple devices")
def test_feedforward_multi_device_sharding():
    """Test actual feedforward multi-device sharding with both activation and parameter sharding.
    
    This test only runs when multiple devices are available.
    """
    mesh, batch_size = _create_multi_device_mesh()
    input_data = _create_test_data(batch_size=batch_size)
    key = jax.random.key(42)
    
    print(f"Using {jax.device_count()} devices: mesh shape = {mesh.shape}")
    print(f"Input data shape: {input_data.shape}")
    
    with sharding.set_mesh(mesh):
        # Test LoRA FeedForward with different sharding strategies
        test_configs = [
            ("LoRA-FSDP", lora.FeedForward(
                features=MODEL_DIM, hidden_dim=HIDDEN_DIM, lora_config=None,
                input_sharding_constraint=sharding.activation_sharding_constraint,
                output_sharding_constraint=sharding.activation_sharding_constraint
            )),
            ("LoRA-Megatron", lora.FeedForward(
                features=MODEL_DIM, hidden_dim=HIDDEN_DIM, lora_config=None,
                input_sharding_constraint=sharding.megatron_input_constraint,
                output_sharding_constraint=sharding.megatron_output_constraint
            )),
            ("Gemma-FSDP", gemma.FeedForward(
                features=MODEL_DIM, hidden_dim=HIDDEN_DIM,
                input_sharding_constraint=sharding.activation_sharding_constraint,
                output_sharding_constraint=sharding.activation_sharding_constraint
            )),
            ("Gemma-Megatron", gemma.FeedForward(
                features=MODEL_DIM, hidden_dim=HIDDEN_DIM,
                input_sharding_constraint=sharding.megatron_input_constraint,
                output_sharding_constraint=sharding.megatron_output_constraint
            )),
        ]
        
        for config_name, ff_module in test_configs:
            print(f"\nTesting {config_name}:")
            
            # Initialize and test forward pass
            params = ff_module.init(key, input_data)
            output = ff_module.apply(params, input_data)
            
            # Verify output is sharded and has correct shape
            expected_shape = (batch_size, SEQ_LEN, MODEL_DIM)
            assert output.shape == expected_shape, \
                f"{config_name} output shape mismatch: expected {expected_shape}, got {output.shape}"
            
            assert hasattr(output, 'sharding') and output.sharding is not None, \
                f"{config_name} output should be sharded"
            
            # Verify shard shapes are correct for different sharding strategies
            actual_shard_shape = output.sharding.shard_shape(output.shape)
            
            if "FSDP" in config_name:
                # FSDP sharding: P(("batch", "fsdp")) - shards across both batch and fsdp dimensions
                expected_shard_shape = (
                    batch_size // (mesh.shape['batch'] * mesh.shape['fsdp']),  # batch dimension sharded across both axes
                    SEQ_LEN,                                                    # seq_len not sharded
                    MODEL_DIM                                                   # model_dim not sharded for FSDP activation sharding
                )
            else:  # Megatron
                # Megatron sharding: P("batch", "fsdp", None) - batch sharded, seq sharded
                expected_shard_shape = (
                    batch_size // mesh.shape['batch'],  # batch dimension sharded
                    SEQ_LEN // mesh.shape['fsdp'],   # seq_len sharded by FSDP
                    MODEL_DIM      # model_dim not sharded
                )
            
            assert actual_shard_shape == expected_shard_shape, \
                f"{config_name} shard shape mismatch: expected {expected_shard_shape}, got {actual_shard_shape}"
            
            print(f"  {config_name} forward pass successful, shard shape verified: {actual_shard_shape}")
            
            # Test parameter sharding for this configuration
            if "FSDP" in config_name:
                param_specs = sharding.fsdp_sharding(params, mesh, log=False)
            else:  # Megatron
                sharded_params_spec = ff_module.megatron_tensor_parallel_sharding_info()
                param_specs = sharding.megatron_tensor_parallel_sharding(
                    params, mesh, sharded_params=sharded_params_spec, log=False
                )
            
            # Apply parameter sharding and test forward pass
            sharded_params = jax.device_put(params, param_specs)
            sharded_output = ff_module.apply(sharded_params, input_data)
            
            assert sharded_output.shape == expected_shape, \
                f"{config_name} sharded params output shape mismatch"
            
            # Verify parameter shard shapes
            gating_param = sharded_params['params']['gating_einsum']
            linear_param = sharded_params['params']['linear']
            
            assert hasattr(gating_param, 'sharding') and gating_param.sharding is not None, \
                f"{config_name} gating parameter should be sharded"
            assert hasattr(linear_param, 'sharding') and linear_param.sharding is not None, \
                f"{config_name} linear parameter should be sharded"
            
            if "Megatron" in config_name:
                # Verify Megatron parameter sharding patterns
                gating_shard_shape = gating_param.sharding.shard_shape(gating_param.shape)
                linear_shard_shape = linear_param.sharding.shard_shape(linear_param.shape)
                
                # Gating: (2, MODEL_DIM, HIDDEN_DIM) -> sharded on last dimension
                expected_gating_shard = (2, MODEL_DIM, HIDDEN_DIM // mesh.shape['fsdp'])
                # Linear: (HIDDEN_DIM, MODEL_DIM) -> sharded on first dimension
                expected_linear_shard = (HIDDEN_DIM // mesh.shape['fsdp'], MODEL_DIM)
                
                assert gating_shard_shape == expected_gating_shard, \
                    f"{config_name} gating param shard shape mismatch: expected {expected_gating_shard}, got {gating_shard_shape}"
                assert linear_shard_shape == expected_linear_shard, \
                    f"{config_name} linear param shard shape mismatch: expected {expected_linear_shard}, got {linear_shard_shape}"
                
                print(f"  {config_name} parameter sharding verified: gating={gating_shard_shape}, linear={linear_shard_shape}")
            else:
                print(f"  {config_name} parameter sharding successful")
    
    print(f"\nAll feedforward multi-device sharding tests passed!")


@pytest.mark.skipif(jax.device_count() < 2, reason="Requires multiple devices")
def test_attention_multi_device_sharding():
    """Test actual attention multi-device sharding with both activation and parameter sharding."""
    mesh, batch_size = _create_multi_device_mesh()
    xs, positions, attn_mask, kv_cache = _create_attention_test_data(batch_size=batch_size)
    key = jax.random.key(42)
    
    print(f"Using {jax.device_count()} devices: mesh shape = {mesh.shape}")
    print(f"Input data shape: {xs[0].shape}")
    
    # Create Gemma config for single expert
    config = gemma.Config(
        width=MODEL_DIM,
        depth=1,
        mlp_dim=MODEL_DIM * 4,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
    )
    
    with sharding.set_mesh(mesh):
        # Test Attention with different sharding strategies
        test_configs = [
            ("Attention-FSDP", gemma.Attention(
                configs=[config],
                input_sharding_constraint=sharding.activation_sharding_constraint,
                output_sharding_constraint=sharding.activation_sharding_constraint
            )),
            ("Attention-Megatron", gemma.Attention(
                configs=[config],
                input_sharding_constraint=sharding.megatron_input_constraint,
                output_sharding_constraint=sharding.megatron_output_constraint
            )),
        ]
        
        for config_name, attn_module in test_configs:
            print(f"\nTesting {config_name}:")
            
            # Initialize and test forward pass
            params = attn_module.init(key, xs, positions, attn_mask, kv_cache)
            output, _ = attn_module.apply(params, xs, positions, attn_mask, kv_cache)
            
            # Verify output is sharded and has correct shape
            expected_shape = (batch_size, SEQ_LEN, MODEL_DIM)
            assert output[0].shape == expected_shape, \
                f"{config_name} output shape mismatch: expected {expected_shape}, got {output[0].shape}"
            
            assert hasattr(output[0], 'sharding') and output[0].sharding is not None, \
                f"{config_name} output should be sharded"
            
            # Verify shard shapes are correct for different sharding strategies
            actual_shard_shape = output[0].sharding.shard_shape(output[0].shape)
            
            if "FSDP" in config_name:
                # FSDP sharding: P(("batch", "fsdp")) - shards across both batch and fsdp dimensions
                expected_shard_shape = (
                    batch_size // (mesh.shape['batch'] * mesh.shape['fsdp']),  # batch dimension sharded across both axes
                    SEQ_LEN,                                                    # seq_len not sharded
                    MODEL_DIM                                                   # model_dim not sharded for FSDP activation sharding
                )
            else:  # Megatron
                # Megatron sharding: P("batch", "fsdp", None) - batch sharded, seq sharded
                expected_shard_shape = (
                    batch_size // mesh.shape['batch'],  # batch dimension sharded
                    SEQ_LEN // mesh.shape['fsdp'],   # seq_len sharded by FSDP
                    MODEL_DIM      # model_dim not sharded
                )
            
            assert actual_shard_shape == expected_shard_shape, \
                f"{config_name} shard shape mismatch: expected {expected_shard_shape}, got {actual_shard_shape}"
            
            print(f"  {config_name} forward pass successful, shard shape verified: {actual_shard_shape}")
            
            # Test parameter sharding for this configuration
            if "FSDP" in config_name:
                param_specs = sharding.fsdp_sharding(params, mesh, log=False)
            else:  # Megatron
                sharded_params_spec = attn_module.megatron_tensor_parallel_sharding_info()
                param_specs = sharding.megatron_tensor_parallel_sharding(
                    params, mesh, sharded_params=sharded_params_spec, log=False
                )
            
            # Apply parameter sharding and test forward pass
            sharded_params = jax.device_put(params, param_specs)
            sharded_output, _ = attn_module.apply(sharded_params, xs, positions, attn_mask, kv_cache)
            
            assert sharded_output[0].shape == expected_shape, \
                f"{config_name} sharded params output shape mismatch"
            
            # Verify parameter shard shapes for attention-specific parameters
            qkv_param = sharded_params['params']['qkv_einsum']['w']
            attn_vec_param = sharded_params['params']['attn_vec_einsum']['w']
            
            assert hasattr(qkv_param, 'sharding') and qkv_param.sharding is not None, \
                f"{config_name} qkv parameter should be sharded"
            assert hasattr(attn_vec_param, 'sharding') and attn_vec_param.sharding is not None, \
                f"{config_name} attn_vec parameter should be sharded"
            
            if "Megatron" in config_name:
                # Verify Megatron parameter sharding patterns
                qkv_shard_shape = qkv_param.sharding.shard_shape(qkv_param.shape)
                attn_vec_shard_shape = attn_vec_param.sharding.shard_shape(attn_vec_param.shape)
                
                # QKV: (3, NUM_HEADS, MODEL_DIM, HEAD_DIM) -> sharded on NUM_HEADS (index 1)
                expected_qkv_shard = (3, NUM_HEADS // mesh.shape['fsdp'], MODEL_DIM, HEAD_DIM)
                # Attn_vec: (NUM_HEADS, HEAD_DIM, MODEL_DIM) -> sharded on NUM_HEADS (index 0)
                expected_attn_vec_shard = (NUM_HEADS // mesh.shape['fsdp'], HEAD_DIM, MODEL_DIM)
                
                assert qkv_shard_shape == expected_qkv_shard, \
                    f"{config_name} qkv param shard shape mismatch: expected {expected_qkv_shard}, got {qkv_shard_shape}"
                assert attn_vec_shard_shape == expected_attn_vec_shard, \
                    f"{config_name} attn_vec param shard shape mismatch: expected {expected_attn_vec_shard}, got {attn_vec_shard_shape}"
                
                print(f"  {config_name} parameter sharding verified: qkv={qkv_shard_shape}, attn_vec={attn_vec_shard_shape}")
            else:
                print(f"  {config_name} parameter sharding successful")
    
    print(f"\nAll attention multi-device sharding tests passed!")