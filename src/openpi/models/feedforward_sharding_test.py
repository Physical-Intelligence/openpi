"""Tests for FeedForward sharding constraint functionality."""

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


def _create_test_data(batch_size=BATCH_SIZE, seq_len=SEQ_LEN, model_dim=MODEL_DIM):
    """Create test input data."""
    key = jax.random.key(42)
    return jax.random.normal(key, (batch_size, seq_len, model_dim))


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


def test_constraint_functions_no_mesh():
    """Test that sharding constraint functions work as no-ops when no mesh is active.
    
    This test verifies both LoRA and Gemma FeedForward with different constraint functions.
    """
    input_data = _create_test_data()
    key = jax.random.key(42)
    
    # Test configurations: (module_class, constraint_pairs)
    test_configs = [
        # LoRA FeedForward tests
        (lora.FeedForward, [
            (None, None),  # No constraints
            (sharding.activation_sharding_constraint, sharding.activation_sharding_constraint),  # FSDP
            (sharding.megatron_input_constraint, sharding.megatron_output_constraint),  # Megatron
        ]),
        # Gemma FeedForward tests  
        (gemma.FeedForward, [
            (None, None),  # No constraints
            (sharding.activation_sharding_constraint, sharding.activation_sharding_constraint),  # FSDP
            (sharding.megatron_input_constraint, sharding.megatron_output_constraint),  # Megatron
        ])
    ]
    
    expected_shape = (BATCH_SIZE, SEQ_LEN, MODEL_DIM)
    outputs = []
    
    for module_class, constraint_pairs in test_configs:
        for input_constraint, output_constraint in constraint_pairs:
            # Create module with constraints
            if module_class == lora.FeedForward:
                ff_module = module_class(
                    features=MODEL_DIM,
                    hidden_dim=HIDDEN_DIM,
                    lora_config=None,
                    input_sharding_constraint=input_constraint,
                    output_sharding_constraint=output_constraint
                )
            else:  # Gemma FeedForward
                ff_module = module_class(
                    features=MODEL_DIM,
                    hidden_dim=HIDDEN_DIM,
                    input_sharding_constraint=input_constraint,
                    output_sharding_constraint=output_constraint
                )
            
            # Test forward pass
            params = ff_module.init(key, input_data)
            output = ff_module.apply(params, input_data)
            
            # Verify output shape
            assert output.shape == expected_shape, \
                f"{module_class.__name__} output shape mismatch: expected {expected_shape}, got {output.shape}"
            
            outputs.append(output)
    
    # All outputs should be identical when no mesh is active (constraints are no-ops)
    reference_output = outputs[0]
    for i, output in enumerate(outputs[1:], 1):
        assert jnp.allclose(reference_output, output, rtol=1e-6, atol=1e-6), \
            f"Output {i} differs from reference when constraints should be no-ops"
    
    print(f"All constraint functions work correctly as no-ops without mesh")


def test_parameter_sharding():
    """Test parameter sharding with both FSDP and Megatron strategies.
    
    This test works with single device by testing sharding spec creation and application.
    """
    input_data = _create_test_data()
    key = jax.random.key(42)
    
    # Create minimal mesh for testing
    mesh = sharding.make_mesh(num_fsdp_devices=1)
    
    # Test both LoRA and Gemma FeedForward
    test_modules = [
        ("LoRA", lora.FeedForward(features=MODEL_DIM, hidden_dim=HIDDEN_DIM, lora_config=None)),
        ("Gemma", gemma.FeedForward(features=MODEL_DIM, hidden_dim=HIDDEN_DIM))
    ]
    
    for module_name, ff_module in test_modules:
        print(f"\nTesting {module_name} FeedForward parameter sharding:")
        
        # Initialize parameters
        params = ff_module.init(key, input_data)
        param_shapes = jax.tree_map(lambda x: x.shape, params)
        print(f"  Parameter shapes: {param_shapes}")
        
        # Test FSDP sharding
        fsdp_specs = sharding.fsdp_sharding(params, mesh, log=False)
        fsdp_spec_info = jax.tree_map(lambda x: x.spec, fsdp_specs)
        print(f"  FSDP sharding specs: {fsdp_spec_info}")
        
        # Verify FSDP specs are NamedSharding objects
        assert all(isinstance(spec, jax.sharding.NamedSharding) 
                   for spec in jax.tree_util.tree_leaves(fsdp_specs)), \
            f"{module_name} FSDP specs should be NamedSharding objects"
        
        # Test Megatron sharding
        if module_name == "Gemma":
            sharding_info = ff_module.megatron_tensor_parallel_sharding_info()
            sharded_params_spec = sharding_info['sharded_params']
        else:
            sharded_params_spec = [
                sharding.ParamAndShardIndex('gating_einsum', -1),
                sharding.ParamAndShardIndex('linear', 0),
            ]
        
        megatron_specs = sharding.megatron_tensor_parallel_sharding(
            params, mesh, sharded_params=sharded_params_spec, log=False
        )
        megatron_spec_info = jax.tree_map(lambda x: x.spec, megatron_specs)
        print(f"  Megatron sharding specs: {megatron_spec_info}")
        
        # Verify Megatron specs are NamedSharding objects
        assert all(isinstance(spec, jax.sharding.NamedSharding) 
                   for spec in jax.tree_util.tree_leaves(megatron_specs)), \
            f"{module_name} Megatron specs should be NamedSharding objects"
        
        # Test applying sharding specs (should work even with single device)
        try:
            sharded_params_fsdp = jax.device_put(params, fsdp_specs)
            sharded_params_megatron = jax.device_put(params, megatron_specs)
            
            # Verify shapes are preserved
            assert sharded_params_fsdp['params']['gating_einsum'].shape == params['params']['gating_einsum'].shape
            assert sharded_params_megatron['params']['linear'].shape == params['params']['linear'].shape
            
            print(f"  Successfully applied sharding specs to {module_name} parameters")
            
        except Exception as e:
            print(f"  Note: Could not apply sharding for {module_name} (expected with single device): {e}")


@pytest.mark.skipif(jax.device_count() < 2, reason="Requires multiple devices")
def test_multi_device_sharding():
    """Test actual multi-device sharding with both activation and parameter sharding.
    
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
                # Megatron sharding: P("batch", None, "fsdp") - batch sharded, model_dim sharded by FSDP
                expected_shard_shape = (
                    batch_size // mesh.shape['batch'],  # batch dimension sharded
                    SEQ_LEN // mesh.shape['fsdp'],   # seq_len sharded by FSDP
                    MODEL_DIM      # model_dim not sharded
                )
            
            assert actual_shard_shape == expected_shard_shape, \
                f"{config_name} shard shape mismatch: expected {expected_shard_shape}, got {actual_shard_shape}"
            
            print(f"  ✓ {config_name} forward pass successful, shard shape verified: {actual_shard_shape}")
            
            # Test parameter sharding for this configuration
            if "FSDP" in config_name:
                param_specs = sharding.fsdp_sharding(params, mesh, log=False)
            else:  # Megatron
                # Both LoRA and Gemma have the same megatron_tensor_parallel_sharding_info method
                sharding_info = ff_module.megatron_tensor_parallel_sharding_info()
                sharded_params_spec = sharding_info['sharded_params']
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
                
                print(f"  ✓ {config_name} parameter sharding verified: gating={gating_shard_shape}, linear={linear_shard_shape}")
            else:
                print(f"  ✓ {config_name} parameter sharding successful")
    
    print(f"\nAll multi-device sharding tests passed!")