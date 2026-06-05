"""Minimal pi05 inference verification — no server, no robot.

Runs entirely under jax.disable_jit() so the debugger can step through
actual Python execution line by line.
"""
import logging
import time

import jax
import numpy as np

from openpi.policies import droid_policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

logging.basicConfig(level=logging.INFO)


def main():
    # Hardware check
    devices = jax.devices()
    print(f"JAX devices: {devices}")
    assert len(devices) > 0 and "cuda" in str(devices[0]).lower(), f"Expected CUDA device, got: {devices}"

    # Load pi05_droid from local cache
    config = _config.get_config("pi05_droid")
    checkpoint_dir = "gs://openpi-assets/checkpoints/pi05_droid"
    print(f"Loading config: pi05_droid (action_dim={config.model.action_dim}, "
          f"action_horizon={config.model.action_horizon})")

    t0 = time.time()
    with jax.disable_jit():
        policy = _policy_config.create_trained_policy(config, checkpoint_dir)
        print(f"Model loaded in {time.time() - t0:.1f}s")

        example = droid_policy.make_droid_example()
        t0 = time.time()
        result = policy.infer(example)
        infer_time = (time.time() - t0) * 1000
    print(f"Inference (eager, JIT disabled): {infer_time:.1f}ms")

    # Validate output
    expected_shape = (config.model.action_horizon, 8)  # DROID: 15×8
    assert result["actions"].shape == expected_shape, f"Expected {expected_shape}, got {result['actions'].shape}"
    print(f"Actions shape: {result['actions'].shape} pass")
    print(f"Action value range: [{result['actions'].min():.3f}, {result['actions'].max():.3f}]")

    print()
    print("=== Inference verification PASSED ===")
    print(f"Model: pi05 (pi05_droid)")
    print(f"Inference latency: {infer_time:.1f}ms")
    print("GPU memory: check with nvidia-smi")


if __name__ == "__main__":
    main()
