from openpi_client import action_chunk_broker
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import time

from openpi import transforms as _transforms
from openpi.policies import aloha_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


def test_jax_async_dispatch_under_reports_without_sync():
    """Document #983 bug class: CPU timers stop before jitted work completes."""

    @jax.jit
    def slow_matmul():
        x = jnp.ones((1024, 1024))
        return x @ x.T

    start = time.monotonic()
    result = slow_matmul()
    cpu_only_ms = (time.monotonic() - start) * 1000

    start = time.monotonic()
    result = slow_matmul()
    jax.block_until_ready(result)
    synced_ms = (time.monotonic() - start) * 1000

    assert synced_ms >= cpu_only_ms


def test_jax_policy_infer_syncs_actions_before_timing(monkeypatch):
    calls: list[object] = []
    real_block_until_ready = jax.block_until_ready

    def spy_block_until_ready(value):
        calls.append(value)
        return real_block_until_ready(value)

    monkeypatch.setattr(jax, "block_until_ready", spy_block_until_ready)

    policy = object.__new__(_policy.Policy)
    policy._is_pytorch_model = False
    policy._input_transform = _transforms.compose([])
    policy._output_transform = _transforms.compose([])
    policy._sample_kwargs = {}
    policy._rng = jax.random.key(0)

    @jax.jit
    def fake_sample(_rng, _observation, **_kwargs):
        return jnp.ones((1, 4, 14))

    policy._sample_actions = fake_sample

    obs = {
        "state": np.ones((14,), dtype=np.float32),
        "image": {"cam_high": np.zeros((224, 224, 3), dtype=np.uint8)},
        "image_mask": {"cam_high": np.array(True)},
    }
    result = policy.infer(obs)

    assert calls, "expected jax.block_until_ready to run on the JAX inference path"
    assert "policy_timing" in result
    assert result["policy_timing"]["infer_ms"] >= 0


@pytest.mark.manual
def test_infer():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    example = aloha_policy.make_aloha_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 14)


@pytest.mark.manual
def test_broker():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = aloha_policy.make_aloha_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["actions"].shape == (14,)
