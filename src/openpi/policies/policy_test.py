import numpy as np
from openpi_client import action_chunk_broker
from openpi_client import base_policy
import pytest
import torch

from openpi.policies import aloha_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


class _FakeTorchModel:
    action_horizon = 3
    action_dim = 4

    def __init__(self):
        self.sample_calls = 0

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def sample_actions(self, device, observation, noise=None, **kwargs):
        del kwargs
        self.sample_calls += 1
        assert device == self.device
        assert observation.state.ndim == 2
        if noise is not None:
            return noise
        row_values = observation.state[:, :1].to(torch.float32)
        return row_values[:, None, :].expand(-1, self.action_horizon, self.action_dim).clone()


def _make_model_obs(value: float) -> dict:
    return {
        "image": {"base_0_rgb": np.zeros((2, 2, 3), dtype=np.float32)},
        "image_mask": {"base_0_rgb": np.True_},
        "state": np.asarray([value, 0.0], dtype=np.float32),
    }


class _CountingInputTransform:
    def __init__(self):
        self.calls = 0

    def __call__(self, data):
        self.calls += 1
        assert data["state"].ndim == 1
        return data


class _TrimOutputTransform:
    def __init__(self):
        self.calls = 0

    def __call__(self, data):
        self.calls += 1
        assert data["actions"].ndim == 2
        return {"actions": data["actions"][:, :2]}


def _make_policy(*, transforms=(), output_transforms=()):
    model = _FakeTorchModel()
    policy = _policy.Policy(
        model,
        transforms=transforms,
        output_transforms=output_transforms,
        is_pytorch=True,
        pytorch_device="cpu",
    )
    return policy, model


def test_infer_keeps_single_observation_shape():
    policy, model = _make_policy()

    result = policy.infer(_make_model_obs(5.0))

    assert result["actions"].shape == (model.action_horizon, model.action_dim)
    np.testing.assert_allclose(result["actions"], 5.0)
    assert model.sample_calls == 1


def test_infer_batch_keeps_batch_dimension_and_uses_one_model_call():
    policy, model = _make_policy()

    result = policy.infer_batch([_make_model_obs(5.0), _make_model_obs(7.0)])

    assert result["actions"].shape == (2, model.action_horizon, model.action_dim)
    np.testing.assert_allclose(result["actions"][0], 5.0)
    np.testing.assert_allclose(result["actions"][1], 7.0)
    assert result["policy_timing"]["batch_size"] == 2
    assert model.sample_calls == 1


def test_infer_batch_applies_transforms_per_sample():
    input_transform = _CountingInputTransform()
    output_transform = _TrimOutputTransform()
    policy, model = _make_policy(transforms=[input_transform], output_transforms=[output_transform])

    result = policy.infer_batch([_make_model_obs(1.0), _make_model_obs(2.0)])

    assert result["actions"].shape == (2, model.action_horizon, 2)
    assert input_transform.calls == 2
    assert output_transform.calls == 2
    assert model.sample_calls == 1


def test_infer_batch_validates_observations_and_noise():
    policy, model = _make_policy()
    obs_batch = [_make_model_obs(1.0), _make_model_obs(2.0)]

    noise = np.ones((2, model.action_horizon, model.action_dim), dtype=np.float32)
    result = policy.infer_batch(obs_batch, noise=noise)
    np.testing.assert_allclose(result["actions"], noise)

    singleton_noise = np.ones((model.action_horizon, model.action_dim), dtype=np.float32)
    result = policy.infer(_make_model_obs(1.0), noise=singleton_noise)
    np.testing.assert_allclose(result["actions"], singleton_noise)

    with pytest.raises(ValueError, match="obs_batch must contain at least one observation"):
        policy.infer_batch([])
    with pytest.raises(TypeError, match="obs_batch must be a non-empty sequence"):
        policy.infer_batch({"not": "a list"})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="obs_batch must contain only observation dictionaries"):
        policy.infer_batch([_make_model_obs(1.0), "bad"])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="Batched noise must have shape"):
        policy.infer_batch(obs_batch, noise=singleton_noise)
    with pytest.raises(ValueError, match="Noise batch size 1 does not match observation batch size 2"):
        policy.infer_batch(obs_batch, noise=noise[:1])


class _InferOnlyPolicy(base_policy.BasePolicy):
    def infer(self, obs):
        del obs
        return {"actions": np.zeros((2, 1), dtype=np.float32)}


def test_base_policy_and_action_chunk_broker_do_not_fallback_to_sequential_batching():
    with pytest.raises(NotImplementedError, match="does not support batched inference"):
        _InferOnlyPolicy().infer_batch([{}])

    broker = action_chunk_broker.ActionChunkBroker(_InferOnlyPolicy(), action_horizon=1)
    with pytest.raises(NotImplementedError, match="does not support batched inference"):
        broker.infer_batch([{}])


class _BatchPolicy(_InferOnlyPolicy):
    def __init__(self):
        self.obs_batch = None

    def infer_batch(self, obs_batch):
        self.obs_batch = obs_batch
        return {"actions": np.zeros((len(obs_batch), 2, 1), dtype=np.float32)}


def test_policy_recorder_forwards_infer_batch(tmp_path):
    inner = _BatchPolicy()
    recorder = _policy.PolicyRecorder(inner, str(tmp_path))

    result = recorder.infer_batch([{"x": np.asarray([1])}, {"x": np.asarray([2])}])

    assert result["actions"].shape == (2, 2, 1)
    assert inner.obs_batch is not None
    assert (tmp_path / "step_0.npy").exists()


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
