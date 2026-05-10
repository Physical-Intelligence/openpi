from openpi_client import action_chunk_broker
import pytest

from openpi.models import pi0_config
from openpi.policies import aloha_policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


def test_create_trained_policy_uses_configured_pytorch_precision(monkeypatch, tmp_path):
    precision_calls = []
    load_calls = []

    class DummyPaliGemmaWithExpert:
        def to_bfloat16_for_selected_params(self, precision):
            precision_calls.append(precision)

    class DummyPytorchModel:
        def __init__(self):
            self.paligemma_with_expert = DummyPaliGemmaWithExpert()
            self.device = None
            self.eval_called = False

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.eval_called = True

        def sample_actions(self, *args, **kwargs):
            raise AssertionError("sample_actions should not be called when constructing the policy")

    dummy_model = DummyPytorchModel()

    def fake_load_pytorch(self, train_config, weight_path):
        load_calls.append((self, train_config, weight_path))
        return dummy_model

    monkeypatch.setattr(pi0_config.Pi0Config, "load_pytorch", fake_load_pytorch)

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.safetensors").touch()
    train_config = _config.TrainConfig(
        name="test_config",
        exp_name="test_run",
        model=pi0_config.Pi0Config(),
        pytorch_training_precision="float32",
    )

    _policy_config.create_trained_policy(train_config, checkpoint_dir, norm_stats={}, pytorch_device="cpu")

    assert load_calls == [(train_config.model, train_config, str(checkpoint_dir / "model.safetensors"))]
    assert precision_calls == ["float32"]
    assert dummy_model.device == "cpu"
    assert dummy_model.eval_called


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
