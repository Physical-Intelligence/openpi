from openpi_client import action_chunk_broker
import pytest

from openpi.models import exported as _exported
from openpi.policies import aloha_policy
from openpi.policies import policy_config as _policy_config


def create_policy_config() -> _policy_config.PolicyConfig:
    model = _exported.PiModel.from_checkpoint("s3://openpi-assets/exported/pi0_base/model")

    return _policy_config.PolicyConfig(
        model=model,
        norm_stats=model.norm_stats("trossen_biarm_single_base_cam_24dim"),
        input_layers=[aloha_policy.AlohaInputs(action_dim=model.action_dim)],
        output_layers=[aloha_policy.AlohaOutputs()],
    )


@pytest.mark.manual
def test_infer():
    config = create_policy_config()
    policy = _policy_config.create_policy(config)

    example = aloha_policy.make_aloha_example()
    outputs = policy.infer(example)

    assert outputs["actions"].shape == (config.model.action_horizon, 14)


@pytest.mark.manual
def test_broker():
    config = create_policy_config()
    policy = _policy_config.create_policy(config)

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = aloha_policy.make_aloha_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["actions"].shape == (14,)
