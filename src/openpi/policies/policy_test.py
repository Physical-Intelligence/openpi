import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import action_chunk_broker
import pytest

from openpi.models import model as _model
from openpi.policies import aloha_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


def test_batch_and_convert_matches_the_eager_path():
    # The fused path must produce the same Observation as one dispatch per leaf followed by from_dict's uint8 branch.
    inputs = {
        "image": {"left_wrist_0_rgb": np.random.default_rng(0).integers(0, 256, (8, 8, 3), dtype=np.uint8)},
        "image_mask": {"left_wrist_0_rgb": np.True_},
        "state": np.zeros(6, np.float64),
        "tokenized_prompt": np.zeros(8, np.int64),
        "tokenized_prompt_mask": np.ones(8, np.bool_),
    }

    fused = _model.Observation.from_dict(_policy._batch_and_convert(inputs))  # noqa: SLF001
    eager = _model.Observation.from_dict(jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs))

    for key in inputs["image"]:
        assert fused.images[key].dtype == eager.images[key].dtype == jnp.float32
        assert fused.images[key].shape == eager.images[key].shape
        assert fused.image_masks[key].shape == eager.image_masks[key].shape
        np.testing.assert_allclose(np.asarray(fused.images[key]), np.asarray(eager.images[key]), atol=1e-6, rtol=0)
    np.testing.assert_array_equal(np.asarray(fused.state), np.asarray(eager.state))
    np.testing.assert_array_equal(np.asarray(fused.tokenized_prompt), np.asarray(eager.tokenized_prompt))


def test_batch_and_convert_only_rescales_image_leaves():
    inputs = {
        "image": {"cam": np.array([[[0, 128, 255]]], dtype=np.uint8)},
        "tokenized_prompt": np.array([0, 128, 255], dtype=np.uint8),
    }

    converted = _policy._batch_and_convert(inputs)  # noqa: SLF001

    assert converted["tokenized_prompt"].dtype == np.uint8
    np.testing.assert_array_equal(np.asarray(converted["tokenized_prompt"]), inputs["tokenized_prompt"][None])
    assert converted["image"]["cam"].dtype == jnp.float32
    np.testing.assert_allclose(
        np.asarray(converted["image"]["cam"]),
        inputs["image"]["cam"][None].astype(np.float32) / 255.0 * 2.0 - 1.0,
        atol=1e-6,
        rtol=0,
    )


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
