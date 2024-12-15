from openpi_client import action_chunk_broker

from openpi.models import exported as _exported
from openpi.policies import aloha_policy


def test_infer():
    model = aloha_policy.load_pi0_model()

    policy = aloha_policy.create_aloha_policy(
        model,
        aloha_policy.PolicyConfig(norm_stats=aloha_policy.make_aloha_norm_stats()),
    )

    example = aloha_policy.make_aloha_example()
    outputs = policy.infer(example)

    assert outputs["qpos"].shape == (model.action_horizon, 14)


def test_exported_aloha_sim():
    ckpt_path = "checkpoints/pi0_sim/model"
    model = _exported.PiModel.from_checkpoint(ckpt_path)

    policy = aloha_policy.create_aloha_policy(
        model,
        aloha_policy.PolicyConfig(
            norm_stats=_exported.import_norm_stats(ckpt_path, "huggingface_aloha_sim_transfer_cube"),
            adapt_to_pi=False,
        ),
    )

    example = aloha_policy.make_aloha_example()
    outputs = policy.infer(example)

    assert outputs["qpos"].shape == (model.action_horizon, 14)


def test_broker():
    model = aloha_policy.load_pi0_model()

    policy = action_chunk_broker.ActionChunkBroker(
        aloha_policy.create_aloha_policy(
            model,
            aloha_policy.PolicyConfig(norm_stats=aloha_policy.make_aloha_norm_stats()),
        ),
        # Only execute the first half of the chunk.
        action_horizon=model.action_horizon // 2,
    )

    example = aloha_policy.make_aloha_example()
    for _ in range(model.action_horizon):
        outputs = policy.infer(example)
        assert outputs["qpos"].shape == (14,)
