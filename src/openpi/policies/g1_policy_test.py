import numpy as np
import pytest

from openpi.models import model as _model
from openpi.policies import g1_policy


@pytest.mark.parametrize("image_shape", [(224, 224, 3), (3, 224, 224)])
def test_g1_inputs_use_head_camera_and_mask_padding(image_shape):
    transform = g1_policy.G1Inputs(model_type=_model.ModelType.PI05)

    result = transform(
        {
            "head_image": np.zeros(image_shape, dtype=np.uint8),
            "state": np.zeros(29, dtype=np.float32),
            "actions": np.zeros((10, 21), dtype=np.float32),
            "prompt": "slice the fruit",
        }
    )

    assert result["image"]["base_0_rgb"].shape == (224, 224, 3)
    assert result["image_mask"] == {
        "base_0_rgb": np.True_,
        "left_wrist_0_rgb": np.False_,
        "right_wrist_0_rgb": np.False_,
    }
    assert result["state"].shape == (29,)
    assert result["actions"].shape == (10, 21)


def test_g1_inputs_use_aligned_depth_in_auxiliary_image_slot():
    transform = g1_policy.G1Inputs(
        model_type=_model.ModelType.PI05,
        state_dim=24,
        task_action_dim=14,
        use_depth_image=True,
    )

    depth = np.full((240, 320, 3), 73, dtype=np.uint8)
    result = transform(
        {
            "head_image": np.zeros((240, 320, 3), dtype=np.uint8),
            "depth_image": depth,
            "state": np.zeros(24, dtype=np.float32),
            "actions": np.zeros((10, 14), dtype=np.float32),
            "prompt": "grasp the Coke can, lift it, and present it",
        }
    )

    np.testing.assert_array_equal(result["image"]["left_wrist_0_rgb"], depth)
    assert result["image_mask"] == {
        "base_0_rgb": np.True_,
        "left_wrist_0_rgb": np.True_,
        "right_wrist_0_rgb": np.False_,
    }
    assert result["actions"].shape == (10, 14)


def test_g1_left_only_rgbd_contract_uses_17_state_and_7_actions():
    transform = g1_policy.G1Inputs(
        model_type=_model.ModelType.PI05,
        state_dim=17,
        task_action_dim=7,
        use_depth_image=True,
    )

    result = transform(
        {
            "head_image": np.zeros((240, 320, 3), dtype=np.uint8),
            "depth_image": np.zeros((240, 320, 3), dtype=np.uint8),
            "state": np.zeros(17, dtype=np.float32),
            "actions": np.zeros((10, 7), dtype=np.float32),
            "prompt": "grasp the Coke can, lift it, and present it",
        }
    )

    assert result["state"].shape == (17,)
    assert result["actions"].shape == (10, 7)
    assert g1_policy.G1Outputs(task_action_dim=7)(
        {"actions": np.zeros((10, 32), dtype=np.float32)}
    )["actions"].shape == (10, 7)


def test_rgbd_augmentation_shares_spatial_translation() -> None:
    feature = np.zeros((32, 48, 3), dtype=np.uint8)
    feature[10:14, 20:24] = 255
    data = {
        "image": {"base_0_rgb": feature.copy(), "left_wrist_0_rgb": feature.copy()},
        "image_mask": {"left_wrist_0_rgb": np.True_},
    }
    transform = g1_policy.G1RgbdAugment(
        max_translation_px=5,
        max_brightness_delta=0.0,
        max_contrast_delta=0.0,
        max_channel_gain_delta=0.0,
        max_depth_noise_std=0.0,
        occlusion_probability=0.0,
    )

    np.random.seed(7)
    result = transform(data)

    np.testing.assert_array_equal(result["image"]["base_0_rgb"], result["image"]["left_wrist_0_rgb"])
    assert result["image"]["base_0_rgb"].shape == feature.shape
    assert not np.array_equal(result["image"]["base_0_rgb"], feature)


def test_rgbd_augmentation_is_seeded_and_preserves_depth_channels() -> None:
    rgb = np.full((24, 32, 3), 120, dtype=np.uint8)
    depth = np.full((24, 32, 3), 80, dtype=np.uint8)
    transform = g1_policy.G1RgbdAugment(occlusion_probability=1.0)

    def apply_once() -> dict:
        return transform(
            {
                "image": {"base_0_rgb": rgb.copy(), "left_wrist_0_rgb": depth.copy()},
                "image_mask": {"left_wrist_0_rgb": np.True_},
            }
        )

    np.random.seed(19)
    first = apply_once()
    np.random.seed(19)
    second = apply_once()

    np.testing.assert_array_equal(first["image"]["base_0_rgb"], second["image"]["base_0_rgb"])
    np.testing.assert_array_equal(first["image"]["left_wrist_0_rgb"], second["image"]["left_wrist_0_rgb"])
    augmented_depth = first["image"]["left_wrist_0_rgb"]
    np.testing.assert_array_equal(augmented_depth[..., 0], augmented_depth[..., 1])
    np.testing.assert_array_equal(augmented_depth[..., 1], augmented_depth[..., 2])


def test_g1_rgbd_contract_requires_depth():
    transform = g1_policy.G1Inputs(
        model_type=_model.ModelType.PI05,
        state_dim=24,
        task_action_dim=14,
        use_depth_image=True,
    )

    with pytest.raises(ValueError, match="aligned G1 depth"):
        transform(
            {
                "head_image": np.zeros((240, 320, 3), dtype=np.uint8),
                "state": np.zeros(24, dtype=np.float32),
            }
        )


def test_g1_outputs_drop_pi_padding():
    outputs = g1_policy.G1Outputs(task_action_dim=21)

    result = outputs({"actions": np.zeros((10, 32), dtype=np.float32)})

    assert result["actions"].shape == (10, 21)


def test_g1_coke_contract_uses_24_state_and_21_actions():
    transform = g1_policy.G1Inputs(model_type=_model.ModelType.PI05, state_dim=24, task_action_dim=21)

    result = transform(
        {
            "head_image": np.zeros((240, 320, 3), dtype=np.uint8),
            "state": np.zeros(24, dtype=np.float32),
            "actions": np.zeros((10, 21), dtype=np.float32),
            "prompt": "pick up the Coke can and hold it upright",
        }
    )

    assert result["state"].shape == (24,)
    assert result["actions"].shape == (10, 21)
    assert g1_policy.G1Outputs(task_action_dim=21)({"actions": np.zeros((10, 32))})["actions"].shape == (10, 21)


def test_g1_contract_rejects_wrong_state_width():
    transform = g1_policy.G1Inputs(model_type=_model.ModelType.PI05, state_dim=24, task_action_dim=21)

    with pytest.raises(ValueError, match="Expected 24 G1 state"):
        transform(
            {
                "head_image": np.zeros((240, 320, 3), dtype=np.uint8),
                "state": np.zeros(29, dtype=np.float32),
            }
        )
