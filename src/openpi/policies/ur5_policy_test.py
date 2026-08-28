import numpy as np
import pytest

from openpi.models import model as _model
from openpi.policies import ur5_policy


def test_make_ur5_example():
    example = ur5_policy.make_ur5_example()
    assert example["joints"].shape == (6,)
    assert example["gripper"].shape == (1,)
    assert example["base_rgb"].shape == (224, 224, 3)
    assert example["wrist_rgb"].shape == (224, 224, 3)
    assert "prompt" in example


@pytest.mark.parametrize("model_type", [_model.ModelType.PI0, _model.ModelType.PI0_FAST])
def test_ur5_inputs_shapes(model_type):
    transform = ur5_policy.UR5Inputs(model_type=model_type)
    example = ur5_policy.make_ur5_example()
    out = transform(example)

    assert out["state"].shape == (7,)
    assert out["image"]["base_0_rgb"].shape == (224, 224, 3)
    assert out["image"]["left_wrist_0_rgb"].shape == (224, 224, 3)
    assert out["image"]["right_wrist_0_rgb"].shape == (224, 224, 3)


def test_ur5_inputs_image_mask_pi0():
    transform = ur5_policy.UR5Inputs(model_type=_model.ModelType.PI0)
    out = transform(ur5_policy.make_ur5_example())
    assert out["image_mask"]["base_0_rgb"] == np.True_
    assert out["image_mask"]["left_wrist_0_rgb"] == np.True_
    # Unused right-wrist slot should be masked out for pi0.
    assert out["image_mask"]["right_wrist_0_rgb"] == np.False_


def test_ur5_inputs_image_mask_pi0_fast():
    transform = ur5_policy.UR5Inputs(model_type=_model.ModelType.PI0_FAST)
    out = transform(ur5_policy.make_ur5_example())
    # pi0-FAST attends to all image slots.
    assert out["image_mask"]["right_wrist_0_rgb"] == np.True_


def test_ur5_inputs_right_wrist_is_zeros():
    transform = ur5_policy.UR5Inputs(model_type=_model.ModelType.PI0)
    out = transform(ur5_policy.make_ur5_example())
    assert np.all(out["image"]["right_wrist_0_rgb"] == 0)


def test_ur5_inputs_state_concatenation():
    transform = ur5_policy.UR5Inputs(model_type=_model.ModelType.PI0)
    joints = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
    gripper = np.array([0.7], dtype=np.float32)
    data = {
        "joints": joints,
        "gripper": gripper,
        "base_rgb": np.zeros((224, 224, 3), dtype=np.uint8),
        "wrist_rgb": np.zeros((224, 224, 3), dtype=np.uint8),
    }
    out = transform(data)
    np.testing.assert_array_equal(out["state"], np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]))


def test_ur5_inputs_image_conversion_from_float_chw():
    """Images stored as float32 (C, H, W) by LeRobot should be converted to uint8 (H, W, C)."""
    transform = ur5_policy.UR5Inputs(model_type=_model.ModelType.PI0)
    float_chw = np.random.rand(3, 224, 224).astype(np.float32)
    data = {
        "joints": np.zeros(6, dtype=np.float32),
        "gripper": np.zeros(1, dtype=np.float32),
        "base_rgb": float_chw,
        "wrist_rgb": float_chw,
    }
    out = transform(data)
    img = out["image"]["base_0_rgb"]
    assert img.dtype == np.uint8
    assert img.shape == (224, 224, 3)


def test_ur5_inputs_actions_passthrough():
    transform = ur5_policy.UR5Inputs(model_type=_model.ModelType.PI0)
    actions = np.random.rand(10, 7).astype(np.float32)
    data = {**ur5_policy.make_ur5_example(), "actions": actions}
    out = transform(data)
    np.testing.assert_array_equal(out["actions"], actions)


def test_ur5_inputs_no_actions_key_absent():
    transform = ur5_policy.UR5Inputs(model_type=_model.ModelType.PI0)
    data = ur5_policy.make_ur5_example()
    data.pop("actions", None)
    out = transform(data)
    assert "actions" not in out


def test_ur5_outputs_slices_seven_dims():
    transform = ur5_policy.UR5Outputs()
    # Simulate model output padded to a larger action_dim (e.g. 32).
    padded = np.random.rand(10, 32).astype(np.float32)
    out = transform({"actions": padded})
    assert out["actions"].shape == (10, 7)
    np.testing.assert_array_equal(out["actions"], padded[:, :7])
