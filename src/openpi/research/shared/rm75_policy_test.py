"""Tests for rm75_policy module."""

import numpy as np

from openpi.models import model as _model
from openpi.research.shared.action_transforms import ACTION_DIM
from openpi.research.shared.rm75_policy import LeRobotRM75DataConfig
from openpi.research.shared.rm75_policy import RM75Inputs
from openpi.research.shared.rm75_policy import RM75Outputs
from openpi.research.shared.rm75_policy import make_rm75_example


def test_module_imports():
    """Verify the rm75_policy module can be imported."""
    from openpi.research.shared import rm75_policy  # noqa: F401


def test_make_rm75_example():
    """Verify make_rm75_example returns correct keys and shapes."""
    example = make_rm75_example()

    assert "observation/wrist_image" in example
    assert "observation/joint_position" in example
    assert "observation/joint_velocity" in example
    assert "observation/gripper_position" in example
    assert "observation/scene_image" in example
    assert "prompt" in example

    assert example["observation/wrist_image"].shape == (224, 224, 3)
    assert example["observation/wrist_image"].dtype == np.uint8
    assert example["observation/joint_position"].shape == (7,)
    assert example["observation/joint_velocity"].shape == (7,)
    assert example["observation/gripper_position"].shape == (1,)
    assert example["observation/scene_image"].shape == (224, 224, 3)
    assert example["observation/scene_image"].dtype == np.uint8
    assert example["prompt"] == "do something"


def test_rm75_inputs_pi0():
    """Test RM75Inputs with PI0 model type."""
    inputs_fn = RM75Inputs(model_type=_model.ModelType.PI0)
    example = make_rm75_example()
    result = inputs_fn(example)

    # Check required output keys.
    assert "state" in result
    assert "image" in result
    assert "image_mask" in result
    assert "prompt" in result

    # State: joint_position(7) + gripper_position(1) = 8D.
    assert result["state"].shape == (8,)

    # Image keys for PI0.
    assert set(result["image"].keys()) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}

    # All images should be (224, 224, 3).
    for key in result["image"]:
        assert result["image"][key].shape == (224, 224, 3), f"Image {key} has wrong shape"

    # Wrist and base images should have actual data (non-zero); right_wrist is zero-padded.
    assert result["image"]["left_wrist_0_rgb"].any(), "Wrist image should not be all zeros"
    assert result["image"]["base_0_rgb"].any(), "Base image should not be all zeros"
    assert not result["image"]["right_wrist_0_rgb"].any(), "Right wrist image should be all zeros"

    # Masking: base_0 and left_wrist are real.
    assert result["image_mask"]["base_0_rgb"] == np.True_
    assert result["image_mask"]["left_wrist_0_rgb"] == np.True_
    assert result["image_mask"]["right_wrist_0_rgb"] == np.False_


def test_rm75_inputs_pi0_fast():
    """Test RM75Inputs with PI0_FAST model type — different keys and all masks True."""
    inputs_fn = RM75Inputs(model_type=_model.ModelType.PI0_FAST)
    example = make_rm75_example()
    result = inputs_fn(example)

    # Image keys for PI0_FAST.
    assert set(result["image"].keys()) == {"base_0_rgb", "base_1_rgb", "wrist_0_rgb"}

    # All images (224, 224, 3).
    for key in result["image"]:
        assert result["image"][key].shape == (224, 224, 3), f"Image {key} has wrong shape"

    # Wrist image is at wrist_0_rgb; base_0 has scene image, base_1 is zero-padded.
    assert result["image"]["wrist_0_rgb"].any(), "Wrist image should not be all zeros"
    assert result["image"]["base_0_rgb"].any(), "Base 0 image should not be all zeros"
    assert not result["image"]["base_1_rgb"].any(), "Base 1 image should be all zeros"

    # FAST models: all masks True (don't mask out padding).
    assert result["image_mask"]["base_0_rgb"] == np.True_
    assert result["image_mask"]["base_1_rgb"] == np.True_
    assert result["image_mask"]["wrist_0_rgb"] == np.True_

    # State shape now 8D.
    assert result["state"].shape == (8,)


def test_rm75_inputs_pi05():
    """Test PI05 behaves same as PI0 for image mapping."""
    inputs_fn = RM75Inputs(model_type=_model.ModelType.PI05)
    example = make_rm75_example()
    result = inputs_fn(example)

    # Same image key layout as PI0.
    assert set(result["image"].keys()) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}

    # Same masking as PI0.
    assert result["image_mask"]["base_0_rgb"] == np.True_
    assert result["image_mask"]["left_wrist_0_rgb"] == np.True_
    assert result["image_mask"]["right_wrist_0_rgb"] == np.False_


def test_rm75_inputs_with_actions():
    """Test that actions pass through when present in input data."""
    inputs_fn = RM75Inputs(model_type=_model.ModelType.PI0)
    example = make_rm75_example()
    fake_actions = np.random.rand(50, 8).astype(np.float32)
    example["actions"] = fake_actions
    result = inputs_fn(example)

    assert "actions" in result
    np.testing.assert_array_equal(result["actions"], fake_actions)


def test_rm75_inputs_chw_image():
    """Test that CHW float images are converted to HWC uint8."""
    inputs_fn = RM75Inputs(model_type=_model.ModelType.PI0)
    example = make_rm75_example()
    # Replace with a CHW float image.
    example["observation/wrist_image"] = np.random.rand(3, 224, 224).astype(np.float32)
    example["observation/scene_image"] = np.random.randint(256, size=(224, 224, 3), dtype=np.uint8)
    result = inputs_fn(example)

    # Should be converted to HWC uint8.
    wrist = result["image"]["left_wrist_0_rgb"]
    assert wrist.shape == (224, 224, 3)
    assert wrist.dtype == np.uint8


def test_rm75_outputs():
    """Test RM75Outputs slices actions to first 8 dims."""
    outputs_fn = RM75Outputs()
    # Actions wider than 8 dims — e.g. model outputs 32.
    data = {"actions": np.random.rand(1, 32).astype(np.float32)}
    result = outputs_fn(data)

    assert result["actions"].shape == (1, 8)
    np.testing.assert_array_equal(result["actions"], data["actions"][:, :8])


def test_rm75_outputs_exact_dim():
    """Test RM75Outputs passes through correctly when actions are exactly 8D."""
    outputs_fn = RM75Outputs()
    original_actions = np.random.rand(1, 8).astype(np.float32)
    data = {"actions": original_actions}
    result = outputs_fn(data)

    assert result["actions"].shape == (1, 8)
    np.testing.assert_array_equal(result["actions"], original_actions)


def test_action_dim_is_8():
    """Verify ACTION_DIM constant from action_transforms is 8."""
    assert ACTION_DIM == 8


def test_lerobot_rm75_data_config_instantiation():
    config = LeRobotRM75DataConfig(repo_id="dummy/repo")
    assert config.repo_id == "dummy/repo"
