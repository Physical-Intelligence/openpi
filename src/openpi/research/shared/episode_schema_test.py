"""Tests for episode_schema module."""

import dataclasses
import json

import numpy as np
import pytest

from openpi.research.shared.episode_schema import Action
from openpi.research.shared.episode_schema import Episode
from openpi.research.shared.episode_schema import EpisodeLabels
from openpi.research.shared.episode_schema import EpisodeMetadata
from openpi.research.shared.episode_schema import EpisodeStep
from openpi.research.shared.episode_schema import Observation
from openpi.research.shared.episode_schema import make_repack_structure
from openpi.transforms import RepackTransform

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dummy_observation(*, with_optional: bool = False) -> Observation:
    """Create a minimal valid Observation."""
    kwargs = {
        "wrist_rgb": np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8),
        "joint_position": np.random.randn(7).astype(np.float32),
        "joint_velocity": np.random.randn(7).astype(np.float32),
        "gripper_position": np.array([0.5], dtype=np.float32),
    }
    if with_optional:
        kwargs["scene_rgb"] = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        kwargs["base_state"] = np.random.randn(3).astype(np.float32)
    return Observation(**kwargs)


def _make_dummy_action() -> Action:
    return Action(joint_pos=np.random.randn(7).astype(np.float32), gripper_cmd=0.7)


def _make_dummy_step(*, with_optional: bool = False) -> EpisodeStep:
    return EpisodeStep(
        observation=_make_dummy_observation(with_optional=with_optional),
        action=_make_dummy_action(),
        timestamp_s=1.23,
    )


def _make_dummy_episode(n_steps: int = 3, *, with_optional: bool = False) -> Episode:
    return Episode(
        metadata=EpisodeMetadata(task_id="grasp", env_id="lunar_a"),
        labels=EpisodeLabels(success=True),
        steps=[_make_dummy_step(with_optional=with_optional) for _ in range(n_steps)],
        prompt="pick up the sample",
    )


# ---------------------------------------------------------------------------
# EpisodeMetadata tests
# ---------------------------------------------------------------------------


def test_episode_metadata_creation():
    meta = EpisodeMetadata(task_id="payload", env_id="nominal")
    assert meta.task_id == "payload"
    assert meta.env_id == "nominal"
    assert meta.operator_id == ""


def test_episode_metadata_frozen():
    meta = EpisodeMetadata(task_id="payload", env_id="nominal")
    with pytest.raises(dataclasses.FrozenInstanceError):
        meta.task_id = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# EpisodeLabels tests
# ---------------------------------------------------------------------------


def test_episode_labels_success():
    labels = EpisodeLabels(success=True)
    assert labels.success is True
    assert labels.fail_type is None


def test_episode_labels_failure():
    labels = EpisodeLabels(success=False, fail_type="timeout")
    assert labels.success is False
    assert labels.fail_type == "timeout"


# ---------------------------------------------------------------------------
# Observation tests
# ---------------------------------------------------------------------------


def test_observation_required_fields():
    obs = _make_dummy_observation()
    assert obs.wrist_rgb.shape == (64, 64, 3)
    assert obs.joint_position.shape == (7,)
    assert obs.joint_velocity.shape == (7,)
    assert obs.gripper_position.shape == (1,)
    assert obs.scene_rgb is None
    assert obs.base_state is None


def test_observation_optional_fields():
    obs = _make_dummy_observation(with_optional=True)
    assert obs.scene_rgb is not None
    assert obs.scene_rgb.shape == (64, 64, 3)
    assert obs.base_state is not None
    assert obs.base_state.shape == (3,)


def test_observation_frozen():
    obs = _make_dummy_observation()
    with pytest.raises(dataclasses.FrozenInstanceError):
        obs.wrist_rgb = np.zeros((64, 64, 3), dtype=np.uint8)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Action tests
# ---------------------------------------------------------------------------


def test_action_to_array():
    act = Action(joint_pos=np.arange(7, dtype=np.float32), gripper_cmd=0.42)
    arr = act.to_array()
    assert arr.shape == (8,)
    np.testing.assert_allclose(arr[:7], np.arange(7))
    np.testing.assert_allclose(arr[7], 0.42, atol=1e-6)


def test_action_from_array():
    arr = np.array([1, 2, 3, 4, 5, 6, 7, 0.9], dtype=np.float32)
    act = Action.from_array(arr)
    np.testing.assert_allclose(act.joint_pos, arr[:7])
    assert abs(act.gripper_cmd - 0.9) < 1e-6


def test_action_from_array_wrong_shape():
    with pytest.raises(ValueError, match="Expected shape"):
        Action.from_array(np.zeros(5))


def test_action_roundtrip():
    """to_array → from_array preserves values."""
    original = _make_dummy_action()
    reconstructed = Action.from_array(original.to_array())
    np.testing.assert_allclose(reconstructed.joint_pos, original.joint_pos, atol=1e-6)
    assert abs(reconstructed.gripper_cmd - original.gripper_cmd) < 1e-6


# ---------------------------------------------------------------------------
# Episode serialisation tests
# ---------------------------------------------------------------------------


def test_episode_to_dict_structure():
    ep = _make_dummy_episode(n_steps=2)
    d = ep.to_dict()

    assert d["schema_version"] == "1.0"
    assert d["metadata"]["task_id"] == "grasp"
    assert d["labels"]["success"] is True
    assert d["prompt"] == "pick up the sample"
    assert len(d["steps"]) == 2

    step0 = d["steps"][0]
    assert "observation" in step0
    assert "action" in step0
    assert "timestamp_s" in step0


def test_episode_roundtrip_without_optional():
    """to_dict → from_dict preserves all data (required fields only)."""
    original = _make_dummy_episode(n_steps=3, with_optional=False)
    d = original.to_dict()
    restored = Episode.from_dict(d)

    assert restored.metadata.task_id == original.metadata.task_id
    assert restored.labels.success == original.labels.success
    assert restored.prompt == original.prompt
    assert len(restored.steps) == len(original.steps)

    for orig_step, rest_step in zip(original.steps, restored.steps, strict=True):
        np.testing.assert_array_equal(rest_step.observation.wrist_rgb, orig_step.observation.wrist_rgb)
        np.testing.assert_allclose(rest_step.observation.joint_position, orig_step.observation.joint_position)
        np.testing.assert_allclose(rest_step.action.joint_pos, orig_step.action.joint_pos, atol=1e-6)
        assert abs(rest_step.action.gripper_cmd - orig_step.action.gripper_cmd) < 1e-6
        assert rest_step.observation.scene_rgb is None
        assert rest_step.observation.base_state is None


def test_episode_roundtrip_with_optional():
    """to_dict → from_dict preserves optional fields."""
    original = _make_dummy_episode(n_steps=2, with_optional=True)
    restored = Episode.from_dict(original.to_dict())

    for orig_step, rest_step in zip(original.steps, restored.steps, strict=True):
        assert rest_step.observation.scene_rgb is not None
        np.testing.assert_array_equal(rest_step.observation.scene_rgb, orig_step.observation.scene_rgb)
        assert rest_step.observation.base_state is not None
        np.testing.assert_allclose(rest_step.observation.base_state, orig_step.observation.base_state)


def test_episode_to_dict_is_json_serialisable():
    """The dict produced by to_dict() must be JSON-compatible."""
    ep = _make_dummy_episode(n_steps=1, with_optional=True)
    d = ep.to_dict()
    # This will raise if any numpy arrays leaked through
    json_str = json.dumps(d)
    assert isinstance(json_str, str)
    # And round-trip back
    reloaded = json.loads(json_str)
    restored = Episode.from_dict(reloaded)
    assert restored.metadata.task_id == ep.metadata.task_id


def test_episode_schema_version_is_class_constant():
    """SCHEMA_VERSION is set automatically (not via __init__)."""
    ep = _make_dummy_episode(n_steps=1)
    assert ep.SCHEMA_VERSION == "1.0"


# ---------------------------------------------------------------------------
# make_repack_structure tests
# ---------------------------------------------------------------------------


def test_make_repack_structure_keys():
    """Repack structure contains all expected observation + action + prompt keys."""
    structure = make_repack_structure()
    expected_keys = {
        "observation/wrist_image",
        "observation/scene_image",
        "observation/joint_position",
        "observation/joint_velocity",
        "observation/gripper_position",
        "actions",
        "prompt",
    }
    assert set(structure.keys()) == expected_keys


def test_make_repack_structure_with_repack_transform():
    """Dummy data flows through RepackTransform using our repack structure."""
    structure = make_repack_structure()
    transform = RepackTransform(structure=structure)

    # Build a dummy input dict matching LeRobot-style flat keys.
    dummy_input = {
        "observation/wrist_image": np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8),
        "observation/scene_image": np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8),
        "observation/joint_position": np.random.randn(7).astype(np.float32),
        "observation/joint_velocity": np.random.randn(7).astype(np.float32),
        "observation/gripper_position": np.array([0.5], dtype=np.float32),
        "actions": np.random.randn(10, 8).astype(np.float32),
        "prompt": "pick up the rock",
    }

    output = transform(dummy_input)

    # All keys should be preserved (identity mapping in this case).
    assert set(output.keys()) == set(structure.keys())
    np.testing.assert_array_equal(output["observation/wrist_image"], dummy_input["observation/wrist_image"])
    np.testing.assert_array_equal(output["actions"], dummy_input["actions"])
    assert output["prompt"] == "pick up the rock"


def test_make_repack_structure_requires_wrist_image():
    """wrist_image key MUST be present in the repack structure (primary camera)."""
    structure = make_repack_structure()
    assert "observation/wrist_image" in structure
