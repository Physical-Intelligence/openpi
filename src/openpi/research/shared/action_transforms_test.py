"""Tests for action_transforms module."""

import numpy as np
import pytest

from openpi import transforms
from openpi.research.shared.action_transforms import ACTION_DIM
from openpi.research.shared.action_transforms import JOINT_DIM
from openpi.research.shared.action_transforms import GRIPPER_RANGE
from openpi.research.shared.action_transforms import RM75_DELTA_MASK
from openpi.research.shared.action_transforms import RM75AbsoluteActions
from openpi.research.shared.action_transforms import RM75DeltaActions
from openpi.research.shared.action_transforms import canonical_to_training
from openpi.research.shared.action_transforms import teleop_to_canonical
from openpi.research.shared.action_transforms import training_to_canonical

# ---------------------------------------------------------------------------
# Constants sanity checks
# ---------------------------------------------------------------------------


def test_constants():
    assert ACTION_DIM == 8
    assert JOINT_DIM == 7


def test_rm75_delta_mask():
    """Mask should be True for first 7 dims, False for gripper."""
    assert RM75_DELTA_MASK == (True, True, True, True, True, True, True, False)
    assert len(RM75_DELTA_MASK) == ACTION_DIM


# ---------------------------------------------------------------------------
# Functional transforms — single step
# ---------------------------------------------------------------------------


def test_teleop_to_canonical_valid():
    action = np.array([0.1, -0.2, 0.3, 0.01, -0.01, 0.02, 0.15, 0.5], dtype=np.float32)
    result = teleop_to_canonical(action)
    assert result.shape == (8,)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result[:7], action[:7])
    np.testing.assert_allclose(result[7], 0.5)


def test_canonical_to_training_valid():
    action = np.array([0.1, -0.2, 0.3, 0.01, -0.01, 0.02, 0.15, 0.8], dtype=np.float32)
    result = canonical_to_training(action)
    np.testing.assert_allclose(result, action)


def test_training_to_canonical_valid():
    action = np.array([0.1, -0.2, 0.3, 0.01, -0.01, 0.02, 0.15, 0.3], dtype=np.float32)
    result = training_to_canonical(action)
    np.testing.assert_allclose(result, action)


def test_gripper_clipping():
    """Gripper values outside [0, 1] must be clipped."""
    action_over = np.array([0, 0, 0, 0, 0, 0, 0, 1.5], dtype=np.float32)
    action_under = np.array([0, 0, 0, 0, 0, 0, 0, -0.3], dtype=np.float32)

    assert teleop_to_canonical(action_over)[7] == 1.0
    assert teleop_to_canonical(action_under)[7] == 0.0
    assert canonical_to_training(action_over)[7] == 1.0
    assert training_to_canonical(action_under)[7] == 0.0


def test_wrong_dim_raises():
    """Actions with wrong last dim should raise ValueError."""
    bad = np.zeros(5, dtype=np.float32)
    with pytest.raises(ValueError, match="must be 8"):
        teleop_to_canonical(bad)
    with pytest.raises(ValueError, match="must be 8"):
        canonical_to_training(bad)
    with pytest.raises(ValueError, match="must be 8"):
        training_to_canonical(bad)


# ---------------------------------------------------------------------------
# Functional transforms — batched
# ---------------------------------------------------------------------------


def test_batched_transforms():
    """All functional transforms handle (B, T, 8) batched shapes."""
    batch = np.random.randn(4, 10, 8).astype(np.float32)
    batch[..., -1] = np.clip(batch[..., -1], 0, 1)  # valid gripper

    for fn in (teleop_to_canonical, canonical_to_training, training_to_canonical):
        result = fn(batch)
        assert result.shape == (4, 10, 8)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# Roundtrip invertibility
# ---------------------------------------------------------------------------


def test_teleop_canonical_training_roundtrip():
    """teleop → canonical → training → canonical should preserve values."""
    original = np.array([0.05, -0.1, 0.2, 0.01, -0.02, 0.03, 0.04, 0.65], dtype=np.float32)
    step1 = teleop_to_canonical(original)
    step2 = canonical_to_training(step1)
    step3 = training_to_canonical(step2)
    np.testing.assert_allclose(step3, original, atol=1e-7)


def test_training_canonical_roundtrip_batched():
    """Roundtrip preserves batched data."""
    batch = np.random.randn(3, 5, 8).astype(np.float32)
    batch[..., -1] = np.clip(batch[..., -1], 0, 1)
    result = training_to_canonical(canonical_to_training(batch))
    np.testing.assert_allclose(result, batch, atol=1e-7)


# ---------------------------------------------------------------------------
# Does not mutate input
# ---------------------------------------------------------------------------


def test_no_input_mutation():
    """Functional transforms must not modify the input array."""
    original = np.array([0.1, -0.2, 0.3, 0.01, -0.01, 0.02, 0.04, 0.5], dtype=np.float32)
    snapshot = original.copy()
    teleop_to_canonical(original)
    np.testing.assert_array_equal(original, snapshot)


# ---------------------------------------------------------------------------
# RM75DeltaActions (DataTransformFn)
# ---------------------------------------------------------------------------


def _make_data_dict(*, state_dim: int = 8, action_horizon: int = 10) -> dict:
    """Build a minimal data dict matching openpi pipeline expectations."""
    return {
        "state": np.random.randn(state_dim).astype(np.float32),
        "actions": np.random.randn(action_horizon, ACTION_DIM).astype(np.float32),
        "prompt": "test prompt",
    }


def test_rm75_delta_actions_basic():
    """RM75DeltaActions subtracts state from first 7 action dims."""
    data = _make_data_dict()
    state = data["state"].copy()
    actions_before = data["actions"].copy()

    delta_fn = RM75DeltaActions()
    result = delta_fn(data)

    # First 7 dims: action - state
    expected = actions_before.copy()
    expected[:, :JOINT_DIM] -= state[:JOINT_DIM]
    np.testing.assert_allclose(result["actions"][:, :JOINT_DIM], expected[:, :JOINT_DIM], atol=1e-6)
    # Gripper (dim 7) unchanged
    np.testing.assert_allclose(result["actions"][:, 7], actions_before[:, 7], atol=1e-6)


def test_rm75_absolute_actions_basic():
    """RM75AbsoluteActions adds state back to first 7 action dims."""
    data = _make_data_dict()
    state = data["state"].copy()
    actions_before = data["actions"].copy()

    abs_fn = RM75AbsoluteActions()
    result = abs_fn(data)

    expected = actions_before.copy()
    expected[:, :JOINT_DIM] += state[:JOINT_DIM]
    np.testing.assert_allclose(result["actions"][:, :JOINT_DIM], expected[:, :JOINT_DIM], atol=1e-6)
    np.testing.assert_allclose(result["actions"][:, 7], actions_before[:, 7], atol=1e-6)


def test_delta_absolute_roundtrip():
    """DeltaActions → AbsoluteActions should recover original actions."""
    data = _make_data_dict()
    original_actions = data["actions"].copy()
    original_state = data["state"].copy()

    # Forward: absolute → delta
    delta_fn = RM75DeltaActions()
    data = delta_fn(data)

    # Inverse: delta → absolute (need to restore state since transforms share data dict)
    data["state"] = original_state
    abs_fn = RM75AbsoluteActions()
    data = abs_fn(data)

    np.testing.assert_allclose(data["actions"], original_actions, atol=1e-6)


def test_absolute_delta_roundtrip():
    """AbsoluteActions → DeltaActions should recover original actions."""
    data = _make_data_dict()
    original_actions = data["actions"].copy()
    original_state = data["state"].copy()

    abs_fn = RM75AbsoluteActions()
    data = abs_fn(data)

    data["state"] = original_state
    delta_fn = RM75DeltaActions()
    data = delta_fn(data)

    np.testing.assert_allclose(data["actions"], original_actions, atol=1e-6)


def test_delta_no_actions_passthrough():
    """If 'actions' key is missing, DeltaActions is a no-op."""
    data = {"state": np.zeros(8), "prompt": "test"}
    delta_fn = RM75DeltaActions()
    result = delta_fn(data)
    assert "actions" not in result


def test_absolute_no_actions_passthrough():
    """If 'actions' key is missing, AbsoluteActions is a no-op."""
    data = {"state": np.zeros(8), "prompt": "test"}
    abs_fn = RM75AbsoluteActions()
    result = abs_fn(data)
    assert "actions" not in result


def test_delta_none_mask_passthrough():
    """If mask is None, DeltaActions is a no-op."""
    data = _make_data_dict()
    original = data["actions"].copy()
    delta_fn = RM75DeltaActions(mask=None)
    result = delta_fn(data)
    np.testing.assert_array_equal(result["actions"], original)


# ---------------------------------------------------------------------------
# DataTransformFn protocol compliance
# ---------------------------------------------------------------------------


def test_rm75_delta_is_data_transform_fn():
    """RM75DeltaActions must satisfy the DataTransformFn protocol."""
    assert isinstance(RM75DeltaActions(), transforms.DataTransformFn)


def test_rm75_absolute_is_data_transform_fn():
    """RM75AbsoluteActions must satisfy the DataTransformFn protocol."""
    assert isinstance(RM75AbsoluteActions(), transforms.DataTransformFn)
