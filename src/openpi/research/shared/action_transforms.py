"""Action transform layer for RM75 Delta EE + gripper action space.

Provides the authoritative transform path:
  teleop output → canonical Delta EE + gripper → training format

Action dimension layout (7D total):
  [0:3]  — delta position  (metres)
  [3:6]  — delta orientation  (axis-angle, radians)
  [6]    — gripper command  (normalised [0, 1])

Follows openpi's ``DataTransformFn`` pattern (like ``DeltaActions``).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from openpi import transforms

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTION_DIM: int = 7
"""Total action dimensionality: 6D delta EE + 1D gripper."""

EE_DIM: int = 6
"""Delta end-effector dimensionality (3 pos + 3 orient)."""

GRIPPER_RANGE: tuple[float, float] = (0.0, 1.0)
"""Valid gripper command range after normalisation."""

# Mask for DeltaActions / AbsoluteActions:
#   True on first 6 dims (delta applied), False on gripper (absolute).
RM75_DELTA_MASK: tuple[bool, ...] = transforms.make_bool_mask(EE_DIM, -1)


# ---------------------------------------------------------------------------
# Functional transforms  (teleop ↔ canonical ↔ training)
# ---------------------------------------------------------------------------


def _validate_and_clip(action: np.ndarray) -> np.ndarray:
    """Validate shape, ensure float32, and clip gripper to [0, 1].

    Accepts both single-step ``(7,)`` and batched ``(…, 7)`` shapes.
    """
    action = np.asarray(action, dtype=np.float32)
    if action.shape[-1] != ACTION_DIM:
        raise ValueError(f"Last action dim must be {ACTION_DIM}, got {action.shape[-1]}")
    # Clip gripper to valid range (in-place on copy)
    action = action.copy()
    action[..., -1] = np.clip(action[..., -1], *GRIPPER_RANGE)
    return action


def teleop_to_canonical(teleop_action: np.ndarray) -> np.ndarray:
    """Convert raw teleop controller output to canonical Delta EE + gripper.

    Currently an identity-like transform (validate + clip) since the teleop
    system already produces Delta EE + gripper.  If a future teleop
    controller uses a different representation this function is the single
    place to adapt.

    Args:
        teleop_action: ``(7,)`` or ``(…, 7)`` raw teleop output.

    Returns:
        Canonical action array with same shape, float32, gripper clipped.
    """
    return _validate_and_clip(teleop_action)


def canonical_to_training(canonical_action: np.ndarray) -> np.ndarray:
    """Convert canonical Delta EE + gripper to openpi training format.

    The canonical format **is** the training format for our RM75 pipeline —
    both use ``[delta_pos(3), delta_orient(3), gripper(1)]``.  This function
    validates the shape and clips the gripper, acting as a safety gate.

    Args:
        canonical_action: ``(7,)`` or ``(…, 7)`` canonical action.

    Returns:
        Training-format action array, float32, gripper clipped.
    """
    return _validate_and_clip(canonical_action)


def training_to_canonical(training_action: np.ndarray) -> np.ndarray:
    """Convert openpi training format back to canonical Delta EE + gripper.

    Exact inverse of ``canonical_to_training`` (trivially invertible since
    the two formats are identical up to validation + clipping).

    Args:
        training_action: ``(7,)`` or ``(…, 7)`` training action.

    Returns:
        Canonical action array, float32, gripper clipped.
    """
    return _validate_and_clip(training_action)


# ---------------------------------------------------------------------------
# DataTransformFn classes (plug into openpi data pipeline)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RM75DeltaActions(transforms.DataTransformFn):
    """Convert absolute joint-space actions to delta actions for RM75.

    Follows the same pattern as ``transforms.DeltaActions``:
    ``actions[…, :6] -= state[…, :6]`` for the masked (delta) dimensions,
    leaving the gripper (dim 6) as-is (absolute).

    Intended for use in ``DataConfig.data_transforms.inputs``.
    """

    mask: tuple[bool, ...] = RM75_DELTA_MASK

    def __call__(self, data: transforms.DataDict) -> transforms.DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        # Subtract state from masked dims to produce delta.
        actions = np.array(actions, dtype=np.float32)
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions
        return data


@dataclasses.dataclass(frozen=True)
class RM75AbsoluteActions(transforms.DataTransformFn):
    """Convert delta actions back to absolute joint-space actions for RM75.

    Inverse of ``RM75DeltaActions``: adds state back to the masked dims.

    Intended for use in ``DataConfig.data_transforms.outputs``.
    """

    mask: tuple[bool, ...] = RM75_DELTA_MASK

    def __call__(self, data: transforms.DataDict) -> transforms.DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        # Add state back to masked dims to recover absolute.
        actions = np.array(actions, dtype=np.float32)
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions
        return data
