"""Unified episode schema for SpaceCIL and LunarCompose.

Defines frozen dataclass-based schema covering:
- obs: wrist_rgb, scene_rgb (optional), joint_position, joint_velocity,
       gripper_position, base_state (optional)
- action: 7D absolute joint position + 1D gripper command = 8D total
- lang: instruction prompt
- label: success, fail_type
- meta: task_id, env_id, operator_id, session_id, camera_preset_id,
        calibration_version, scene_revision, object_revision

Provides:
- ``Episode.to_dict()`` / ``Episode.from_dict()`` for serialization
- ``make_repack_structure()`` returning a dict suitable for ``RepackTransform``
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Metadata & Labels
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class EpisodeMetadata:
    """Metadata fields for a single episode."""

    task_id: str
    env_id: str
    operator_id: str = ""
    session_id: str = ""
    camera_preset_id: str = ""
    calibration_version: str = ""
    scene_revision: str = ""
    object_revision: str = ""


@dataclasses.dataclass(frozen=True)
class EpisodeLabels:
    """Labels for a single episode."""

    success: bool
    fail_type: str | None = None


# ---------------------------------------------------------------------------
# Per-step data
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Observation:
    """Single-step observation bundle.

    ``wrist_rgb`` is **required** (primary policy camera).
    ``scene_rgb`` and ``base_state`` are optional peripherals.
    """

    wrist_rgb: np.ndarray  # (H, W, 3) uint8
    joint_position: np.ndarray  # (7,) float — 7-DoF joint angles (arm only, excl. gripper)
    joint_velocity: np.ndarray  # (7,) float
    gripper_position: np.ndarray  # (1,) float, normalised [0, 1]
    scene_rgb: np.ndarray | None = None  # (H, W, 3) uint8, optional
    base_state: np.ndarray | None = None  # (N,) float, optional


@dataclasses.dataclass(frozen=True)
class Action:
    """Single-step action in canonical absolute joint position + gripper space.

    ``joint_pos``: 7D — absolute joint angles (radians) for 7-DoF arm.
    ``gripper_cmd``: scalar in [0, 1].
    """

    joint_pos: np.ndarray  # (7,) float
    gripper_cmd: float  # [0, 1]

    def to_array(self) -> np.ndarray:
        """Flatten to (8,) training array: [joint_pos(7), gripper(1)]."""
        return np.concatenate([self.joint_pos, np.array([self.gripper_cmd], dtype=np.float32)])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> Action:
        """Reconstruct from (8,) array."""
        arr = np.asarray(arr, dtype=np.float32)
        if arr.shape != (8,):
            raise ValueError(f"Expected shape (8,), got {arr.shape}")
        return cls(joint_pos=arr[:7], gripper_cmd=float(arr[7]))


@dataclasses.dataclass(frozen=True)
class EpisodeStep:
    """One timestep inside an episode."""

    observation: Observation
    action: Action
    timestamp_s: float = 0.0


# ---------------------------------------------------------------------------
# Full episode
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Episode:
    """Complete episode container.

    Serialisable via ``to_dict()`` / ``from_dict()`` for checkpoint I/O and
    debugging.  Arrays are stored as nested Python lists in the dict
    representation so the output is JSON-friendly.
    """

    SCHEMA_VERSION: str = dataclasses.field(default="1.0", init=False)

    metadata: EpisodeMetadata
    labels: EpisodeLabels
    steps: list[EpisodeStep]
    prompt: str = ""

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the episode to a JSON-compatible dict."""
        return {
            "schema_version": self.SCHEMA_VERSION,
            "metadata": dataclasses.asdict(self.metadata),
            "labels": dataclasses.asdict(self.labels),
            "prompt": self.prompt,
            "steps": [_step_to_dict(s) for s in self.steps],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Episode:
        """Reconstruct an Episode from a dict produced by ``to_dict``."""
        metadata = EpisodeMetadata(**d["metadata"])
        labels = EpisodeLabels(**d["labels"])
        steps = [_step_from_dict(s) for s in d["steps"]]
        return cls(metadata=metadata, labels=labels, steps=steps, prompt=d.get("prompt", ""))


# ---------------------------------------------------------------------------
# RepackTransform helper
# ---------------------------------------------------------------------------


def make_repack_structure() -> dict[str, str]:
    """Return a flat ``{dst_key: src_key}`` mapping for ``RepackTransform``.

    This maps LeRobot-style dataset keys to the inference-like keys expected
    by downstream ``DataTransformFn`` classes (e.g. ``RM75Inputs``).

    Source keys use ``/`` separators (LeRobot convention).
    Destination keys also use ``/`` separators (openpi convention).
    """
    return {
        "observation/wrist_image": "observation/wrist_image",
        "observation/scene_image": "observation/scene_image",
        "observation/joint_position": "observation/joint_position",
        "observation/joint_velocity": "observation/joint_velocity",
        "observation/gripper_position": "observation/gripper_position",
        "actions": "actions",
        "prompt": "prompt",
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ndarray_to_list(arr: np.ndarray | None) -> list | None:
    if arr is None:
        return None
    return np.asarray(arr).tolist()


def _list_to_ndarray(lst: list | None, dtype: type = np.float32) -> np.ndarray | None:
    if lst is None:
        return None
    return np.asarray(lst, dtype=dtype)


def _step_to_dict(step: EpisodeStep) -> dict[str, Any]:
    obs = step.observation
    act = step.action
    return {
        "observation": {
            "wrist_rgb": _ndarray_to_list(obs.wrist_rgb),
            "joint_position": _ndarray_to_list(obs.joint_position),
            "joint_velocity": _ndarray_to_list(obs.joint_velocity),
            "gripper_position": _ndarray_to_list(obs.gripper_position),
            "scene_rgb": _ndarray_to_list(obs.scene_rgb),
            "base_state": _ndarray_to_list(obs.base_state),
        },
        "action": {
            "joint_pos": _ndarray_to_list(act.joint_pos),
            "gripper_cmd": act.gripper_cmd,
        },
        "timestamp_s": step.timestamp_s,
    }


def _step_from_dict(d: dict[str, Any]) -> EpisodeStep:
    obs_d = d["observation"]
    act_d = d["action"]
    obs = Observation(
        wrist_rgb=np.asarray(obs_d["wrist_rgb"], dtype=np.uint8),
        joint_position=np.asarray(obs_d["joint_position"], dtype=np.float32),
        joint_velocity=np.asarray(obs_d["joint_velocity"], dtype=np.float32),
        gripper_position=np.asarray(obs_d["gripper_position"], dtype=np.float32),
        scene_rgb=_list_to_ndarray(obs_d.get("scene_rgb"), dtype=np.uint8),
        base_state=_list_to_ndarray(obs_d.get("base_state")),
    )
    act = Action(
        joint_pos=np.asarray(act_d["joint_pos"], dtype=np.float32),
        gripper_cmd=float(act_d["gripper_cmd"]),
    )
    return EpisodeStep(observation=obs, action=act, timestamp_s=d.get("timestamp_s", 0.0))
