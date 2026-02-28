"""Unified episode schema for SpaceCIL and LunarCompose.

Defines frozen dataclass-based schema covering:
- obs: wrist_rgb, scene_rgb (optional), q, dq, gripper, base_state
- action: delta_ee, gripper_cmd
- lang: instruction
- label: success, fail_type
- meta: task_id, env_id, operator_id, session_id, camera_preset_id,
        calibration_version, scene_revision, object_revision
"""

import dataclasses


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
