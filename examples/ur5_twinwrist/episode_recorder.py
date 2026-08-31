# ruff: noqa: RUF001, RUF002, RUF003
"""项目内 10 Hz 原始 HDF5 Episode 录制状态机。"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
import fcntl
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from .config_loader import config_hash
from .record_hdf5 import EpisodeWriter
from .record_hdf5 import recover_stale_episodes

CAMERA_ROLES = ("front", "side", "top")


def next_episode_id(root: str | Path) -> int:
    """返回成功、失败和临时文件都不会碰撞的下一个编号。"""

    path = Path(root)
    identifiers: list[int] = []
    for directory in (path, path / "rejected"):
        for candidate in directory.glob("episode_*.hdf5"):
            token = candidate.name.removeprefix("episode_").split(".", 1)[0]
            if token.isdigit():
                identifiers.append(int(token))
    return max(identifiers, default=-1) + 1


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def episode_attributes(project: Mapping[str, Any], observation: Mapping[str, Any]) -> dict[str, Any]:
    """从权威 YAML 和首帧健康状态生成 HDF5 元数据。"""

    root = Path(__file__).resolve().parents[2]
    camera_serials = {
        str(item["role"]): str(item["serial"])
        for item in project["hardware"]["cameras"]["devices"]
    }
    gripper_health = observation.get("hardware_health", {}).get("gripper", {})
    wrist_pose = project["poses"]["wrist"]
    wrist_safety = project["safety"]["wrist"]
    state_source = str(gripper_health.get("state_source", project["hardware"]["gripper"]["state_source"]))
    return {
        "task": str(project["collection"]["task"]["prompt"]),
        "task_id": str(project["collection"]["task"]["id"]),
        "fps": float(project["collection"]["capture"]["record_hz"]),
        "capture_fps": float(project["collection"]["capture"]["camera_fps"]),
        "success": False,
        "git_commit": _git_commit(root),
        "camera_serials": camera_serials,
        "gripper_backend": str(project["hardware"]["gripper"]["backend"]),
        "gripper_state_source": state_source,
        "action_space": str(project["collection"]["action"]["semantics"]),
        "state_space": list(project["collection"]["state"]["fields"]),
        "action_fields": list(project["collection"]["action"]["fields"]),
        "wrist_coordinate": str(wrist_pose["coordinate"]),
        "wrist_servo_zero_raw": list(wrist_pose["servo_zero_raw"]),
        "wrist_raw_limits": {
            "j1": [int(wrist_safety["j1_min_raw"]), int(wrist_safety["j1_max_raw"])],
            "j2": [int(wrist_safety["j2_min_raw"]), int(wrist_safety["j2_max_raw"])],
        },
        "robot_config_hash": config_hash(dict(project)),
    }


def _validate_frame(
    observation: Mapping[str, Any],
    action: np.ndarray,
    *,
    expected_image_shape: tuple[int, int, int] | None = None,
) -> None:
    state = np.asarray(observation.get("qpos"), dtype=np.float64)
    if state.shape != (9,) or not np.isfinite(state).all():
        raise ValueError("observation.qpos 必须是 9 维有限数")
    if action.shape != (9,) or not np.isfinite(action).all():
        raise ValueError("最终 action 必须是 9 维有限数")
    images = observation.get("images")
    if not isinstance(images, Mapping) or set(images) != set(CAMERA_ROLES):
        raise ValueError("observation.images 必须恰好包含 front/side/top")
    shapes = set()
    for role in CAMERA_ROLES:
        image = np.asarray(images[role])
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or image.size == 0:
            raise ValueError(f"{role} 图像必须是非空 uint8 HxWx3")
        if expected_image_shape is not None and image.shape != expected_image_shape:
            raise ValueError(
                f"{role} 图像 shape={image.shape}, 与 collection.yaml {expected_image_shape} 不一致"
            )
        shapes.add(image.shape)
    if len(shapes) != 1:
        raise ValueError("三路图像 shape 必须一致")


class EpisodeRecorder:
    """只接收缓存 observation 和已经发送成功的 action，不访问硬件。"""

    def __init__(self, root: str | Path, project: Mapping[str, Any]) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.project = dict(project)
        self._period_ns = round(1e9 / float(project["collection"]["capture"]["record_hz"]))
        self._expected_image_shape = tuple(
            int(value) for value in project["collection"]["capture"]["image_shape"]
        )
        self._writer: EpisodeWriter | None = None
        self._last_written_ns: int | None = None
        self._last_camera_sequences: dict[str, int] | None = None
        self._frames = 0
        self._lock_stream = (self.root / ".writer.lock").open("a+")
        try:
            fcntl.flock(self._lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock_stream.close()
            raise RuntimeError(f"另一个录制器正在使用 {self.root}") from exc
        recover_stale_episodes(self.root)
        self._episode_id = next_episode_id(self.root)

    @property
    def active(self) -> bool:
        return self._writer is not None

    @property
    def frame_count(self) -> int:
        return self._frames

    @property
    def episode_id(self) -> int:
        return self._episode_id

    def start(self, observation: Mapping[str, Any]) -> Path:
        if self._writer is not None:
            raise RuntimeError("已有 Episode 正在录制")
        placeholder = np.zeros(9, dtype=np.float32)
        _validate_frame(
            observation,
            placeholder,
            expected_image_shape=self._expected_image_shape,
        )
        shape = np.asarray(observation["images"]["front"]).shape
        self._writer = EpisodeWriter(
            self.root,
            self._episode_id,
            tuple(int(value) for value in shape),
            episode_attributes(self.project, observation),
        )
        self._last_written_ns = None
        self._last_camera_sequences = None
        self._frames = 0
        return self._writer.tmp

    def append_if_due(self, observation: Mapping[str, Any], final_action: Any) -> bool:
        writer = self._writer
        if writer is None:
            return False
        action = np.asarray(final_action, dtype=np.float32).reshape(-1)
        _validate_frame(
            observation,
            action,
            expected_image_shape=self._expected_image_shape,
        )
        timestamp_ns = int(observation["timestamp_ns"])
        if self._last_written_ns is not None:
            if timestamp_ns <= self._last_written_ns:
                self.reject("control timestamp is not strictly monotonic")
                raise ValueError("control timestamp is not strictly monotonic")
            if timestamp_ns - self._last_written_ns < self._period_ns:
                return False
        sequences = observation.get("camera_sequences")
        if isinstance(sequences, Mapping):
            parsed = {role: int(sequences[role]) for role in CAMERA_ROLES}
            if self._last_camera_sequences is not None and any(
                parsed[role] <= self._last_camera_sequences[role] for role in CAMERA_ROLES
            ):
                self.reject("camera frame repeated or sequence moved backwards")
                raise ValueError("camera frame repeated or sequence moved backwards")
            self._last_camera_sequences = parsed
        writer.append(dict(observation), action)
        self._last_written_ns = timestamp_ns
        self._frames += 1
        return True

    def save(self) -> Path:
        if self._writer is None:
            raise RuntimeError("没有活动 Episode")
        if self._frames == 0:
            path = self.reject("episode contains no frames")
            raise RuntimeError(f"空 Episode 已拒绝: {path}")
        writer, self._writer = self._writer, None
        path = writer.finish(success=True)
        self._advance()
        return path

    def reject(self, reason: str) -> Path:
        if self._writer is None:
            raise RuntimeError("没有活动 Episode")
        writer, self._writer = self._writer, None
        path = writer.reject(reason)
        self._advance()
        return path

    def close(self) -> None:
        if self._writer is not None:
            with suppress(Exception):
                self.reject("collector closed before operator save")
        if not self._lock_stream.closed:
            fcntl.flock(self._lock_stream.fileno(), fcntl.LOCK_UN)
            self._lock_stream.close()

    def _advance(self) -> None:
        self._episode_id += 1
        self._last_written_ns = None
        self._last_camera_sequences = None
        self._frames = 0

    def __enter__(self) -> EpisodeRecorder:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


__all__ = ["EpisodeRecorder", "episode_attributes", "next_episode_id"]
