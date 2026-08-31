"""Crash-safe raw HDF5 episode writer; no LeRobot code runs in the control loop."""

from __future__ import annotations

import argparse
from contextlib import suppress
import json
import os
from pathlib import Path

import h5py
import numpy as np

try:
    from .config_loader import config_hash
    from .config_loader import load_project_config
    from .robot_runtime import FakeHardware
except ImportError:
    from config_loader import config_hash
    from config_loader import load_project_config
    from robot_runtime import FakeHardware


class EpisodeWriter:
    def __init__(self, root: str | Path, episode_id: int, image_shape: tuple[int, int, int], attrs: dict):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "rejected").mkdir(exist_ok=True)
        self.tmp = self.root / f"episode_{episode_id:04d}.tmp.hdf5"
        self.final = self.root / f"episode_{episode_id:04d}.hdf5"
        if self.tmp.exists() or self.final.exists():
            raise FileExistsError(self.tmp if self.tmp.exists() else self.final)
        self.file = h5py.File(self.tmp, "x")
        self._datasets = {}
        for key, shape, dtype in (
            ("observations/qpos", (9,), "f4"),
            ("action", (9,), "f4"),
            ("timestamps/control", (), "i8"),
            ("timestamps/front", (), "i8"),
            ("timestamps/side", (), "i8"),
            ("timestamps/top", (), "i8"),
            ("timestamps_host/front", (), "i8"),
            ("timestamps_host/side", (), "i8"),
            ("timestamps_host/top", (), "i8"),
            ("observations/images/front", image_shape, "u1"),
            ("observations/images/side", image_shape, "u1"),
            ("observations/images/top", image_shape, "u1"),
        ):
            self._datasets[key] = self.file.create_dataset(
                key, shape=(0, *shape), maxshape=(None, *shape), dtype=dtype, chunks=(1, *shape) if shape else (256,)
            )
        for key, value in attrs.items():
            self.file.attrs[key] = json.dumps(value) if isinstance(value, dict | list | tuple) else value
        self.closed = False

    def append(self, observation: dict, action: np.ndarray):
        values = {
            "observations/qpos": observation["qpos"],
            "action": action,
            "timestamps/control": observation["timestamp_ns"],
        }
        for name in ("front", "side", "top"):
            values[f"timestamps/{name}"] = observation["camera_timestamps_ns"][name]
            host_timestamps = observation.get("camera_host_timestamps_ns", observation["camera_timestamps_ns"])
            values[f"timestamps_host/{name}"] = host_timestamps[name]
            values[f"observations/images/{name}"] = observation["images"][name]
        for key, value in values.items():
            ds = self._datasets[key]
            ds.resize(ds.shape[0] + 1, axis=0)
            ds[-1] = value

    def finish(self, *, success=True):
        self.file.attrs["success"] = bool(success)
        self.file.flush()
        with suppress(AttributeError, OSError):
            os.fsync(self.file.id.get_vfd_handle())
        self.file.close()
        self.closed = True
        destination = self.final if success else self.root / "rejected" / self.tmp.name.replace(".tmp", "")
        os.replace(self.tmp, destination)
        return destination

    def reject(self, reason: str):
        self.file.attrs["rejection_reason"] = reason
        return self.finish(success=False)

    def __enter__(self):
        return self

    def __exit__(self, kind, value, tb):
        if not self.closed:
            self.reject("exception" if kind else "not finalized")


def recover_stale_episodes(root: str | Path) -> list[Path]:
    """Move crash-left ``*.tmp.hdf5`` files to ``rejected`` without deleting data."""
    root = Path(root)
    rejected = root / "rejected"
    rejected.mkdir(parents=True, exist_ok=True)
    recovered: list[Path] = []
    for temporary in sorted(root.glob("episode_*.tmp.hdf5")):
        destination = rejected / temporary.name.replace(".tmp.hdf5", ".hdf5")
        suffix = 1
        while destination.exists():
            destination = rejected / temporary.name.replace(".tmp.hdf5", f".recovered_{suffix}.hdf5")
            suffix += 1
        try:
            with h5py.File(temporary, "r+") as episode:
                episode.attrs["success"] = False
                episode.attrs["rejection_reason"] = "recovered stale temp file after process interruption"
                episode.flush()
        except OSError:
            # Preserve even a truncated HDF5 file for forensic inspection.
            destination = destination.with_name(destination.stem + ".corrupt.hdf5")
        os.replace(temporary, destination)
        recovered.append(destination)
    return recovered


def record_fake(root: str | Path, frames=12):
    project = load_project_config()
    servo_zero = tuple(int(value) for value in project["poses"]["wrist"]["servo_zero_raw"])
    hardware = FakeHardware(servo_zero_raw=servo_zero)
    hardware.connect()
    obs = hardware.get_observation()
    wrist_safety = project["safety"]["wrist"]
    attrs = {
        "task": "fake acceptance",
        "fps": 10,
        "git_commit": "test",
        "camera_serials": {
            str(item["role"]): str(item["serial"])
            for item in project["hardware"]["cameras"]["devices"]
        },
        "gripper_backend": "fake",
        "gripper_state_source": "commanded",
        "action_space": project["collection"]["action"]["semantics"],
        "wrist_coordinate": project["poses"]["wrist"]["coordinate"],
        "wrist_servo_zero_raw": servo_zero,
        "wrist_raw_limits": {
            "j1": [wrist_safety["j1_min_raw"], wrist_safety["j1_max_raw"]],
            "j2": [wrist_safety["j2_min_raw"], wrist_safety["j2_max_raw"]],
        },
        "robot_config_hash": config_hash(project),
    }
    with EpisodeWriter(root, 0, obs["images"]["front"].shape, attrs) as writer:
        for _ in range(frames):
            writer.append(hardware.get_observation(), np.zeros(9, np.float32))
        return writer.finish(success=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--fake", action="store_true")
    args = parser.parse_args()
    if not args.fake:
        raise SystemExit("真机数采请使用 teleop_collect.py; 本脚本只提供独立 Fake HDF5 验收")
    print(record_fake(args.output))


if __name__ == "__main__":
    main()
