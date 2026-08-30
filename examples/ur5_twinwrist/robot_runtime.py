"""Fail-closed runtime with FakeHardware and an explicit real-motion gate."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
import argparse
from pathlib import Path
import time
import tomllib

import numpy as np


class RobotHardware(ABC):
    @abstractmethod
    def connect(self) -> None: ...
    @abstractmethod
    def get_observation(self) -> dict: ...
    @abstractmethod
    def send_action(self, action: np.ndarray) -> None: ...
    @abstractmethod
    def stop(self) -> None: ...
    @abstractmethod
    def close(self) -> None: ...


class FakeHardware(RobotHardware):
    def __init__(self, shape=(48, 64, 3)):
        self.shape, self.qpos, self.sent_actions = shape, np.zeros(9, np.float32), []
        self.connected = False

    def connect(self):
        self.connected = True

    def get_observation(self):
        now = time.monotonic_ns()
        image = np.full(self.shape, 32, np.uint8)
        return {
            "images": {k: image.copy() for k in ("front", "side", "top")},
            "qpos": self.qpos.copy(),
            "tcp_pose": np.asarray([0, 0, 0.2, 0, 0, 0], np.float32),
            "timestamp_ns": now,
            "camera_timestamps_ns": dict.fromkeys(("front", "side", "top"), now),
            "hardware_health": {"ok": True, "emergency_stop": False},
        }

    def send_action(self, action):
        self.sent_actions.append(np.asarray(action, np.float32).copy())

    def stop(self):
        pass

    def close(self):
        self.connected = False


def safety_filter(action: np.ndarray, observation: dict, config: dict) -> np.ndarray:
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (9,) or not np.isfinite(action).all():
        raise RuntimeError("unsafe action: expected 9 finite values")
    now = time.monotonic_ns()
    timeout_ns = int(config.get("observation_timeout_ms", 200) * 1e6)
    if now - int(observation["timestamp_ns"]) > timeout_ns:
        raise RuntimeError("stale observation")
    health = observation.get("hardware_health", {})
    if not health.get("ok", False) or health.get("emergency_stop", False):
        raise RuntimeError("hardware unhealthy or emergency stop active")
    qpos = np.asarray(observation["qpos"], float)
    if qpos.shape != (9,) or not np.isfinite(qpos).all():
        raise RuntimeError("invalid robot state")
    lo = np.asarray(config.get("ur_joint_min_rad", [-6.28] * 6), float)
    hi = np.asarray(config.get("ur_joint_max_rad", [6.28] * 6), float)
    if np.any(qpos[:6] < lo) or np.any(qpos[:6] > hi):
        raise RuntimeError("UR5 joint soft limit violated")
    linear = float(config.get("max_tcp_linear_m_s", 0.10))
    angular = float(config.get("max_tcp_angular_rad_s", 0.30))
    action[:3] = np.clip(action[:3], -linear, linear)
    action[3:6] = np.clip(action[3:6], -angular, angular)
    wlo = np.asarray(config.get("wrist_min_rad", [-1.57, -1.57]), float)
    whi = np.asarray(config.get("wrist_max_rad", [1.57, 1.57]), float)
    wrist = np.clip(action[6:8], wlo, whi)
    max_step = float(config.get("max_wrist_step_rad", 0.05))
    action[6:8] = np.clip(wrist, qpos[6:8] - max_step, qpos[6:8] + max_step)
    action[8] = np.clip(action[8], 0.0, 1.0)
    tcp = np.asarray(observation.get("tcp_pose", np.zeros(6)), float)
    xyz_min = np.asarray(config.get("workspace_min_m", [-0.8, -0.8, 0.08]), float)
    xyz_max = np.asarray(config.get("workspace_max_m", [0.8, 0.8, 1.2]), float)
    if np.any(tcp[:3] < xyz_min) or np.any(tcp[:3] > xyz_max):
        raise RuntimeError("workspace boundary violated")
    return action.astype(np.float32)


def load_config(path: str | Path) -> dict:
    with open(path, "rb") as stream:
        return tomllib.load(stream)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("collect", "infer"), default="infer")
    parser.add_argument("--shadow", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--prompt", default="perform the task")
    parser.add_argument("--gripper-backend", choices=("hiwonder", "feetech"))
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.example.toml")))
    parser.add_argument("--fake", action="store_true")
    args = parser.parse_args()
    shadow = True if args.mode == "infer" and args.shadow is None else bool(args.shadow)
    if not shadow and not args.enable_motion:
        raise SystemExit("motion denied: pass --enable-motion (inference defaults to --shadow)")
    if not args.fake:
        raise SystemExit("real bridge must be configured locally; use --fake for software acceptance")
    hardware = FakeHardware()
    hardware.connect()
    try:
        obs = hardware.get_observation()
        action = safety_filter(np.zeros(9), obs, load_config(args.config)["robot"])
        print(f"mode={args.mode} shadow={shadow} action={action.tolist()}")
        if not shadow and args.enable_motion:
            hardware.send_action(action)
    except BaseException:
        hardware.stop()
        raise
    finally:
        hardware.stop()
        hardware.close()


if __name__ == "__main__":
    main()
