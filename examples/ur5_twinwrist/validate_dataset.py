# ruff: noqa: E402, RUF001, RUF002, RUF003
"""使用项目 YAML 限位的严格原始 HDF5 validator。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.ur5_twinwrist.config_loader import DEFAULT_CONFIG_DIR
from examples.ur5_twinwrist.config_loader import load_project_config

REQUIRED = (
    "observations/qpos",
    "action",
    "observations/images/front",
    "observations/images/side",
    "observations/images/top",
    "timestamps/control",
    "timestamps/front",
    "timestamps/side",
    "timestamps/top",
)

REQUIRED_ATTRIBUTES = (
    "task",
    "fps",
    "success",
    "git_commit",
    "camera_serials",
    "gripper_backend",
    "gripper_state_source",
    "action_space",
    "wrist_coordinate",
    "wrist_servo_zero_raw",
    "robot_config_hash",
)

DEFAULT_STATE_LIMITS = (
    np.asarray([-2 * np.pi] * 6 + [-4095.0, -4095.0, 0.0]),
    np.asarray([2 * np.pi] * 6 + [4095.0, 4095.0, 1.0]),
)
DEFAULT_ACTION_LIMITS = (
    np.asarray([-0.25] * 3 + [-0.60] * 3 + [-4095.0, -4095.0, 0.0]),
    np.asarray([0.25] * 3 + [0.60] * 3 + [4095.0, 4095.0, 1.0]),
)


def limits_from_project(project: dict) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """把 YAML 中的 UR/腕/夹爪边界转换为 9 维 validator 上下限。"""

    ur5 = project["safety"]["ur5"]
    wrist = project["safety"]["wrist"]
    wrist_pose = project["poses"]["wrist"]
    zero = np.asarray(wrist_pose["servo_zero_raw"], dtype=np.float64)
    wrist_min = np.asarray([wrist["j1_min_raw"], wrist["j2_min_raw"]], dtype=np.float64) - zero
    wrist_max = np.asarray([wrist["j1_max_raw"], wrist["j2_max_raw"]], dtype=np.float64) - zero
    joint_min = ur5.get("joint_min_rad")
    joint_max = ur5.get("joint_max_rad")
    if joint_min is None or joint_max is None:
        joint_min, joint_max = [-2 * np.pi] * 6, [2 * np.pi] * 6
    state = (
        np.asarray([*joint_min, *wrist_min, 0.0], dtype=np.float64),
        np.asarray([*joint_max, *wrist_max, 1.0], dtype=np.float64),
    )
    action = (
        np.asarray(
            [-float(ur5["max_linear_m_s"])] * 3
            + [-float(ur5["max_angular_rad_s"])] * 3
            + [*wrist_min, 0.0],
            dtype=np.float64,
        ),
        np.asarray(
            [float(ur5["max_linear_m_s"])] * 3
            + [float(ur5["max_angular_rad_s"])] * 3
            + [*wrist_max, 1.0],
            dtype=np.float64,
        ),
    )
    return state, action


def validate_episode(
    path: str | Path,
    max_camera_skew_ms=50.0,
    action_limits=None,
    *,
    state_limits=None,
    expected_image_shape: tuple[int, int, int] | None = None,
    max_linear_norm_m_s=0.25,
    max_angular_norm_rad_s=0.60,
    expected_wrist_servo_zero_raw: tuple[int, int] | None = None,
) -> list[str]:
    errors = []
    with h5py.File(path, "r") as f:
        missing_attrs = [key for key in REQUIRED_ATTRIBUTES if key not in f.attrs]
        if missing_attrs:
            errors.append(f"missing attributes: {missing_attrs}")
        if "success" in f.attrs and not bool(f.attrs["success"]):
            errors.append("episode success attribute is false")
        if f.attrs.get("wrist_coordinate") != "yaml_servo_zero_relative_raw":
            errors.append("wrist_coordinate is not yaml_servo_zero_relative_raw")
        if expected_wrist_servo_zero_raw is not None and "wrist_servo_zero_raw" in f.attrs:
            try:
                recorded_zero = tuple(int(value) for value in json.loads(str(f.attrs["wrist_servo_zero_raw"])))
            except (TypeError, ValueError, json.JSONDecodeError):
                errors.append("wrist_servo_zero_raw attribute is invalid")
            else:
                if recorded_zero != expected_wrist_servo_zero_raw:
                    errors.append(
                        f"wrist_servo_zero_raw {recorded_zero} does not match YAML {expected_wrist_servo_zero_raw}"
                    )
        missing = [key for key in REQUIRED if key not in f]
        if missing:
            return [f"missing datasets: {missing}"]
        lengths = {key: len(f[key]) for key in REQUIRED}
        if len(set(lengths.values())) != 1:
            errors.append(f"length mismatch: {lengths}")
        if not lengths["action"]:
            errors.append("episode is empty")
        for key in ("observations/qpos", "action"):
            value = np.asarray(f[key])
            if value.ndim != 2 or value.shape[1] != 9:
                errors.append(f"{key} shape {value.shape}, expected (N,9)")
            if value.dtype != np.float32:
                errors.append(f"{key} dtype {value.dtype}, expected float32")
            if not np.isfinite(value).all():
                errors.append(f"{key} contains NaN or Inf")
        control = np.asarray(f["timestamps/control"])
        if len(control) > 1 and np.any(np.diff(control) <= 0):
            errors.append("control timestamps are not strictly monotonic")
        # RealSense device clocks are independent. Cross-camera skew must use
        # the local synchronizer's monotonic host timestamps; keep the raw
        # device timestamps for duplicate-frame detection and diagnosis.
        aligned_group = "timestamps_host" if "timestamps_host" in f else "timestamps"
        camera_ts = np.stack(
            [np.asarray(f[f"{aligned_group}/{name}"]) for name in ("front", "side", "top")],
            axis=1,
        )
        if camera_ts.size and np.max(np.ptp(camera_ts, axis=1)) > max_camera_skew_ms * 1e6:
            errors.append("camera skew exceeds limit")
        for name in ("front", "side", "top"):
            images = np.asarray(f[f"observations/images/{name}"])
            if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8:
                errors.append(f"{name} image dtype/shape invalid")
            if expected_image_shape is not None and images.shape[1:] != expected_image_shape:
                errors.append(
                    f"{name} image shape {images.shape[1:]}, expected {expected_image_shape}"
                )
            if len(images) and np.any(np.all(images == 0, axis=(1, 2, 3))):
                errors.append(f"{name} contains black frame")
            ts = np.asarray(f[f"timestamps/{name}"])
            if len(ts) > 1 and np.any(np.diff(ts) == 0):
                errors.append(f"{name} contains repeated frame timestamp")
            if len(ts) > 1 and np.any(np.diff(ts) < 0):
                errors.append(f"{name} frame timestamps are not monotonic")
        state_lo, state_hi = (
            np.asarray(value, dtype=np.float64)
            for value in (DEFAULT_STATE_LIMITS if state_limits is None else state_limits)
        )
        action_lo, action_hi = (
            np.asarray(value, dtype=np.float64)
            for value in (DEFAULT_ACTION_LIMITS if action_limits is None else action_limits)
        )
        states = np.asarray(f["observations/qpos"])
        actions = np.asarray(f["action"])
        if state_lo.shape != (9,) or state_hi.shape != (9,):
            raise ValueError("state limits must be two 9D vectors")
        if action_lo.shape != (9,) or action_hi.shape != (9,):
            raise ValueError("action limits must be two 9D vectors")
        if np.any(states < state_lo) or np.any(states > state_hi):
            errors.append("state out of configured bounds")
        if np.any(actions < action_lo) or np.any(actions > action_hi):
            errors.append("action out of configured bounds")
        if len(actions) and np.any(np.linalg.norm(actions[:, :3], axis=1) > max_linear_norm_m_s):
            errors.append("TCP linear action norm exceeds configured bound")
        if len(actions) and np.any(np.linalg.norm(actions[:, 3:6], axis=1) > max_angular_norm_rad_s):
            errors.append("TCP angular action norm exceeds configured bound")
    return errors


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    p.add_argument("--max-camera-skew-ms", type=float)
    p.add_argument("--max-linear-norm-m-s", type=float)
    p.add_argument("--max-angular-norm-rad-s", type=float)
    p.add_argument("--state-min", type=float, nargs=9)
    p.add_argument("--state-max", type=float, nargs=9)
    p.add_argument("--action-min", type=float, nargs=9)
    p.add_argument("--action-max", type=float, nargs=9)
    a = p.parse_args()
    project = load_project_config(a.config_dir)
    project_state_limits, project_action_limits = limits_from_project(project)
    state_limits = (
        project_state_limits[0] if a.state_min is None else a.state_min,
        project_state_limits[1] if a.state_max is None else a.state_max,
    )
    action_limits = (
        project_action_limits[0] if a.action_min is None else a.action_min,
        project_action_limits[1] if a.action_max is None else a.action_max,
    )
    max_skew = (
        float(project["safety"]["timing"]["max_camera_skew_ms"])
        if a.max_camera_skew_ms is None
        else a.max_camera_skew_ms
    )
    linear_norm = (
        float(project["safety"]["ur5"]["max_linear_m_s"])
        if a.max_linear_norm_m_s is None
        else a.max_linear_norm_m_s
    )
    angular_norm = (
        float(project["safety"]["ur5"]["max_angular_rad_s"])
        if a.max_angular_norm_rad_s is None
        else a.max_angular_norm_rad_s
    )
    paths = sorted(Path(a.path).glob("episode_*.hdf5")) if Path(a.path).is_dir() else [Path(a.path)]
    if not paths or any(not path.is_file() for path in paths):
        raise SystemExit(f"没有找到成功 episode: {a.path}")
    failed = False
    for path in paths:
        errors = validate_episode(
            path,
            max_skew,
            action_limits,
            state_limits=state_limits,
            expected_image_shape=tuple(int(value) for value in project["collection"]["capture"]["image_shape"]),
            max_linear_norm_m_s=linear_norm,
            max_angular_norm_rad_s=angular_norm,
            expected_wrist_servo_zero_raw=tuple(
                int(value) for value in project["poses"]["wrist"]["servo_zero_raw"]
            ),
        )
        print(path, "OK" if not errors else "FAIL: " + "; ".join(errors))
        failed |= bool(errors)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
