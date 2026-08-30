"""Strict raw-episode validator."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

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

DEFAULT_STATE_LIMITS = (
    np.asarray([-2 * np.pi] * 6 + [-np.pi, -np.pi, 0.0]),
    np.asarray([2 * np.pi] * 6 + [np.pi, np.pi, 1.0]),
)
DEFAULT_ACTION_LIMITS = (
    np.asarray([-0.25] * 3 + [-0.60] * 3 + [-np.pi, -np.pi, 0.0]),
    np.asarray([0.25] * 3 + [0.60] * 3 + [np.pi, np.pi, 1.0]),
)


def validate_episode(
    path: str | Path,
    max_camera_skew_ms=50.0,
    action_limits=None,
    *,
    state_limits=None,
    max_linear_norm_m_s=0.25,
    max_angular_norm_rad_s=0.60,
) -> list[str]:
    errors = []
    with h5py.File(path, "r") as f:
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
        # RealSense device clocks are independent.  Cross-camera skew must use
        # the legacy synchronizer's monotonic host timestamps; keep the raw
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
    p.add_argument("--max-camera-skew-ms", type=float, default=50)
    p.add_argument("--max-linear-norm-m-s", type=float, default=0.25)
    p.add_argument("--max-angular-norm-rad-s", type=float, default=0.60)
    p.add_argument("--state-min", type=float, nargs=9, default=DEFAULT_STATE_LIMITS[0])
    p.add_argument("--state-max", type=float, nargs=9, default=DEFAULT_STATE_LIMITS[1])
    p.add_argument("--action-min", type=float, nargs=9, default=DEFAULT_ACTION_LIMITS[0])
    p.add_argument("--action-max", type=float, nargs=9, default=DEFAULT_ACTION_LIMITS[1])
    a = p.parse_args()
    paths = sorted(Path(a.path).glob("episode_*.hdf5")) if Path(a.path).is_dir() else [Path(a.path)]
    failed = False
    for path in paths:
        errors = validate_episode(
            path,
            a.max_camera_skew_ms,
            (a.action_min, a.action_max),
            state_limits=(a.state_min, a.state_max),
            max_linear_norm_m_s=a.max_linear_norm_m_s,
            max_angular_norm_rad_s=a.max_angular_norm_rad_s,
        )
        print(path, "OK" if not errors else "FAIL: " + "; ".join(errors))
        failed |= bool(errors)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
