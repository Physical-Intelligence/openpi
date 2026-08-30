from pathlib import Path

import h5py
import numpy as np

from examples.ur5_twinwrist.record_hdf5 import record_fake
from examples.ur5_twinwrist.validate_dataset import validate_episode


def test_fake_hdf5_atomic(tmp_path: Path):
    path = record_fake(tmp_path, 3)
    assert path.name == "episode_0000.hdf5"
    assert not list(tmp_path.glob("*.tmp.hdf5"))
    with h5py.File(path, "r") as f:
        assert f["action"].shape == (3, 9)


def test_validation_failures(tmp_path: Path):
    path = record_fake(tmp_path, 3)
    with h5py.File(path, "r+") as f:
        f["action"][1, 0] = np.nan
        timestamp_group = "timestamps_host" if "timestamps_host" in f else "timestamps"
        f[f"{timestamp_group}/top"][1] = f[f"{timestamp_group}/front"][1] + 100_000_000
    errors = validate_episode(path, max_camera_skew_ms=20)
    assert any("NaN" in e for e in errors)
    assert any("skew" in e for e in errors)


def test_validator_detects_length_repeat_black_and_bounds(tmp_path: Path):
    path = record_fake(tmp_path, 3)
    with h5py.File(path, "r+") as f:
        f["action"].resize(2, axis=0)
        f["action"][0, 8] = 2.0
        f["observations/qpos"][0, 8] = -1.0
        f["observations/images/side"][0] = 0
        f["timestamps/front"][1] = f["timestamps/front"][0]
    errors = validate_episode(path)
    assert any("length mismatch" in error for error in errors)
    assert any("repeated" in error for error in errors)
    assert any("black" in error for error in errors)
    assert any("state out" in error for error in errors)
    assert any("action out" in error for error in errors)
