from pathlib import Path

import h5py
import numpy as np
import pytest

from examples.ur5_twinwrist.config_loader import load_project_config
from examples.ur5_twinwrist.episode_recorder import EpisodeRecorder
from examples.ur5_twinwrist.robot_runtime import FakeHardware


def _observation(hardware: FakeHardware, timestamp_ns: int, sequence: int) -> dict:
    result = hardware.get_observation()
    result["timestamp_ns"] = timestamp_ns
    result["camera_timestamps_ns"] = dict.fromkeys(("front", "side", "top"), timestamp_ns)
    result["camera_host_timestamps_ns"] = dict.fromkeys(("front", "side", "top"), timestamp_ns)
    result["camera_sequences"] = dict.fromkeys(("front", "side", "top"), sequence)
    return result


def test_episode_recorder_rate_limits_and_atomically_saves(tmp_path: Path) -> None:
    hardware = FakeHardware()
    hardware.connect()
    with EpisodeRecorder(tmp_path, load_project_config()) as recorder:
        first = _observation(hardware, 1_000_000_000, 1)
        temporary = recorder.start(first)
        assert temporary.name == "episode_0000.tmp.hdf5"
        assert recorder.append_if_due(first, np.zeros(9))
        assert not recorder.append_if_due(_observation(hardware, 1_050_000_000, 2), np.zeros(9))
        assert recorder.append_if_due(_observation(hardware, 1_100_000_000, 3), np.zeros(9))
        path = recorder.save()
    assert path.name == "episode_0000.hdf5"
    assert not temporary.exists()
    with h5py.File(path, "r") as episode:
        assert episode["action"].shape == (2, 9)
        assert bool(episode.attrs["success"])
        assert (
            episode.attrs["action_space"]
            == "tcp_speedL_6+wrist_j1_j2_yaml_zero_relative_raw_2+gripper_absolute_1"
        )
        assert episode.attrs["wrist_coordinate"] == "yaml_servo_zero_relative_raw"


def test_repeated_camera_rejects_without_deleting_data(tmp_path: Path) -> None:
    hardware = FakeHardware()
    hardware.connect()
    with EpisodeRecorder(tmp_path, load_project_config()) as recorder:
        recorder.start(_observation(hardware, 1_000_000_000, 1))
        recorder.append_if_due(_observation(hardware, 1_000_000_000, 1), np.zeros(9))
        with pytest.raises(ValueError, match="camera frame repeated"):
            recorder.append_if_due(_observation(hardware, 1_100_000_000, 1), np.zeros(9))
        assert not recorder.active
    rejected = tmp_path / "rejected/episode_0000.hdf5"
    assert rejected.is_file()
    with h5py.File(rejected, "r") as episode:
        assert not bool(episode.attrs["success"])


def test_writer_lock_prevents_two_collectors(tmp_path: Path) -> None:
    first = EpisodeRecorder(tmp_path, load_project_config())
    try:
        with pytest.raises(RuntimeError, match="另一个录制器"):
            EpisodeRecorder(tmp_path, load_project_config())
    finally:
        first.close()


def test_episode_recorder_rejects_image_shape_not_matching_yaml(tmp_path: Path) -> None:
    hardware = FakeHardware(shape=(48, 64, 3))
    hardware.connect()
    with (
        EpisodeRecorder(tmp_path, load_project_config()) as recorder,
        pytest.raises(ValueError, match="collection.yaml"),
    ):
        recorder.start(_observation(hardware, 1_000_000_000, 1))
    assert not list(tmp_path.glob("episode_*.tmp.hdf5"))
