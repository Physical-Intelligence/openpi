"""Tests for low-memory ALOHA → LeRobot conversion helpers."""

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest

_CONVERTER_PATH = Path(__file__).with_name("convert_aloha_data_to_lerobot.py")
_SPEC = importlib.util.spec_from_file_location("convert_aloha_data_to_lerobot", _CONVERTER_PATH)
assert _SPEC and _SPEC.loader
converter = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(converter)


def _write_synthetic_episode(path: Path, num_frames: int = 4) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("/observations/qpos", data=np.ones((num_frames, 14), dtype=np.float32))
        f.create_dataset("/action", data=np.zeros((num_frames, 14), dtype=np.float32))
        images = f.create_group("/observations/images")
        for cam in converter.DEFAULT_CAMERAS:
            images.create_dataset(
                cam,
                data=np.zeros((num_frames, 3, 480, 640), dtype=np.uint8),
            )


def test_load_image_frame_reads_single_frame(tmp_path: Path):
    ep_path = tmp_path / "episode_0.hdf5"
    _write_synthetic_episode(ep_path, num_frames=3)

    with h5py.File(ep_path, "r") as ep:
        frame = converter.load_image_frame(ep, "cam_high", 1)

    assert frame.shape == (3, 480, 640)
    assert frame.dtype == np.uint8


def test_populate_episode_does_not_load_all_images_at_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ep_path = tmp_path / "episode_0.hdf5"
    _write_synthetic_episode(ep_path, num_frames=2)
    calls: list[tuple[str, int]] = []

    def spy_load_image_frame(ep, camera, frame_idx):
        calls.append((camera, frame_idx))
        return np.zeros((3, 480, 640), dtype=np.uint8)

    monkeypatch.setattr(converter, "load_image_frame", spy_load_image_frame)

    class FakeDataset:
        def __init__(self):
            self.frames: list[dict] = []

        def add_frame(self, frame):
            self.frames.append(frame)

        def save_episode(self, *, task: str):
            assert task == "test-task"

    dataset = FakeDataset()
    converter.populate_episode(dataset, ep_path, task="test-task", cameras=["cam_high", "cam_low"])

    assert len(dataset.frames) == 2
    assert len(calls) == 4
    assert ("cam_high", 0) in calls
    assert ("cam_low", 1) in calls
