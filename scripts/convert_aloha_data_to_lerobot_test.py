"""Exercise the ALOHA converter with real HDF5 files and the pinned LeRobot writer."""

import importlib.util
import json
from pathlib import Path
import sys

import h5py
from lerobot.common.datasets import lerobot_dataset
import numpy as np
import pytest

_CONVERTER_PATH = Path(__file__).resolve().parents[1] / "examples/aloha_real/convert_aloha_data_to_lerobot.py"
_SPEC = importlib.util.spec_from_file_location("aloha_converter_under_test", _CONVERTER_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
converter = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = converter
_SPEC.loader.exec_module(converter)

_CAMERAS = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")
_REPO_ID = "test/aloha"
_CONFIG = converter.DatasetConfig(image_writer_processes=0, image_writer_threads=0, video_backend="pyav")


@pytest.fixture
def dataset_home(tmp_path, monkeypatch):
    home = tmp_path / "datasets"
    monkeypatch.setattr(converter, "HF_LEROBOT_HOME", home)
    monkeypatch.setattr(lerobot_dataset, "HF_LEROBOT_HOME", home)
    return home


def _write_episode(path: Path, value: float = 0, *, compressed: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as episode:
        episode["observations/qpos"] = np.full((2, 14), value, dtype=np.float32)
        episode["action"] = np.full((2, 14), value + 1, dtype=np.float32)
        episode["observations/qvel"] = np.full((2, 14), value + 2, dtype=np.float32)
        episode["observations/effort"] = np.full((2, 14), value + 3, dtype=np.float32)
        for camera in _CAMERAS:
            if compressed:
                import cv2

                ok, encoded = cv2.imencode(".jpg", np.full((480, 640, 3), 64, dtype=np.uint8))
                assert ok
                episode[f"observations/images/{camera}"] = np.stack([encoded.ravel()] * 2)
            else:
                episode[f"observations/images/{camera}"] = np.full((2, 480, 640, 3), 64, dtype=np.uint8)


def _existing_output(home: Path):
    output = home / _REPO_ID
    (output / "data").mkdir(parents=True)
    (output / "data" / "old.parquet").write_bytes(b"previous dataset contents")
    (output / "info.json").write_text('{"keep": true}')
    return output


def _snapshot(path: Path):
    return {str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()}


@pytest.mark.parametrize(
    "case", ["missing", "empty", "not_directory", "invalid_hdf5", "out_of_range", "negative", "no_selection"]
)
def test_invalid_input_preserves_existing_output(dataset_home, tmp_path, case):
    output = _existing_output(dataset_home)
    original = _snapshot(output)
    raw = tmp_path / "raw"
    episodes = None
    if case == "empty":
        raw.mkdir()
    elif case == "not_directory":
        raw.write_text("not a directory")
    elif case == "invalid_hdf5":
        raw.mkdir()
        (raw / "episode_0.hdf5").write_text("not HDF5")
    elif case in {"out_of_range", "negative", "no_selection"}:
        _write_episode(raw / "episode_0.hdf5")
        episodes = {"out_of_range": [1], "negative": [-1], "no_selection": []}[case]

    with pytest.raises((ValueError, IndexError, OSError)):
        converter.port_aloha(raw, _REPO_ID, episodes=episodes, push_to_hub=False, dataset_config=_CONFIG)
    assert _snapshot(output) == original


def test_failed_download_preserves_existing_output(dataset_home, tmp_path, monkeypatch):
    output = _existing_output(dataset_home)
    original = _snapshot(output)

    def fail_download(**kwargs):
        raise OSError("download failed")

    monkeypatch.setattr(converter, "snapshot_download", fail_download)
    with pytest.raises(OSError, match="download failed"):
        converter.port_aloha(tmp_path / "missing", _REPO_ID, raw_repo_id="test/raw", push_to_hub=False)
    assert _snapshot(output) == original


@pytest.mark.parametrize("location", ["same", "nested", "symlink"])
def test_source_inside_output_is_preserved(dataset_home, tmp_path, location):
    output = _existing_output(dataset_home)
    raw = output if location == "same" else output / "raw"
    _write_episode(raw / "episode_0.hdf5")
    if location == "symlink":
        alias = tmp_path / "source_alias"
        alias.symlink_to(raw, target_is_directory=True)
        raw = alias
    original = _snapshot(output)
    with pytest.raises(ValueError, match="raw_dir"):
        converter.port_aloha(raw, _REPO_ID, push_to_hub=False, dataset_config=_CONFIG)
    assert _snapshot(output) == original


def test_downloaded_input_creates_new_dataset(dataset_home, tmp_path, monkeypatch):
    raw = tmp_path / "download"

    def download(*, repo_id, repo_type, local_dir):
        assert (repo_id, repo_type, local_dir) == ("test/raw", "dataset", raw)
        _write_episode(local_dir / "episode_0.hdf5", value=4)

    monkeypatch.setattr(converter, "snapshot_download", download)
    converter.port_aloha(raw, _REPO_ID, raw_repo_id="test/raw", push_to_hub=False, dataset_config=_CONFIG)
    dataset = lerobot_dataset.LeRobotDataset(_REPO_ID, root=dataset_home / _REPO_ID, video_backend="pyav")
    assert len(dataset) == 2
    assert dataset[0]["task"] == "DEBUG"
    np.testing.assert_array_equal(dataset[0]["observation.state"].numpy(), np.full(14, 4))


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("mode", ["image", "video"])
def test_real_conversion_and_overwrite(dataset_home, tmp_path, compressed, mode):
    output = _existing_output(dataset_home)
    raw = tmp_path / "raw"
    _write_episode(raw / "episode_0.hdf5", value=3, compressed=compressed)
    _write_episode(raw / "episode_1.hdf5", value=7, compressed=compressed)

    converter.port_aloha(
        raw,
        _REPO_ID,
        task="pick up the can",
        episodes=[1, 0],
        push_to_hub=False,
        mode=mode,
        dataset_config=_CONFIG,
    )

    assert not (output / "data" / "old.parquet").exists()
    info = json.loads((output / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 4
    assert info["fps"] == 50
    dataset = lerobot_dataset.LeRobotDataset(_REPO_ID, root=output, video_backend="pyav")
    for index, value in enumerate([7, 7, 3, 3]):
        row = dataset[index]
        np.testing.assert_array_equal(row["observation.state"].numpy(), np.full(14, value))
        np.testing.assert_array_equal(row["action"].numpy(), np.full(14, value + 1))
        np.testing.assert_array_equal(row["observation.velocity"].numpy(), np.full(14, value + 2))
        np.testing.assert_array_equal(row["observation.effort"].numpy(), np.full(14, value + 3))
        assert row["task"] == "pick up the can"
        assert row["episode_index"].item() == index // 2
        assert row["timestamp"].item() == pytest.approx((index % 2) / 50)
        for camera in _CAMERAS:
            image = row[f"observation.images.{camera}"]
            assert tuple(image.shape) == (3, 480, 640)
            np.testing.assert_allclose(image.numpy(), 64 / 255, atol=2 / 255)
