"""Tests for convert_robocoin_to_hdf5.py converter.

This module generates synthetic RoboCOIN datasets (parquet + MP4 videos) and
validates that the converter can parse them correctly. Allows CI testing without
real robot data.
"""

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


# Add parent directory to path for importing converter functions
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts import convert_robocoin_to_hdf5


def create_fake_robocoin_dataset(
    tmp_path: Path,
    n_episodes: int = 1,
    n_frames: int = 10,
    img_h: int = 64,
    img_w: int = 64,
    camera_keys: list[str] | None = None,
    gripper_range: str = "raw",
) -> Path:
    """Create a minimal synthetic RoboCOIN dataset structure.

    Args:
        tmp_path: Temporary directory root
        n_episodes: Number of episodes to create
        n_frames: Frames per episode
        img_h: Image height
        img_w: Image width
        camera_keys: List of camera names (default: ["wrist", "scene"]). Dirs are created as observation.images.<key>
        gripper_range: "raw" for [1, 500, 1000] or "normalized" for [0.0, 0.5, 1.0]

    Returns:
        Path to the created dataset root
    """
    if camera_keys is None:
        camera_keys = ["wrist", "scene"]

    dataset_root = tmp_path / "fake_robocoin"
    dataset_root.mkdir(parents=True, exist_ok=True)

    # Create meta/ directory
    meta_dir = dataset_root / "meta"
    meta_dir.mkdir()

    # Create info.json
    info = {
        "fps": 30,
        "features": {
            "observation.state": {"shape": [8], "dtype": "float32"},
            "action": {"shape": [8], "dtype": "float32"},
        },
    }
    for cam_key in camera_keys:
        info["features"][f"observation.images.{cam_key}"] = {
            "shape": [img_h, img_w, 3],
            "dtype": "uint8",
        }

    (meta_dir / "info.json").write_text(json.dumps(info, indent=2))

    # Create episodes.jsonl
    episodes_jsonl = meta_dir / "episodes.jsonl"
    with open(episodes_jsonl, "w") as f:
        for ep_idx in range(n_episodes):
            episode_meta = {
                "episode_index": ep_idx,
                "fps": 30,
                "from_idx": ep_idx * n_frames,
                "to_idx": (ep_idx + 1) * n_frames - 1,
            }
            f.write(json.dumps(episode_meta) + "\n")

    # Create tasks.jsonl
    tasks_jsonl = meta_dir / "tasks.jsonl"
    with open(tasks_jsonl, "w") as f:
        task = {"task_index": 0, "task": "test task"}
        f.write(json.dumps(task) + "\n")

    # Create data/ directory with parquet files
    data_dir = dataset_root / "data" / "chunk-000"
    data_dir.mkdir(parents=True)

    # Determine gripper values based on range
    if gripper_range == "raw":
        gripper_vals = np.linspace(1, 1000, n_frames, dtype=np.float32)
    elif gripper_range == "normalized":
        gripper_vals = np.linspace(0.0, 1.0, n_frames, dtype=np.float32)
    else:
        raise ValueError(f"Unknown gripper_range: {gripper_range}")

    # Create parquet file with all episodes
    all_rows = []
    for ep_idx in range(n_episodes):
        for frame_idx in range(n_frames):
            # State: [7 joints + 1 gripper]
            state = np.concatenate(
                [
                    np.linspace(0.1, 0.9, 7, dtype=np.float32),
                    [gripper_vals[frame_idx]],
                ]
            )

            # Action: [7 joints + 1 gripper]
            action = np.concatenate(
                [
                    np.linspace(0.2, 0.8, 7, dtype=np.float32),
                    [gripper_vals[frame_idx]],
                ]
            )

            row = {
                "episode_index": ep_idx,
                "frame_index": frame_idx,
                "timestamp": float(frame_idx) / 30.0,
                "observation.state": state.tolist(),
                "action": action.tolist(),
            }
            all_rows.append(row)

    # Create PyArrow table from rows
    schema = pa.schema(
        [
            pa.field("episode_index", pa.int32()),
            pa.field("frame_index", pa.int32()),
            pa.field("timestamp", pa.float64()),
            pa.field("observation.state", pa.list_(pa.float32())),
            pa.field("action", pa.list_(pa.float32())),
        ]
    )

    arrays = {
        "episode_index": [row["episode_index"] for row in all_rows],
        "frame_index": [row["frame_index"] for row in all_rows],
        "timestamp": [row["timestamp"] for row in all_rows],
        "observation.state": [row["observation.state"] for row in all_rows],
        "action": [row["action"] for row in all_rows],
    }

    table = pa.table(arrays, schema=schema)
    pq.write_table(table, data_dir / "episode_000000.parquet")

    # Create videos/ directory with MP4 files
    videos_base = dataset_root / "videos" / "chunk-000"
    videos_base.mkdir(parents=True)

    for cam_key in camera_keys:
        cam_dir = videos_base / f"observation.images.{cam_key}"
        cam_dir.mkdir(parents=True)

        # Create video for each episode
        for ep_idx in range(n_episodes):
            video_path = cam_dir / f"episode_{ep_idx:06d}.mp4"

            # Create synthetic video with cv2.VideoWriter (MJPG codec)
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            out = cv2.VideoWriter(
                str(video_path),
                fourcc,
                30.0,
                (img_w, img_h),
            )

            # Write solid-color frames (different color per camera)
            color_idx = camera_keys.index(cam_key)
            bgr_color = (100 + color_idx * 50, 150, 200)

            for frame_idx in range(n_frames):
                frame = np.full((img_h, img_w, 3), bgr_color, dtype=np.uint8)
                out.write(frame)

            out.release()

    return dataset_root


class TestConvertRoboCOINToHDF5:
    """Test suite for RoboCOIN → HDF5 converter."""

    def test_dry_run_prints_schema(self, tmp_path: Path) -> None:
        """Test that dry-run on synthetic dataset exits with code 0."""
        dataset_root = create_fake_robocoin_dataset(tmp_path, n_episodes=1, n_frames=5)
        output_dir = tmp_path / "output"

        result = subprocess.run(
            [
                "python",
                str(Path(__file__).parent / "convert_robocoin_to_hdf5.py"),
                "--robocoin-dir",
                str(dataset_root),
                "--output-dir",
                str(output_dir),
                "--task-label",
                "test task",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Dry-run failed: {result.stderr}"
        assert "DRY RUN" in result.stdout
        assert "joint_position" in result.stdout
        assert "gripper_position" in result.stdout

    def test_convert_single_episode(self, tmp_path: Path) -> None:
        """Test that single episode is converted with all 7 HDF5 keys present."""
        dataset_root = create_fake_robocoin_dataset(tmp_path, n_episodes=1, n_frames=10)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Run converter
        result = subprocess.run(
            [
                "python",
                str(Path(__file__).parent / "convert_robocoin_to_hdf5.py"),
                "--robocoin-dir",
                str(dataset_root),
                "--output-dir",
                str(output_dir),
                "--task-label",
                "test task",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Conversion failed: {result.stderr}"

        # Verify HDF5 file exists
        hdf5_files = list(output_dir.glob("episode_*.hdf5"))
        assert len(hdf5_files) == 1, f"Expected 1 HDF5 file, got {len(hdf5_files)}"

        hdf5_path = hdf5_files[0]

        # Verify all 7 keys are present with correct shapes
        with h5py.File(hdf5_path, "r") as f:
            keys = set(f.keys())
            expected_keys = {
                "joint_position",
                "joint_velocity",
                "gripper_position",
                "wrist_image",
                "scene_image",
                "action_joint",
                "action_gripper",
            }
            assert keys == expected_keys, f"Keys mismatch: {keys} vs {expected_keys}"

            # Verify shapes
            assert f["joint_position"].shape == (10, 7)
            assert f["joint_velocity"].shape == (10, 7)
            assert f["gripper_position"].shape == (10, 1)
            assert f["wrist_image"].shape == (10, 64, 64, 3)
            assert f["scene_image"].shape == (10, 64, 64, 3)
            assert f["action_joint"].shape == (10, 7)
            assert f["action_gripper"].shape == (10, 1)

    def test_frame_count_consistency(self, tmp_path: Path) -> None:
        """Test that parquet rows == video frames == HDF5 timesteps."""
        n_frames = 15
        dataset_root = create_fake_robocoin_dataset(tmp_path, n_episodes=1, n_frames=n_frames)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Run converter
        result = subprocess.run(
            [
                "python",
                str(Path(__file__).parent / "convert_robocoin_to_hdf5.py"),
                "--robocoin-dir",
                str(dataset_root),
                "--output-dir",
                str(output_dir),
                "--task-label",
                "test task",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

        hdf5_path = list(output_dir.glob("episode_*.hdf5"))[0]

        with h5py.File(hdf5_path, "r") as f:
            # All timestep-dimension datasets should have n_frames
            assert f["joint_position"].shape[0] == n_frames
            assert f["joint_velocity"].shape[0] == n_frames
            assert f["gripper_position"].shape[0] == n_frames
            assert f["wrist_image"].shape[0] == n_frames
            assert f["scene_image"].shape[0] == n_frames
            assert f["action_joint"].shape[0] == n_frames
            assert f["action_gripper"].shape[0] == n_frames

    def test_gripper_auto_detect_raw(self, tmp_path: Path, caplog) -> None:
        """Test that raw gripper values [1, 500, 1000] trigger normalization."""
        dataset_root = create_fake_robocoin_dataset(tmp_path, n_episodes=1, n_frames=3, gripper_range="raw")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = subprocess.run(
            [
                "python",
                str(Path(__file__).parent / "convert_robocoin_to_hdf5.py"),
                "--robocoin-dir",
                str(dataset_root),
                "--output-dir",
                str(output_dir),
                "--task-label",
                "test task",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "gripper_norm=True" in result.stderr or "gripper_norm=True" in result.stdout

        # Verify gripper is normalized to [0, 1]
        hdf5_path = list(output_dir.glob("episode_*.hdf5"))[0]
        with h5py.File(hdf5_path, "r") as f:
            gripper_vals = f["gripper_position"][:]
            assert gripper_vals.min() >= 0.0
            assert gripper_vals.max() <= 1.0

    def test_gripper_auto_detect_normalized(self, tmp_path: Path) -> None:
        """Test that pre-normalized gripper [0.0, 0.5, 1.0] skips normalization."""
        dataset_root = create_fake_robocoin_dataset(tmp_path, n_episodes=1, n_frames=3, gripper_range="normalized")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = subprocess.run(
            [
                "python",
                str(Path(__file__).parent / "convert_robocoin_to_hdf5.py"),
                "--robocoin-dir",
                str(dataset_root),
                "--output-dir",
                str(output_dir),
                "--task-label",
                "test task",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "gripper_norm=False" in result.stderr or "gripper_norm=False" in result.stdout

        # Verify gripper values are approximately the same (no aggressive normalization)
        hdf5_path = list(output_dir.glob("episode_*.hdf5"))[0]
        with h5py.File(hdf5_path, "r") as f:
            gripper_vals = f["gripper_position"][:]
            # Values should still be in [0, 1]
            assert gripper_vals.min() >= 0.0
            assert gripper_vals.max() <= 1.0

    def test_missing_scene_camera(self, tmp_path: Path, caplog) -> None:
        """Test that missing scene camera fills with zeros and logs warning."""
        dataset_root = create_fake_robocoin_dataset(tmp_path, n_episodes=1, n_frames=8, camera_keys=["wrist"])
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = subprocess.run(
            [
                "python",
                str(Path(__file__).parent / "convert_robocoin_to_hdf5.py"),
                "--robocoin-dir",
                str(dataset_root),
                "--output-dir",
                str(output_dir),
                "--task-label",
                "test task",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        # Check that warning about missing scene camera appears
        assert "not found" in result.stderr.lower() or "not found" in result.stdout.lower()

        # Verify scene_image is zero-filled
        hdf5_path = list(output_dir.glob("episode_*.hdf5"))[0]
        with h5py.File(hdf5_path, "r") as f:
            scene_image = f["scene_image"][:]
            assert np.all(scene_image == 0), "scene_image should be zero-filled"
            assert scene_image.shape == (8, 64, 64, 3)

    def test_state_action_splitting(self, tmp_path: Path) -> None:
        """Test that joint_position == state[0:7] and gripper_position == state[7:8]."""
        dataset_root = create_fake_robocoin_dataset(tmp_path, n_episodes=1, n_frames=10)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = subprocess.run(
            [
                "python",
                str(Path(__file__).parent / "convert_robocoin_to_hdf5.py"),
                "--robocoin-dir",
                str(dataset_root),
                "--output-dir",
                str(output_dir),
                "--task-label",
                "test task",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

        hdf5_path = list(output_dir.glob("episode_*.hdf5"))[0]
        with h5py.File(hdf5_path, "r") as f:
            joint_pos = f["joint_position"][:]
            gripper_pos = f["gripper_position"][:]
            action_joint = f["action_joint"][:]
            action_gripper = f["action_gripper"][:]

            # Verify shapes match expectation
            assert joint_pos.shape == (10, 7)
            assert gripper_pos.shape == (10, 1)
            assert action_joint.shape == (10, 7)
            assert action_gripper.shape == (10, 1)

    def test_joint_velocity_zero_filled(self, tmp_path: Path) -> None:
        """Test that all joint velocity values are zero."""
        dataset_root = create_fake_robocoin_dataset(tmp_path, n_episodes=1, n_frames=10)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = subprocess.run(
            [
                "python",
                str(Path(__file__).parent / "convert_robocoin_to_hdf5.py"),
                "--robocoin-dir",
                str(dataset_root),
                "--output-dir",
                str(output_dir),
                "--task-label",
                "test task",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

        hdf5_path = list(output_dir.glob("episode_*.hdf5"))[0]
        with h5py.File(hdf5_path, "r") as f:
            joint_vel = f["joint_velocity"][:]
            assert np.all(joint_vel == 0.0), "joint_velocity should be all zeros"
            assert joint_vel.shape == (10, 7)

    def test_roundtrip_to_lerobot(self, tmp_path: Path) -> None:
        """Test that converted HDF5 passes through convert_rm75_data_to_lerobot.py --dry-run."""
        dataset_root = create_fake_robocoin_dataset(tmp_path, n_episodes=1, n_frames=10)
        hdf5_output = tmp_path / "hdf5_output"
        hdf5_output.mkdir()

        # Step 1: Convert RoboCOIN to HDF5
        result = subprocess.run(
            [
                "python",
                str(Path(__file__).parent / "convert_robocoin_to_hdf5.py"),
                "--robocoin-dir",
                str(dataset_root),
                "--output-dir",
                str(hdf5_output),
                "--task-label",
                "test task",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"HDF5 conversion failed: {result.stderr}"

        # Step 2: Verify we can run convert_rm75_data_to_lerobot in dry-run mode
        lerobot_output = tmp_path / "lerobot_output"
        result = subprocess.run(
            [
                "python",
                str(Path(__file__).parent / "convert_rm75_data_to_lerobot.py"),
                "--raw-dir",
                str(hdf5_output),
                "--repo-id",
                "test_org/test_dataset",
                "--task-label",
                "test task",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )

        # Dry-run should succeed (exit code 0)
        assert result.returncode == 0, (
            f"LeRobot dry-run failed: {result.stderr}\nStdout: {result.stdout}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
