"""Convert RoboCOIN LeRobot v2.1 datasets to RM75 HDF5 format.

This script reads a RoboCOIN dataset (parquet + MP4 videos) and writes HDF5
per-episode files matching the RM75 schema. The output is compatible with
the existing `convert_rm75_data_to_lerobot.py` converter for the full pipeline.

RoboCOIN Structure (LeRobot v2.1 format):
    meta/info.json                          → FPS, camera keys, state shape
    meta/episodes.jsonl                     → Episode metadata
    meta/tasks.jsonl                        → Task descriptions
    data/chunk-*/episode_NNNNNN.parquet     → State/action arrays
    videos/chunk-*/observation.images.<cam>/episode_NNNNNN.mp4  → Video frames

Output HDF5 Schema (per episode):
    joint_position    (N, 7)          float32
    joint_velocity    (N, 7)          float32
    gripper_position  (N, 1)          float32
    wrist_image       (N, H, W, 3)    uint8
    scene_image       (N, H, W, 3)    uint8
    action_joint      (N, 7)          float32
    action_gripper    (N, 1)          float32

Usage:
    # Dry-run: inspect schema without writing
    uv run scripts/convert_robocoin_to_hdf5.py \\
        --robocoin-dir /path/to/robocoin/dataset \\
        --output-dir /path/to/hdf5/output \\
        --task-label "wipe solar panel" \\
        --dry-run

    # Full conversion
    uv run scripts/convert_robocoin_to_hdf5.py \\
        --robocoin-dir /path/to/robocoin/dataset \\
        --output-dir /path/to/hdf5/output \\
        --task-label "wipe solar panel"

    # Convert with action shift (if RoboCOIN action[t] = observation[t+1])
    uv run scripts/convert_robocoin_to_hdf5.py \\
        --robocoin-dir /path/to/robocoin/dataset \\
        --output-dir /path/to/hdf5/output \\
        --task-label "wipe solar panel" \\
        --action-shift 1
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

import cv2
import h5py
import numpy as np
import pyarrow.parquet as pq

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RoboCOIN LeRobot v2.1 datasets to RM75 HDF5 format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--robocoin-dir",
        required=True,
        type=Path,
        help="Root directory of RoboCOIN LeRobot v2.1 dataset.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where HDF5 episode files will be written.",
    )
    parser.add_argument(
        "--task-label",
        required=True,
        type=str,
        help="Language instruction describing the task.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Limit number of episodes to convert (for testing). None = convert all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print schema without writing any files.",
    )
    parser.add_argument(
        "--wrist-camera",
        type=str,
        default="wrist",
        help="Camera key name for wrist camera in RoboCOIN (default: 'wrist').",
    )
    parser.add_argument(
        "--scene-camera",
        type=str,
        default="scene",
        help="Camera key name for scene camera in RoboCOIN (default: 'scene').",
    )
    parser.add_argument(
        "--action-shift",
        type=int,
        default=0,
        help=(
            "Shift actions by N steps. Use 1 if RoboCOIN action[t] corresponds to "
            "observation[t+1]. Default: 0 (no shift)."
        ),
    )
    return parser.parse_args()


def discover_robocoin_dataset(dataset_dir: Path) -> dict[str, Any]:
    """Validate RoboCOIN dataset structure and read meta/info.json.

    Returns:
        Dictionary with keys: fps, features, episode_count, parquet_paths, video_dirs
    """
    # Validate structure
    meta_dir = dataset_dir / "meta"
    if not meta_dir.exists():
        raise FileNotFoundError(f"meta/ directory not found in {dataset_dir}")

    info_path = meta_dir / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"meta/info.json not found in {dataset_dir}")

    # Read info.json
    info = json.loads(info_path.read_text())
    fps = info.get("fps", 30)
    features = info.get("features", {})

    # Validate state shape
    state_shape = features.get("observation.state", {}).get("shape", [])
    if state_shape != [8]:
        raise ValueError(
            f"Expected observation.state shape [8], got {state_shape}. "
            "RoboCOIN state must be 8D (7 joints + 1 gripper)."
        )

    # Discover data chunks and video directories
    data_dir = dataset_dir / "data"
    videos_dir = dataset_dir / "videos"

    if not data_dir.exists():
        raise FileNotFoundError(f"data/ directory not found in {dataset_dir}")
    if not videos_dir.exists():
        raise FileNotFoundError(f"videos/ directory not found in {dataset_dir}")

    # Find all parquet chunks
    parquet_paths = sorted(data_dir.glob("chunk-*/episode_*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    # Find video directories
    video_dirs = sorted(videos_dir.glob("chunk-*/observation.images.*"))

    return {
        "fps": fps,
        "features": features,
        "parquet_paths": parquet_paths,
        "video_dirs": video_dirs,
        "info": info,
    }


def decode_video_frames(video_path: Path) -> np.ndarray:
    """Decode MP4 video to numpy array.

    Args:
        video_path: Path to MP4 file

    Returns:
        Array of shape (N, H, W, 3) with dtype uint8, RGB format
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR (OpenCV default) to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb_frame)

    cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")

    return np.stack(frames, axis=0)  # (N, H, W, 3) uint8


def detect_gripper_normalization(gripper_values: np.ndarray) -> bool:
    """Auto-detect if gripper values are raw SDK values or pre-normalized.

    Returns:
        True if values appear to be raw (1-1000) and need normalization
    """
    max_val = float(np.max(gripper_values))
    return max_val > 1.5


def normalize_gripper(raw: np.ndarray, needs_norm: bool) -> np.ndarray:
    """Normalize gripper values to [0, 1] if needed.

    Args:
        raw: Raw gripper values
        needs_norm: If True, apply (raw - 1) / 999 transform

    Returns:
        Normalized gripper values as float32
    """
    if needs_norm:
        return ((raw - 1.0) / 999.0).astype(np.float32)
    return raw.astype(np.float32)


def apply_action_shift(actions: np.ndarray, action_shift: int) -> np.ndarray:
    """Apply temporal shift to actions.

    Args:
        actions: Action array (N, D)
        action_shift: Number of steps to shift (positive = backward, negative = forward)

    Returns:
        Shifted action array (N, D)
    """
    if action_shift == 0:
        return actions

    if action_shift > 0:
        # Shift backward: action[t] → action[t - shift]
        # Pad end with last action
        shifted = np.concatenate(
            [actions[action_shift:], np.tile(actions[-1:], (action_shift, 1))],
            axis=0,
        )
    else:
        # Shift forward: action[t] → action[t - shift]
        # Pad start with first action
        shifted = np.concatenate(
            [np.tile(actions[:1], (-action_shift, 1)), actions[:action_shift]],
            axis=0,
        )

    return shifted


def convert_episode(
    episode_idx: int,
    parquet_path: Path,
    video_dirs: dict[str, Path],
    output_dir: Path,
    wrist_camera_key: str,
    scene_camera_key: str,
    action_shift: int = 0,
) -> int:
    """Convert a single RoboCOIN episode to HDF5.

    Args:
        episode_idx: Episode index
        parquet_path: Path to parquet file with state/action data
        video_dirs: Mapping of camera key → video directory
        output_dir: Where to write HDF5 file
        wrist_camera_key: Camera key name for wrist camera
        scene_camera_key: Camera key name for scene camera
        action_shift: Temporal shift to apply to actions

    Returns:
        Number of frames in the episode
    """
    # Read parquet
    table = pq.read_table(parquet_path)
    df = table.to_pandas()

    # Filter by episode index
    episode_data = df[df["episode_index"] == episode_idx]
    if len(episode_data) == 0:
        raise ValueError(f"No data found for episode {episode_idx} in {parquet_path}")

    # Extract state and action
    states = np.stack(episode_data["observation.state"].values)  # (N, 8)
    actions = np.stack(episode_data["action"].values)  # (N, 8)

    assert states.shape[1] == 8, f"Expected state dim 8, got {states.shape[1]}"
    assert actions.shape[1] == 8, f"Expected action dim 8, got {actions.shape[1]}"

    n_frames = states.shape[0]

    # Detect gripper normalization (use state gripper values)
    gripper_needs_norm = detect_gripper_normalization(states[:, 7:8])

    # Extract and decode videos
    # Construct video path: videos/chunk-NNN/observation.images.<cam>/episode_NNNNNN.mp4
    episode_name = parquet_path.stem  # "episode_000000", etc.
    chunk_name = parquet_path.parent.name  # "chunk-000", etc.

    # Find wrist camera video
    wrist_video_dir = None
    for vdir in video_dirs:
        if wrist_camera_key in vdir.name and chunk_name in vdir.parent.name:
            wrist_video_dir = vdir
            break

    if wrist_video_dir is None:
        # Try more lenient matching
        for vdir in video_dirs:
            if wrist_camera_key in vdir.name:
                wrist_video_dir = vdir
                break

    if wrist_video_dir is None:
        raise FileNotFoundError(
            f"Could not find wrist camera video directory (looking for key containing '{wrist_camera_key}')"
        )

    wrist_video_path = wrist_video_dir / f"{episode_name}.mp4"
    if not wrist_video_path.exists():
        raise FileNotFoundError(f"Wrist video not found: {wrist_video_path}")

    wrist_frames = decode_video_frames(wrist_video_path)
    assert wrist_frames.shape[0] == n_frames, f"Wrist frames {wrist_frames.shape[0]} != episode length {n_frames}"

    # Find scene camera video (may not exist)
    scene_video_dir = None
    for vdir in video_dirs:
        if scene_camera_key in vdir.name:
            scene_video_dir = vdir
            break

    if scene_video_dir is None:
        logger.warning(
            f"Scene camera (key '{scene_camera_key}') not found for episode {episode_idx}. "
            f"Zero-filling scene_image with shape {wrist_frames.shape[0:1] + wrist_frames.shape[1:]}"
        )
        scene_frames = np.zeros_like(wrist_frames)
    else:
        scene_video_path = scene_video_dir / f"{episode_name}.mp4"
        if not scene_video_path.exists():
            logger.warning(f"Scene video not found: {scene_video_path}. Zero-filling scene_image.")
            scene_frames = np.zeros_like(wrist_frames)
        else:
            scene_frames = decode_video_frames(scene_video_path)
            assert scene_frames.shape[0] == n_frames, (
                f"Scene frames {scene_frames.shape[0]} != episode length {n_frames}"
            )

    # Apply action shift if needed
    if action_shift != 0:
        actions = apply_action_shift(actions, action_shift)

    # Write HDF5
    output_path = output_dir / f"episode_{episode_idx:06d}.hdf5"
    with h5py.File(output_path, "w") as f:
        f.create_dataset("joint_position", data=states[:, 0:7].astype(np.float32))
        f.create_dataset("joint_velocity", data=np.zeros((n_frames, 7), dtype=np.float32))
        f.create_dataset(
            "gripper_position",
            data=normalize_gripper(states[:, 7:8], gripper_needs_norm),
        )
        f.create_dataset("wrist_image", data=wrist_frames)
        f.create_dataset("scene_image", data=scene_frames)
        f.create_dataset("action_joint", data=actions[:, 0:7].astype(np.float32))
        f.create_dataset(
            "action_gripper",
            data=normalize_gripper(actions[:, 7:8], gripper_needs_norm),
        )

    logger.info(
        f"Wrote episode {episode_idx:06d}: {n_frames} frames, "
        f"gripper_norm={gripper_needs_norm}, "
        f"frames={wrist_frames.shape}, "
        f"→ {output_path}"
    )

    return n_frames


def print_dry_run_schema(dataset_info: dict[str, Any]) -> None:
    """Print dataset schema without writing files."""
    print("\n=== DRY RUN: RoboCOIN Dataset Schema Inspection ===\n")

    info = dataset_info["info"]
    features = dataset_info["features"]
    parquet_paths = dataset_info["parquet_paths"]

    print(f"Dataset root: {dataset_info.get('dataset_dir', 'N/A')}")
    print(f"FPS: {dataset_info['fps']}")
    print()

    print("Discovered Features:")
    for key, spec in features.items():
        shape = spec.get("shape", "unknown")
        dtype = spec.get("dtype", "unknown")
        print(f"  {key:<40} shape={shape} dtype={dtype}")

    print()
    print("Data Structure:")
    print(f"  Parquet files: {len(parquet_paths)}")
    if parquet_paths:
        # Sample first parquet to count episodes
        table = pq.read_table(parquet_paths[0])
        df = table.to_pandas()
        unique_episodes = df["episode_index"].nunique()
        print(f"  Episodes in first chunk: {unique_episodes}")

    print()
    print("Output HDF5 Schema (per episode):")
    print(f"  {'Key':<30} {'Shape':<20} {'Dtype'}")
    print(f"  {'-' * 30} {'-' * 20} {'-' * 10}")
    print(f"  {'joint_position':<30} {'(N, 7)':<20} {'float32'}")
    print(f"  {'joint_velocity':<30} {'(N, 7)':<20} {'float32'}")
    print(f"  {'gripper_position':<30} {'(N, 1)':<20} {'float32'}")
    print(f"  {'wrist_image':<30} {'(N, H, W, 3)':<20} {'uint8'}")
    print(f"  {'scene_image':<30} {'(N, H, W, 3)':<20} {'uint8'}")
    print(f"  {'action_joint':<30} {'(N, 7)':<20} {'float32'}")
    print(f"  {'action_gripper':<30} {'(N, 1)':<20} {'float32'}")

    print()
    print("[DRY RUN] No files written. Remove --dry-run to convert.")
    print()


def main() -> None:
    args = parse_args()

    # Validate inputs
    if not args.robocoin_dir.exists():
        print(f"ERROR: Dataset directory not found: {args.robocoin_dir}", file=sys.stderr)
        sys.exit(1)

    # Discover dataset
    try:
        dataset_info = discover_robocoin_dataset(args.robocoin_dir)
        dataset_info["dataset_dir"] = args.robocoin_dir
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print_dry_run_schema(dataset_info)
        return

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Discover unique episodes across all parquet files
    all_episodes = set()
    for parquet_path in dataset_info["parquet_paths"]:
        table = pq.read_table(parquet_path)
        df = table.to_pandas()
        episodes_in_file = sorted(df["episode_index"].unique())
        all_episodes.update(episodes_in_file)

    all_episodes = sorted(list(all_episodes))

    if args.max_episodes is not None:
        all_episodes = all_episodes[: args.max_episodes]

    print(f"Found {len(all_episodes)} episode(s). Converting to HDF5 at {args.output_dir}")

    # Find which parquet file contains each episode
    episode_to_parquet = {}
    for parquet_path in dataset_info["parquet_paths"]:
        table = pq.read_table(parquet_path)
        df = table.to_pandas()
        for ep_idx in df["episode_index"].unique():
            episode_to_parquet[ep_idx] = parquet_path

    # Convert episodes
    total_frames = 0
    for episode_idx in all_episodes:
        parquet_path = episode_to_parquet[episode_idx]

        # Build video_dirs mapping
        video_dirs = {}
        for vdir in dataset_info["video_dirs"]:
            # Extract camera key from path like "observation.images.wrist"
            parts = vdir.name.split(".")
            if len(parts) >= 3:
                camera_key = parts[-1]
                video_dirs[camera_key] = vdir

        try:
            n_frames = convert_episode(
                episode_idx,
                parquet_path,
                video_dirs,
                args.output_dir,
                args.wrist_camera,
                args.scene_camera,
                args.action_shift,
            )
            total_frames += n_frames
        except Exception as e:
            print(
                f"ERROR converting episode {episode_idx}: {e}",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"\nSuccessfully converted {len(all_episodes)} episode(s) ({total_frames} total frames) to {args.output_dir}")


if __name__ == "__main__":
    main()
