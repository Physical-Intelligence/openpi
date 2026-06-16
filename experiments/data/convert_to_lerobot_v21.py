# convert_to_lerobot_v21.py: Converts processed single-arm LeRobot v3.0 datasets to v2.1 format.
# Uses the installed LeRobot API to create properly formatted datasets that work with OpenPI.

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# LeRobot v2.1 home directory
from lerobot.common.constants import HF_LEROBOT_HOME


def convert_dataset(src_dir: Path, repo_id: str, fps: int = 30, task_name: str | None = None):
    """Convert a v3.0 single-arm dataset to v2.1 format using LeRobot API."""
    print(f"\nConverting: {src_dir.name} -> {repo_id}")

    # Read source data
    with open(src_dir / "meta" / "info.json") as f:
        info = json.load(f)

    total_episodes = info["total_episodes"]
    action_dim = info["features"]["action"]["shape"][0]
    cam_keys = [k for k in info["features"] if "images" in k]

    print(f"  Episodes: {total_episodes}, Action dim: {action_dim}, Cameras: {cam_keys}")

    # Read all frames
    table = pq.read_table(src_dir / "data" / "chunk-000" / "file-000.parquet")
    df = table.to_pandas()

    # Determine task name
    if task_name is None:
        tasks_pq = src_dir / "meta" / "tasks.parquet"
        if tasks_pq.exists():
            tasks_df = pq.read_table(tasks_pq).to_pandas()
            task_name = tasks_df.index[0] if len(tasks_df) > 0 else "unknown task"
        else:
            task_name = "pick object"

    # Clean up existing output
    output_path = Path(HF_LEROBOT_HOME) / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    # Define features matching our single-arm data
    features = {
        "observation.images.cam_high": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.cam_right_wrist": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": info["features"]["observation.state"]["names"],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": info["features"]["action"]["names"],
        },
    }

    # Create the dataset using LeRobot API
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        use_videos=True,
    )

    # Decode video frames — one mp4 file per camera for individual datasets
    import av

    cam_all_frames = {}
    for cam_key in cam_keys:
        chunk_dir = src_dir / "videos" / cam_key / "chunk-000"
        if not chunk_dir.exists():
            chunk_dir = src_dir / "videos" / cam_key
        video_files = sorted(chunk_dir.glob("*.mp4"))

        all_frames = []
        for video_path in video_files:
            container = av.open(str(video_path))
            for frame in container.decode(video=0):
                all_frames.append(frame.to_ndarray(format="rgb24"))
            container.close()

        if all_frames:
            cam_all_frames[cam_key] = all_frames
            print(f"  Decoded {len(all_frames)} frames from {cam_key} ({len(video_files)} file(s))")

    for ep_idx in range(total_episodes):
        ep_frames = df[df["episode_index"] == ep_idx].sort_values("frame_index")

        if len(ep_frames) == 0:
            continue

        for i, (_, row) in enumerate(ep_frames.iterrows()):
            frame_data = {
                "observation.state": np.array(row["observation.state"], dtype=np.float32),
                "action": np.array(row["action"], dtype=np.float32),
                "task": task_name,
            }

            # Use global index to map to video frame
            vid_frame_idx = int(row["index"])
            for cam_key in cam_keys:
                if cam_key in cam_all_frames and vid_frame_idx < len(cam_all_frames[cam_key]):
                    frame_data[cam_key] = cam_all_frames[cam_key][vid_frame_idx]
                else:
                    frame_data[cam_key] = np.zeros((480, 640, 3), dtype=np.uint8)

            dataset.add_frame(frame_data)

        dataset.save_episode()
        print(f"  Episode {ep_idx}: {len(ep_frames)} frames")

    print(f"  Saved to: {output_path}")
    return output_path


def convert_and_merge(src_dirs: list[Path], repo_id: str, fps: int = 30, task_name: str | None = None):
    """Convert multiple individual v3.0 datasets to v2.1 and merge into one dataset.

    Each source is converted individually (avoiding multi-video merge issues),
    then all episodes are combined into a single v2.1 dataset.
    """
    print(f"\nConverting and merging {len(src_dirs)} datasets -> {repo_id}")

    # Read source info for all datasets
    all_info = []
    for src_dir in src_dirs:
        with open(src_dir / "meta" / "info.json") as f:
            all_info.append(json.load(f))

    action_dim = all_info[0]["features"]["action"]["shape"][0]
    cam_keys = [k for k in all_info[0]["features"] if "images" in k]

    # Determine task name from first dataset if not provided
    if task_name is None:
        tasks_pq = src_dirs[0] / "meta" / "tasks.parquet"
        if tasks_pq.exists():
            tasks_df = pq.read_table(tasks_pq).to_pandas()
            task_name = tasks_df.index[0] if len(tasks_df) > 0 else "unknown task"
        else:
            task_name = "pick object"

    # Clean up existing output
    output_path = Path(HF_LEROBOT_HOME) / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    # Define features
    features = {
        "observation.images.cam_high": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.cam_right_wrist": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": all_info[0]["features"]["observation.state"]["names"],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": all_info[0]["features"]["action"]["names"],
        },
    }

    # Create combined dataset
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        use_videos=True,
    )

    import av

    total_eps = 0
    total_frames = 0
    for src_dir, info in zip(src_dirs, all_info):
        print(f"\n  Processing: {src_dir.name}")
        n_eps = info["total_episodes"]

        # Read parquet data
        table = pq.read_table(src_dir / "data" / "chunk-000" / "file-000.parquet")
        df = table.to_pandas()

        # Decode video frames (one file per camera for individual datasets)
        cam_frames = {}
        for cam_key in cam_keys:
            chunk_dir = src_dir / "videos" / cam_key / "chunk-000"
            if not chunk_dir.exists():
                chunk_dir = src_dir / "videos" / cam_key
            video_files = sorted(chunk_dir.glob("*.mp4"))

            frames = []
            for vf in video_files:
                container = av.open(str(vf))
                for frame in container.decode(video=0):
                    frames.append(frame.to_ndarray(format="rgb24"))
                container.close()

            if frames:
                cam_frames[cam_key] = frames
                print(f"    {cam_key}: {len(frames)} frames")

        # Add episodes
        for ep_idx in range(n_eps):
            ep_df = df[df["episode_index"] == ep_idx].sort_values("frame_index")
            if len(ep_df) == 0:
                continue

            for _, row in ep_df.iterrows():
                frame_data = {
                    "observation.state": np.array(row["observation.state"], dtype=np.float32),
                    "action": np.array(row["action"], dtype=np.float32),
                    "task": task_name,
                }
                vid_idx = int(row["index"])
                for cam_key in cam_keys:
                    if cam_key in cam_frames and vid_idx < len(cam_frames[cam_key]):
                        frame_data[cam_key] = cam_frames[cam_key][vid_idx]
                    else:
                        frame_data[cam_key] = np.zeros((480, 640, 3), dtype=np.uint8)

                dataset.add_frame(frame_data)

            dataset.save_episode()
            total_frames += len(ep_df)

        total_eps += n_eps
        print(f"    Added {n_eps} episodes from {src_dir.name}")

    print(f"\n  Merged: {total_eps} episodes, {total_frames} frames -> {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert v3.0 single-arm datasets to LeRobot v2.1 format"
    )
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=None,
        help="Path to a single processed v3.0 dataset directory",
    )
    parser.add_argument(
        "--src-dirs",
        type=Path,
        nargs="+",
        default=None,
        help="Paths to multiple processed v3.0 datasets (will be merged)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Output repo ID (stored under HF_LEROBOT_HOME)",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--task-name", type=str, default=None,
                        help="Task description for the dataset (e.g. 'pick marker')")
    args = parser.parse_args()

    if args.src_dirs:
        convert_and_merge(args.src_dirs, args.repo_id, args.fps, args.task_name)
    elif args.src_dir:
        convert_dataset(args.src_dir, args.repo_id, args.fps, args.task_name)
    else:
        parser.error("Either --src-dir or --src-dirs is required")


if __name__ == "__main__":
    main()
