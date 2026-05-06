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


def convert_dataset(src_dir: Path, repo_id: str, fps: int = 30):
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

    # Read tasks
    tasks_pq = src_dir / "meta" / "tasks.parquet"
    if tasks_pq.exists():
        tasks_df = pq.read_table(tasks_pq).to_pandas()
        task_name = tasks_df.index[0] if len(tasks_df) > 0 else "unknown task"
    else:
        task_name = "pick lipbalm"

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

    # Source video files (we'll decode and re-encode via LeRobot)
    # For efficiency with video data, we add frames episode by episode
    import av

    # Decode all video frames upfront per camera (datasets are small)
    cam_all_frames = {}
    for cam_key in cam_keys:
        video_path = src_dir / "videos" / cam_key / "chunk-000" / "file-000.mp4"
        if not video_path.exists():
            # Try merged dataset video naming
            for vf in (src_dir / "videos" / cam_key / "chunk-000").glob("*.mp4"):
                video_path = vf
                break

        if video_path.exists():
            container = av.open(str(video_path))
            frames = []
            for frame in container.decode(video=0):
                frames.append(frame.to_ndarray(format="rgb24"))
            container.close()
            cam_all_frames[cam_key] = frames
            print(f"  Decoded {len(frames)} frames from {cam_key}")

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


def main():
    parser = argparse.ArgumentParser(
        description="Convert v3.0 single-arm datasets to LeRobot v2.1 format"
    )
    parser.add_argument(
        "--src-dir",
        type=Path,
        required=True,
        help="Path to processed v3.0 dataset directory",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Output repo ID (stored under HF_LEROBOT_HOME)",
    )
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    convert_dataset(args.src_dir, args.repo_id, args.fps)


if __name__ == "__main__":
    main()
