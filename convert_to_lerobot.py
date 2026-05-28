"""Convert a v3.0 bimanual-YAM LeRobotDataset into the v2.1 format that
openpi's pinned lerobot can load.

The source datasets are published in lerobot v3.0 format, which the lerobot
revision pinned by openpi can't parse. We work around that by spawning a
subprocess in a separate uv env (with a newer lerobot) that dumps one
episode at a time to a temp dir, then write those frames into a fresh v2.1
LeRobotDataset here. Peak scratch disk is one episode's worth of JPEGs.

Usage:
    uv run convert_to_lerobot.py \\
        --src_repo_id allenai/24112025-yam-01 \\
        --dst_repo_id leokswang/YAM_lerobot_format

    # Smoke test with just 2 episodes:
    uv run convert_to_lerobot.py --src_repo_id ... --dst_repo_id ... --num_episodes 2

    # Push when done:
    uv run convert_to_lerobot.py --src_repo_id ... --dst_repo_id ... --push_to_hub

A second invocation with the same --dst_repo_id resumes from the next
unfinished episode (an existing output dir is reopened, not wiped).
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import tyro
from huggingface_hub import hf_hub_download
from lerobot.common.datasets.lerobot_dataset import (
    HF_LEROBOT_HOME,
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from lerobot.common.datasets.video_utils import get_safe_default_codec

DUMPER = Path(__file__).parent / "scripts" / "_dump_yam_episode.py"


def fetch_src_num_episodes(src_repo_id: str) -> int:
    info_path = hf_hub_download(
        repo_id=src_repo_id, filename="meta/info.json", repo_type="dataset"
    )
    return int(json.loads(Path(info_path).read_text())["total_episodes"])


def dump_episode(src_repo_id: str, episode_index: int, out_dir: Path) -> None:
    """Shell out to the newer-lerobot env to decode one v3.0 episode to disk."""
    subprocess.run(
        [
            "uv", "run", "--no-project", "--python", "3.11",
            "--with", "lerobot>=0.4.0",
            "python", str(DUMPER),
            "--src_repo_id", src_repo_id,
            "--episode_index", str(episode_index),
            "--out_dir", str(out_dir),
        ],
        check=True,
    )


def read_jpeg_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"failed to read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def open_for_resume(dst_repo_id: str) -> LeRobotDataset:
    """Reopen an in-progress on-disk dataset for appending more episodes.

    LeRobotDataset.create() refuses to run when the output dir exists; the
    regular constructor is read-only (no image_writer / episode_buffer).
    Hot-wire one that mirrors create()'s state but reads existing meta.
    """
    obj = LeRobotDataset.__new__(LeRobotDataset)
    obj.meta = LeRobotDatasetMetadata(repo_id=dst_repo_id)
    obj.repo_id = obj.meta.repo_id
    obj.root = obj.meta.root
    obj.revision = None
    obj.tolerance_s = 1e-4
    obj.image_writer = None
    obj.start_image_writer(num_processes=5, num_threads=10)
    obj.episode_buffer = obj.create_episode_buffer()
    obj.episodes = None
    obj.hf_dataset = obj.create_hf_dataset()
    obj.image_transforms = None
    obj.delta_timestamps = None
    obj.delta_indices = None
    obj.episode_data_index = None
    obj.video_backend = get_safe_default_codec()
    return obj


def main(
    *,
    src_repo_id: str,
    dst_repo_id: str,
    num_episodes: int | None = None,
    start_episode: int | None = None,
    scratch_dir: Path | None = None,
    push_to_hub: bool = False,
    robot_type: str = "yam",
):
    src_num_episodes = fetch_src_num_episodes(src_repo_id)
    if num_episodes is None:
        num_episodes = src_num_episodes
    output_path = HF_LEROBOT_HOME / dst_repo_id

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    image_feature = {
        "dtype": "image",
        "shape": (360, 640, 3),
        "names": ["height", "width", "channel"],
    }

    if output_path.exists() and (output_path / "meta" / "info.json").exists():
        dataset = open_for_resume(dst_repo_id)
        resume_from = dataset.meta.total_episodes
        print(
            f"[resume] found existing dataset at {output_path} with {resume_from} episodes",
            flush=True,
        )
        if start_episode is None:
            start_episode = resume_from
        elif start_episode != resume_from:
            print(
                f"[resume] WARNING: --start_episode={start_episode} but on-disk has {resume_from} episodes",
                flush=True,
            )
    else:
        if output_path.exists():
            shutil.rmtree(output_path)
        if start_episode is None:
            start_episode = 0
        dataset = LeRobotDataset.create(
            repo_id=dst_repo_id,
            robot_type=robot_type,
            fps=30,
            features={
                "observation.images.top": image_feature,
                "observation.images.left": image_feature,
                "observation.images.right": image_feature,
                "observation.state": {
                    "dtype": "float32",
                    "shape": (14,),
                    "names": ["state"],
                },
                "action": {
                    "dtype": "float32",
                    "shape": (14,),
                    "names": ["action"],
                },
            },
            image_writer_threads=10,
            image_writer_processes=5,
        )

    end_episode = min(start_episode + num_episodes, src_num_episodes)

    # Per-episode subprocess dump -> read back -> add_frame -> save_episode -> cleanup
    scratch_root = Path(tempfile.mkdtemp(dir=scratch_dir, prefix="yam_dump_"))
    try:
        for ep in range(start_episode, end_episode):
            ep_dir = scratch_root / f"ep_{ep:05d}"
            print(f"[ep {ep}] dumping v3.0 frames...", flush=True)
            dump_episode(src_repo_id, ep, ep_dir)

            length = int((ep_dir / "length.txt").read_text())
            if length == 0:
                print(f"[ep {ep}] skipping empty episode", flush=True)
                shutil.rmtree(ep_dir)
                continue
            state = np.load(ep_dir / "state.npy")
            action = np.load(ep_dir / "action.npy")
            task = (ep_dir / "task.txt").read_text()
            assert state.shape == (length, 14)
            assert action.shape == (length, 14)

            print(f"[ep {ep}] writing {length} frames...", flush=True)
            for i in range(length):
                dataset.add_frame({
                    "observation.images.top": read_jpeg_rgb(ep_dir / "top" / f"{i:06d}.jpg"),
                    "observation.images.left": read_jpeg_rgb(ep_dir / "left" / f"{i:06d}.jpg"),
                    "observation.images.right": read_jpeg_rgb(ep_dir / "right" / f"{i:06d}.jpg"),
                    "observation.state": state[i],
                    "action": action[i],
                    "task": task,
                })
            dataset.save_episode()
            shutil.rmtree(ep_dir)
    finally:
        shutil.rmtree(scratch_root, ignore_errors=True)

    print(f"wrote dataset to {output_path}", file=sys.stderr)

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["yam", "lerobot", src_repo_id.replace("/", "_")],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
