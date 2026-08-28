"""
Minimal example script for converting a UR5 dataset to LeRobot format.

This script assumes your raw UR5 data is stored as a directory of episodes,
where each episode is a directory containing:
  - A sequence of RGB images from the base camera (base_rgb_*.png or base_rgb_*.jpg)
  - A sequence of RGB images from the wrist camera (wrist_rgb_*.png or wrist_rgb_*.jpg)
  - A numpy file (states.npy) of shape (T, 7): 6 joint angles + 1 gripper position
  - A numpy file (actions.npy) of shape (T, 7): 6 joint angle targets + 1 gripper command
  - A text file (task.txt) containing the task description (language instruction)

Adapt the loading logic below to match your actual data format.

Usage:
    uv run examples/ur5/convert_ur5_data_to_lerobot.py --data_dir /path/to/ur5_data

To push to the Hugging Face Hub:
    uv run examples/ur5/convert_ur5_data_to_lerobot.py --data_dir /path/to/ur5_data --push_to_hub
"""

import pathlib
import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from PIL import Image
import tyro

REPO_NAME = "your_hf_username/ur5_dataset"  # Change to your Hugging Face repo id


def main(data_dir: str, *, push_to_hub: bool = False) -> None:
    data_path = pathlib.Path(data_dir)
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create the LeRobot dataset.
    # The feature names here must match the keys used in the repack transform defined in
    # LeRobotUR5DataConfig (src/openpi/training/config.py).
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="ur5",
        fps=10,
        features={
            "base_rgb": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_rgb": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "joints": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["joints"],
            },
            "gripper": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Iterate over episodes. Each sub-directory of data_dir is one episode.
    episode_dirs = sorted(p for p in data_path.iterdir() if p.is_dir())
    if not episode_dirs:
        raise ValueError(f"No episode directories found under {data_dir}")

    for episode_dir in episode_dirs:
        states = np.load(episode_dir / "states.npy")  # (T, 7)
        actions = np.load(episode_dir / "actions.npy")  # (T, 7)
        task = (episode_dir / "task.txt").read_text().strip()

        base_images = sorted(episode_dir.glob("base_rgb_*"))
        wrist_images = sorted(episode_dir.glob("wrist_rgb_*"))

        num_steps = len(states)
        assert len(base_images) == num_steps, (
            f"Expected {num_steps} base images in {episode_dir}, got {len(base_images)}"
        )
        assert len(wrist_images) == num_steps, (
            f"Expected {num_steps} wrist images in {episode_dir}, got {len(wrist_images)}"
        )

        for t in range(num_steps):
            base_img = np.array(Image.open(base_images[t]).convert("RGB"))
            wrist_img = np.array(Image.open(wrist_images[t]).convert("RGB"))

            dataset.add_frame(
                {
                    "base_rgb": base_img,
                    "wrist_rgb": wrist_img,
                    "joints": states[t, :6].astype(np.float32),
                    "gripper": states[t, 6:7].astype(np.float32),
                    "actions": actions[t].astype(np.float32),
                    "task": task,
                }
            )

        dataset.save_episode()

    if push_to_hub:
        dataset.push_to_hub(
            tags=["ur5", "manipulation"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
