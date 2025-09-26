"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $HF_LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro

import numpy as np
import pickle

import os, struct

def main(scheduler_pkl_path="./test_logs/scheduler_00000010.pkl",
         repo_name="hchen/libero"): 
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float64",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float64",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    with open(scheduler_pkl_path, "rb") as f:
        scheduler = pickle.load(f)

    archive = scheduler.archive
    all_trajectories = archive.data("trajectories")

    include_failures = True

    for elite_trajectories in all_trajectories:
        for traj_id, traj in enumerate(elite_trajectories):
            episode_len = np.array(traj["image"]).shape[0]

            if not include_failures and not traj["success"]:
                continue

            for step in range(episode_len):
                dataset.add_frame(
                    {
                        "image": traj["image"][step],
                        "wrist_image": traj["wrist_image"][step],
                        "state": traj["state"][step],
                        "actions": traj["action"][step],
                        "task": traj["prompt"]
                    }
                )

            dataset.save_episode()
            print(f"Saved trajectory {traj_id}!")
        
        print(dataset)

if __name__ == "__main__":
    main(scheduler_pkl_path="./test_logs/scheduler_00000010.pkl",
         repo_name="hchen/libero")