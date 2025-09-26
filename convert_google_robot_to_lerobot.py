import numpy as np
import json

from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro

google_demo_conversion_cfg = {
    "repo_id": "hchen/google_robot",
    "demo_dir": "./simpler_env_demos/demo_collection/collected_data"
}

def main(data_cfg):
    demo_root_dir = Path(data_cfg["demo_dir"])
    task_dirs = [d for d in demo_root_dir.iterdir() if d.is_dir()]
    task_metadata_paths = [d / "metadata.json" for d in task_dirs]
    
    task_metadata = [json.loads(p.read_text()) for p in task_metadata_paths]

    dataset = LeRobotDataset.create(
        repo_id=data_cfg["repo_id"],
        robot_type="google_robot",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"]
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            }
        },
        image_writer_threads=10,
        image_writer_processes=5
    )

    for single_task_ds in task_metadata:
        success_episode_files = ["./simpler_env_demos/demo_collection/" + file for file in single_task_ds["success_episode_files"]]
        for i, episode_file in enumerate(success_episode_files):
            traj_data = np.load(episode_file)
            
            traj_images = traj_data["images"]
            traj_states = traj_data["states"]
            traj_actions = traj_data["actions"]
            traj_language = str(traj_data["language"])

            episode_len = len(traj_images)

            for t in range(episode_len):
                dataset.add_frame(
                    {
                        "image": traj_images[t],
                        "state": traj_states[t],
                        "actions": traj_actions[t],
                        "task": traj_language
                    }
                )
            
            dataset.save_episode()
            print(f"Saved trajectory {i} out of {single_task_ds['total_saved_episodes']} for {single_task_ds['environment_name']}")

if __name__ == "__main__":
    main(data_cfg=google_demo_conversion_cfg)