"""Convert a DROID *RLDS* (TFDS) dataset to LeRobot format with DROID **joint-space** fields.

Motivation: the off-the-shelf `lerobot/droid_100` port stores a 7-dim *Cartesian* state/action,
which does not match openpi's DROID configs. The official `convert_droid_data_to_lerobot.py` produces
the right 8-dim joint schema, but reads raw DROID `trajectory.h5` + `recordings/MP4`. This script
produces the SAME LeRobot schema as that official converter, but sources from an already-downloaded
RLDS/TFDS build (e.g. `lerobot-raw/droid_100_raw`, TFDS builder `r2d2_faceblur`). The output therefore
loads directly through `LeRobotDROIDDataConfig` (see `pi05_droid_finetune`).

Output schema (matches examples/droid/convert_droid_data_to_lerobot.py):
  exterior_image_1_left, exterior_image_2_left, wrist_image_left : image (180, 320, 3)
  joint_position : float32 (7,)
  gripper_position : float32 (1,)
  actions : float32 (8,)   # joint_velocity (7) + gripper_position (1)
  task : language instruction

Usage (set HF_LEROBOT_HOME first so the dataset lands on scratch, not $HOME):
  uv run --group rlds examples/droid/convert_droid_rlds_to_lerobot.py \
      --rlds-data-dir /path/to/droid_rlds --builder-name r2d2_faceblur --version 1.0.0 \
      --repo-id samratsahoo/droid_100_joint
"""

import shutil

import numpy as np
import tyro


def _instruction(step: dict) -> str:
    """Pick the first non-empty language instruction from a step, decoding bytes."""
    for key in ("language_instruction", "language_instruction_2", "language_instruction_3"):
        val = step.get(key, b"")
        if isinstance(val, bytes):
            val = val.decode("utf-8")
        val = val.strip()
        if val:
            return val
    return "do something"


def main(
    rlds_data_dir: str,
    repo_id: str,
    *,
    builder_name: str = "r2d2_faceblur",
    version: str = "1.0.0",
    max_episodes: int | None = None,
    push_to_hub: bool = False,
):
    # Import heavy deps lazily so `--help` works without the rlds group.
    import tensorflow as tf

    # Keep TF off the GPU so it doesn't clobber other frameworks; conversion is CPU/IO bound.
    tf.config.set_visible_devices([], "GPU")
    import tensorflow_datasets as tfds
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=15,  # DROID is recorded at 15 fps
        features={
            "exterior_image_1_left": {"dtype": "image", "shape": (180, 320, 3), "names": ["height", "width", "channel"]},
            "exterior_image_2_left": {"dtype": "image", "shape": (180, 320, 3), "names": ["height", "width", "channel"]},
            "wrist_image_left": {"dtype": "image", "shape": (180, 320, 3), "names": ["height", "width", "channel"]},
            "joint_position": {"dtype": "float32", "shape": (7,), "names": ["joint_position"]},
            "gripper_position": {"dtype": "float32", "shape": (1,), "names": ["gripper_position"]},
            # Joint *velocity* (7) + gripper position (1); matches the action space pi05-DROID was pretrained on.
            "actions": {"dtype": "float32", "shape": (8,), "names": ["actions"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    builder = tfds.builder(builder_name, data_dir=rlds_data_dir, version=version)
    n_total = builder.info.splits["train"].num_examples
    rlds = builder.as_dataset(split="train")
    n = n_total if max_episodes is None else min(max_episodes, n_total)
    print(f"Converting {n}/{n_total} episodes from {builder_name}:{version}")

    for ep_idx, episode in enumerate(rlds.take(n)):
        steps = list(episode["steps"].as_numpy_iterator())
        task = _instruction(steps[0])
        for step in steps:
            obs = step["observation"]
            act = step["action_dict"]
            # RLDS DROID images are already RGB uint8 at 180x320x3 -- write directly (no resize/flip).
            dataset.add_frame(
                {
                    "exterior_image_1_left": obs["exterior_image_1_left"],
                    "exterior_image_2_left": obs["exterior_image_2_left"],
                    "wrist_image_left": obs["wrist_image_left"],
                    "joint_position": obs["joint_position"].astype(np.float32),
                    "gripper_position": obs["gripper_position"].astype(np.float32),
                    "actions": np.concatenate([act["joint_velocity"], act["gripper_position"]]).astype(np.float32),
                    "task": task,
                }
            )
        dataset.save_episode()
        print(f"  [{ep_idx + 1}/{n}] {len(steps)} steps | task: {task!r}")

    print(f"Done. Wrote dataset to {output_path}")
    if push_to_hub:
        dataset.push_to_hub(tags=["droid", "panda", "rlds"], private=False, push_videos=True, license="apache-2.0")


if __name__ == "__main__":
    tyro.cli(main)
