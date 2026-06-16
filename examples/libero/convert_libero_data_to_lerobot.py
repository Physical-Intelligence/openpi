"""Convert a local LIBERO RLDS mirror into a local LeRobot dataset.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /root/pi_train/modified_libero_rlds

This writes a LeRobot dataset under HF_LEROBOT_HOME/<repo_name>.
By default it targets physical-intelligence/libero so the standard pi05_libero
training config can run fully offline from the local cache.
"""

from collections.abc import Iterator, Sequence
from io import BytesIO
from pathlib import Path
import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from PIL import Image
from tfrecord.reader import sequence_loader
import tyro

DEFAULT_REPO_NAME = "physical-intelligence/libero"
DEFAULT_RAW_DATASET_NAMES = (
    "libero_10_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
)


def _decode_image(value: bytes | np.bytes_) -> np.ndarray:
    return np.asarray(Image.open(BytesIO(bytes(value))).convert("RGB"))


def _iter_episodes(dataset_root: Path) -> Iterator[dict[str, np.ndarray]]:
    for shard_path in sorted((dataset_root / "1.0.0").glob("*.tfrecord-*")):
        for context, _ in sequence_loader(str(shard_path), None):
            yield context


def main(
    data_dir: str,
    *,
    repo_name: str = DEFAULT_REPO_NAME,
    raw_dataset_names: Sequence[str] = DEFAULT_RAW_DATASET_NAMES,
    max_episodes_per_dataset: int | None = None,
):
    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="panda",
        fps=10,
        use_videos=False,
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
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
    )

    base_dir = Path(data_dir)
    total_episodes = 0
    for raw_dataset_name in raw_dataset_names:
        dataset_root = base_dir / raw_dataset_name
        saved_for_dataset = 0
        for context in _iter_episodes(dataset_root):
            num_steps = len(context["steps/is_last"])
            actions = np.asarray(context["steps/action"], dtype=np.float32).reshape(num_steps, 7)
            states = np.asarray(context["steps/observation/state"], dtype=np.float32).reshape(num_steps, 8)
            main_images = context["steps/observation/image"]
            wrist_images = context["steps/observation/wrist_image"]
            tasks = context["steps/language_instruction"]

            for step_idx in range(num_steps):
                dataset.add_frame(
                    {
                        "image": _decode_image(main_images[step_idx]),
                        "wrist_image": _decode_image(wrist_images[step_idx]),
                        "state": states[step_idx],
                        "actions": actions[step_idx],
                        "task": bytes(tasks[step_idx]).decode("utf-8"),
                    }
                )

            dataset.save_episode()
            saved_for_dataset += 1
            total_episodes += 1
            print(
                f"saved {raw_dataset_name} episode {saved_for_dataset} -> total={total_episodes}",
                flush=True,
            )

            if max_episodes_per_dataset is not None and saved_for_dataset >= max_episodes_per_dataset:
                break

    print(f"Finished writing {total_episodes} episodes to {output_path}")


if __name__ == "__main__":
    tyro.cli(main)
