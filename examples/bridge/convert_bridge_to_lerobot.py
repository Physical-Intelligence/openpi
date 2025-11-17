"""
Simple script for converting Bridge dataset (TFDS format) to LeRobot format.

This is a minimal conversion script similar to the Libero example.
For more options (camera selection, video encoding, etc.), use convert_bridge_to_lerobot.py

Usage:
    # Convert to default HuggingFace cache location
    uv run --group rlds examples/bridge/convert_bridge_to_lerobot_simple.py \
        --data_dir /path/to/bridge_release/data/tfds

    # Convert to custom output directory
    uv run --group rlds examples/bridge/convert_bridge_to_lerobot_simple.py \
        --data_dir /path/to/bridge_release/data/tfds \
        --output_dir /custom/output/path \
        --repo_name bridge_lerobot

Note: Install RLDS dependencies first:
    uv sync --group rlds

The Bridge dataset has:
- 60,096 training episodes + 3,475 test episodes
- WidowX robot with 7D actions: [x, y, z, roll, pitch, yaw, gripper]
- Images at 224×224×3 RGB (we use image_0 and image_1 cameras)
"""

import shutil
from pathlib import Path
from PIL import Image
import numpy as np

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
import tensorflow as tf
import tensorflow_datasets as tfds
import tyro


def resize_image(image: np.ndarray, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resize image to target size."""
    image = Image.fromarray(image)
    return np.array(image.resize(size, resample=Image.BICUBIC))


def decode_image_if_needed(image: np.ndarray) -> np.ndarray:
    """Decode image if it's JPEG encoded."""
    if image.dtype == object or image.ndim == 0 or (image.ndim == 1 and len(image) < 1000):
        import io
        if isinstance(image, np.ndarray) and image.ndim == 0:
            image = image.item()
        image = np.array(Image.open(io.BytesIO(image)))
    return image


def main(
    data_dir: str,
    *,
    output_dir: str | None = None,
    repo_name: str = "bridge_lerobot",
    split: str = "train"
):
    """Convert Bridge TFDS dataset to LeRobot format.

    Args:
        data_dir: Path to TFDS dataset directory (parent of 'bridge_dataset' folder)
        output_dir: Custom output directory (default: HuggingFace cache)
        repo_name: Name of the output dataset (default: "bridge_lerobot")
        split: Which split to convert ('train' or 'test')
    """
    # Disable GPU for TensorFlow (only needed for data loading)
    tf.config.set_visible_devices([], "GPU")

    # Determine output path
    if output_dir is not None:
        output_path = Path(output_dir) / repo_name
        print(f"Using custom output directory: {output_path}")
    else:
        output_path = HF_LEROBOT_HOME / repo_name
        print(f"Using default HuggingFace cache: {output_path}")

    # Clean up existing dataset
    if output_path.exists():
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)

    # Create LeRobot dataset
    print(f"Creating LeRobot dataset: {repo_name}")
    print(f"Output path: {output_path}")

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        root=output_path,  # Explicitly specify output location
        robot_type="widowx",
        fps=15,
        features={
            "image_0": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "image_1": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["state"],
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

    # Load Bridge dataset from TFDS
    print(f"Loading Bridge dataset from {data_dir}")
    try:
        builder = tfds.builder("bridge_dataset", data_dir=data_dir)
        raw_dataset = builder.as_dataset(split=split)
    except (AssertionError, ValueError):
        # Try alternative name
        builder = tfds.builder("bridge", data_dir=data_dir)
        raw_dataset = builder.as_dataset(split=split)

    # Convert episodes
    episode_count = 0
    skipped_count = 0

    for episode in raw_dataset:
        # Get language instruction
        has_language = episode.get("episode_metadata", {}).get("has_language", True)
        if hasattr(has_language, "numpy"):
            has_language = has_language.numpy()

        steps = episode["steps"]
        language_instruction = None

        for step in steps:
            if language_instruction is None:
                language_instruction = step.get("language_instruction", "")
                if hasattr(language_instruction, "numpy"):
                    language_instruction = language_instruction.numpy()
                if isinstance(language_instruction, bytes):
                    language_instruction = language_instruction.decode("utf-8")

        # Skip episodes without language annotations (~30% of dataset)
        if not has_language or not language_instruction or language_instruction.strip() == "":
            skipped_count += 1
            continue

        # Process steps
        for step in steps:
            obs = step["observation"]

            # Get state and action
            state = obs["state"].numpy().astype(np.float32)
            action = step["action"].numpy().astype(np.float32)

            # Get and process images
            image_0 = decode_image_if_needed(obs["image_0"].numpy())
            image_1 = decode_image_if_needed(obs["image_1"].numpy())

            # Resize to 224x224
            image_0 = resize_image(image_0)
            image_1 = resize_image(image_1)

            # Add frame
            dataset.add_frame({
                "image_0": image_0,
                "image_1": image_1,
                "state": state,
                "actions": action,
                "task": language_instruction,
            })

        dataset.save_episode()
        episode_count += 1

        # Clear hf_dataset to prevent OOM (data is already saved to parquet)
        dataset.hf_dataset = dataset.create_hf_dataset()

        if episode_count % 100 == 0:
            print(f"Converted {episode_count} episodes...")

    print(f"\nSuccessfully converted {episode_count} episodes")
    print(f"Skipped {skipped_count} episodes without language annotations")

if __name__ == "__main__":
    tyro.cli(main)
