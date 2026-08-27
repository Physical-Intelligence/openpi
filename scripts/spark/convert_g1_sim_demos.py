#!/usr/bin/env python3
"""Convert synchronized G1 simulation shards into a LeRobot dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np

EXPECTED_FORMAT = "g1-fruit-ninja-pi05-raw-v1"
EXPECTED_STATE_DIM = 29
EXPECTED_ACTION_DIM = 21


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True, help="Hugging Face owner/dataset identifier")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    return parser.parse_args()


def _validate_repo_id(repo_id: str) -> None:
    parts = repo_id.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError("--repo-id must be an owner/dataset identifier")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    if any(set(part) - allowed for part in parts):
        raise ValueError("--repo-id contains unsupported characters")


def _load_manifest(raw_dir: Path) -> dict:
    manifest_path = raw_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != EXPECTED_FORMAT:
        raise ValueError(f"unsupported raw format: {manifest.get('format')!r}")
    if manifest.get("fps") != 50:
        raise ValueError(f"expected 50 Hz demonstrations, got {manifest.get('fps')!r}")
    if manifest.get("state", {}).get("shape") != [EXPECTED_STATE_DIM]:
        raise ValueError("manifest does not contain the 29-value G1 state contract")
    if manifest.get("action", {}).get("shape") != [EXPECTED_ACTION_DIM]:
        raise ValueError("manifest does not contain the 21-value G1 action contract")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("manifest contains no recorded episodes")
    return manifest


def _load_episode(raw_dir: Path, entry: dict, image_shape: tuple[int, int, int]):
    path = raw_dir / entry["path"]
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_sha256 = entry.get("sha256")
    if not isinstance(expected_sha256, str) or _sha256(path) != expected_sha256:
        raise ValueError(f"{path} does not match its recorded SHA-256")
    with np.load(path, allow_pickle=False) as archive:
        image = archive["observation_images_head"]
        state = archive["observation_state"]
        action = archive["action"]
        timestamp = archive["timestamp"]
        frame_index = archive["frame_index"]

    frames = len(image)
    if frames < 1 or entry.get("frames") != frames:
        raise ValueError(f"{path} has an invalid or mismatched frame count")
    if image.shape != (frames, *image_shape) or image.dtype != np.uint8:
        raise ValueError(f"{path} has invalid RGB shape or dtype: {image.shape} {image.dtype}")
    if state.shape != (frames, EXPECTED_STATE_DIM) or state.dtype != np.float32:
        raise ValueError(f"{path} has invalid state shape or dtype: {state.shape} {state.dtype}")
    if action.shape != (frames, EXPECTED_ACTION_DIM) or action.dtype != np.float32:
        raise ValueError(f"{path} has invalid action shape or dtype: {action.shape} {action.dtype}")
    if timestamp.shape != (frames,) or frame_index.shape != (frames,):
        raise ValueError(f"{path} has invalid timing arrays")
    if not np.array_equal(frame_index, np.arange(frames)):
        raise ValueError(f"{path} frame indices are not contiguous")
    expected_timestamps = np.arange(frames, dtype=np.float64) / 50.0
    if not np.allclose(timestamp, expected_timestamps, atol=1e-6):
        raise ValueError(f"{path} timestamps are not synchronized at 50 Hz")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError(f"{path} contains non-finite state or action values")
    return image, state, action


def main() -> None:
    args = parse_args()
    _validate_repo_id(args.repo_id)
    manifest = _load_manifest(args.raw_dir)
    image_shape = tuple(manifest["image"]["shape"])
    if len(image_shape) != 3 or image_shape[-1] != 3:
        raise ValueError(f"invalid RGB shape in manifest: {image_shape}")

    output_path = HF_LEROBOT_HOME / args.repo_id
    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="unitree_g1",
        fps=50,
        features={
            "observation.images.head": {
                "dtype": "image",
                "shape": image_shape,
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (EXPECTED_STATE_DIM,),
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (EXPECTED_ACTION_DIM,),
                "names": ["action"],
            },
        },
        use_videos=True,
        image_writer_threads=8,
        image_writer_processes=4,
    )

    converted = 0
    for entry in manifest["episodes"]:
        if not entry.get("success", False) and not args.allow_failures:
            raise ValueError(
                f"{entry.get('path')} is marked unsuccessful; refusing to convert without --allow-failures"
            )
        image, state, action = _load_episode(args.raw_dir, entry, image_shape)
        task = entry.get("task") or manifest["task"]
        for frame_index in range(len(image)):
            dataset.add_frame(
                {
                    "observation.images.head": image[frame_index],
                    "observation.state": state[frame_index],
                    "action": action[frame_index],
                    "task": task,
                }
            )
        dataset.save_episode()
        converted += 1

    if converted == 0:
        raise RuntimeError("no episodes were converted")
    if args.push_to_hub:
        dataset.push_to_hub(
            tags=["unitree-g1", "fruit-ninja", "simulation", "pi0.5"],
            private=True,
            push_videos=True,
            license="apache-2.0",
        )
    print(json.dumps({"status": "passed", "episodes": converted, "output": str(output_path)}))


if __name__ == "__main__":
    main()
