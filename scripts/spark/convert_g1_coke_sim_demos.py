#!/usr/bin/env python3
"""Convert validated Isaac G1 Coke-pickup demonstrations to LeRobot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np

EXPECTED_FORMAT = "g1-coke-pickup-pi05-raw-v1"
EXPECTED_FPS = 50
EXPECTED_STATE_DIM = 24
EXPECTED_ACTION_DIM = 7


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_id(value: str) -> str:
    parts = value.split("/")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    if len(parts) != 2 or not all(parts) or any(set(part) - allowed for part in parts):
        raise argparse.ArgumentTypeError(
            "repo id must be owner/dataset using letters, numbers, dot, dash, or underscore"
        )
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--repo-id", type=_repo_id, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    return parser.parse_args()


def _load_manifest(raw_dir: Path) -> dict[str, object]:
    manifest = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != EXPECTED_FORMAT:
        raise ValueError(f"unsupported raw format {manifest.get('format')!r}")
    if manifest.get("fps") != EXPECTED_FPS:
        raise ValueError(f"expected {EXPECTED_FPS} Hz demonstrations")
    if manifest.get("state", {}).get("shape") != [EXPECTED_STATE_DIM]:
        raise ValueError("manifest does not contain the 24-value G1 upper-body state")
    if manifest.get("action", {}).get("shape") != [EXPECTED_ACTION_DIM]:
        raise ValueError("manifest does not contain the seven-value Coke task action")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("manifest contains no episodes")
    return manifest


def _load_episode(raw_dir: Path, entry: dict[str, object], image_shape: tuple[int, int, int]):
    path = raw_dir / str(entry["path"])
    if _sha256(path) != entry.get("sha256"):
        raise ValueError(f"{path} does not match its recorded SHA-256")
    with np.load(path, allow_pickle=False) as archive:
        image = archive["observation_images_head"]
        state = archive["observation_state"]
        action = archive["action"]
        timestamp = archive["timestamp"]
        frame_index = archive["frame_index"]
    frames = len(image)
    if frames < 1 or entry.get("frames") != frames:
        raise ValueError(f"invalid or mismatched frame count in {path}")
    if image.shape != (frames, *image_shape) or image.dtype != np.uint8:
        raise ValueError(f"invalid RGB array in {path}: {image.shape} {image.dtype}")
    if state.shape != (frames, EXPECTED_STATE_DIM) or state.dtype != np.float32:
        raise ValueError(f"invalid state array in {path}: {state.shape} {state.dtype}")
    if action.shape != (frames, EXPECTED_ACTION_DIM) or action.dtype != np.float32:
        raise ValueError(f"invalid action array in {path}: {action.shape} {action.dtype}")
    if not np.isfinite(state).all() or not np.isfinite(action).all() or np.max(np.abs(action)) > 1.000001:
        raise ValueError(f"non-finite or out-of-range values in {path}")
    if not np.array_equal(frame_index, np.arange(frames, dtype=np.int64)):
        raise ValueError(f"non-contiguous frame indices in {path}")
    if not np.allclose(timestamp, np.arange(frames, dtype=np.float64) / EXPECTED_FPS, atol=1.0e-6):
        raise ValueError(f"timestamps are not synchronized at {EXPECTED_FPS} Hz in {path}")
    return image, state, action


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.expanduser().resolve()
    manifest = _load_manifest(raw_dir)
    image_shape = tuple(manifest["image"]["shape"])
    if image_shape != (240, 320, 3):
        raise ValueError(f"expected rendered 240x320 RGB, got {image_shape}")

    output_path = HF_LEROBOT_HOME / args.repo_id
    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")
        shutil.rmtree(output_path)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="unitree_g1",
        fps=EXPECTED_FPS,
        features={
            "observation.images.head": {
                "dtype": "image",
                "shape": image_shape,
                "names": ["height", "width", "channel"],
            },
            "observation.state": {"dtype": "float32", "shape": (EXPECTED_STATE_DIM,), "names": ["state"]},
            "action": {"dtype": "float32", "shape": (EXPECTED_ACTION_DIM,), "names": ["action"]},
        },
        use_videos=True,
        image_writer_threads=8,
        image_writer_processes=4,
    )

    converted = 0
    for entry in manifest["episodes"]:
        if not entry.get("success", False) and not args.allow_failures:
            raise ValueError(f"{entry.get('path')} is unsuccessful; refusing conversion without --allow-failures")
        image, state, action = _load_episode(raw_dir, entry, image_shape)
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

    if args.push_to_hub:
        dataset.push_to_hub(
            tags=["unitree-g1", "coke-pickup", "isaaclab", "simulation", "pi0.5"],
            private=True,
            push_videos=True,
            license="apache-2.0",
        )
    print(json.dumps({"status": "passed", "episodes": converted, "output": str(output_path)}))


if __name__ == "__main__":
    main()
