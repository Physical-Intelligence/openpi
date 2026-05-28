"""Dump one episode of allenai/24112025-yam-01 (lerobot v3.0) to disk.

Intended to be run via uv with a newer lerobot:

    uv run --no-project --python 3.11 --with "lerobot>=0.4.0" \
        python scripts/_dump_yam_episode.py --episode_index N --out_dir DIR

Writes:
    DIR/state.npy           float32 (T, 14)
    DIR/action.npy          float32 (T, 14)
    DIR/task.txt            single task string
    DIR/length.txt          T as ascii
    DIR/top/{idx:06d}.jpg
    DIR/left/{idx:06d}.jpg
    DIR/right/{idx:06d}.jpg

`convert_to_lerobot.py` runs this once per episode and consumes the output.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

JPEG_QUALITY = 95


def chw_float_to_hwc_uint8(t):
    arr = t.permute(1, 2, 0).numpy()
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_repo_id", type=str, required=True)
    parser.add_argument("--episode_index", type=int, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for cam in ("top", "left", "right"):
        (args.out_dir / cam).mkdir(exist_ok=True)

    try:
        dataset = LeRobotDataset(args.src_repo_id, episodes=[args.episode_index], tolerance_s=1e-3)
    except ValueError as exc:
        if "no data" in str(exc).lower():
            print(f"[warn] episode {args.episode_index} has no data, skipping", file=sys.stderr)
            (args.out_dir / "length.txt").write_text("0")
            return
        raise

    states, actions = [], []
    task_str = None

    skipped = 0
    write_idx = 0
    for i in range(len(dataset)):
        try:
            frame = dataset[i]
        except (RuntimeError, Exception) as exc:
            print(f"[warn] skipping frame {i}: {exc}", file=sys.stderr)
            skipped += 1
            continue
        states.append(frame["observation.state"].numpy())
        actions.append(frame["action"].numpy())
        if task_str is None:
            task_str = frame["task"]

        for cam, key in (
            ("top", "observation.images.top"),
            ("left", "observation.images.left"),
            ("right", "observation.images.right"),
        ):
            img = chw_float_to_hwc_uint8(frame[key])
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                str(args.out_dir / cam / f"{write_idx:06d}.jpg"),
                bgr,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )
        write_idx += 1

    np.save(args.out_dir / "state.npy", np.stack(states).astype(np.float32))
    np.save(args.out_dir / "action.npy", np.stack(actions).astype(np.float32))
    (args.out_dir / "task.txt").write_text(task_str or "")
    (args.out_dir / "length.txt").write_text(str(write_idx))


if __name__ == "__main__":
    main()
