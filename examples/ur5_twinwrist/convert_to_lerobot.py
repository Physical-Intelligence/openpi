"""Offline HDF5 to the exact LeRobot revision locked by uv.lock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import matplotlib.pyplot as plt
import numpy as np

WRIST_COORDINATE = "yaml_servo_zero_relative_raw"
ACTION_SPACE = "tcp_speedL_6+wrist_j1_j2_yaml_zero_relative_raw_2+gripper_absolute_1"


def _validate_source_contract(paths: list[Path]) -> None:
    """在创建 LeRobot 目录前拒绝旧角度腕数据或混合 action space。"""

    for path in paths:
        with h5py.File(path, "r") as episode:
            if not bool(episode.attrs.get("success", False)):
                raise ValueError(f"{path} 不是 success episode")
            if episode.attrs.get("wrist_coordinate") != WRIST_COORDINATE:
                raise ValueError(f"{path} 的 wrist_coordinate 不是 {WRIST_COORDINATE}")
            if episode.attrs.get("action_space") != ACTION_SPACE:
                raise ValueError(f"{path} 的 action_space 与当前项目不一致")
            try:
                zero = [int(value) for value in json.loads(str(episode.attrs["wrist_servo_zero_raw"]))]
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path} 缺少合法 wrist_servo_zero_raw") from exc
            if len(zero) != 2 or any(not 0 <= value <= 4095 for value in zero):
                raise ValueError(f"{path} 的 wrist_servo_zero_raw 非法")
            qpos = episode.get("observations/qpos")
            action = episode.get("action")
            if qpos is None or action is None or qpos.ndim != 2 or action.ndim != 2:
                raise ValueError(f"{path} 缺少二维 qpos/action")
            if qpos.shape[1] != 9 or action.shape[1] != 9 or len(qpos) != len(action) or not len(action):
                raise ValueError(f"{path} 的 qpos/action shape 或长度不符合 9 维合同")


def convert(source: str | Path, repo_id: str, output_root: str | Path | None = None) -> Path:
    dataset_root = None if output_root is None else Path(output_root).resolve() / repo_id
    paths = sorted(Path(source).glob("episode_*.hdf5"))
    if not paths:
        raise FileNotFoundError(f"no completed episodes in {source}")
    _validate_source_contract(paths)
    with h5py.File(paths[0], "r") as sample:
        image_shape = tuple(sample["observations/images/front"].shape[1:])
        fps = int(sample.attrs["fps"])
    features = {
        **{
            f"observation.images.{name}": {
                "dtype": "image",
                "shape": image_shape,
                "names": ["height", "width", "channel"],
            }
            for name in ("front", "side", "top")
        },
        "observation.state": {"dtype": "float32", "shape": (9,), "names": ["state"]},
        "action": {"dtype": "float32", "shape": (9,), "names": ["action"]},
    }
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=dataset_root,
        robot_type="ur5_twinwrist",
        fps=fps,
        features=features,
        image_writer_threads=4,
        image_writer_processes=0,
    )
    for path in paths:
        with h5py.File(path, "r") as episode:
            if not bool(episode.attrs.get("success", False)):
                continue
            task = str(episode.attrs["task"])
            for i in range(len(episode["action"])):
                dataset.add_frame(
                    {
                        "observation.state": np.asarray(episode["observations/qpos"][i], np.float32),
                        "action": np.asarray(episode["action"][i], np.float32),
                        "task": task,
                        **{
                            f"observation.images.{name}": np.asarray(episode[f"observations/images/{name}"][i])
                            for name in ("front", "side", "top")
                        },
                    }
                )
            dataset.save_episode()
    root = Path(dataset.root)
    reloaded = LeRobotDataset(repo_id, root=root)
    if len(reloaded) == 0:
        raise RuntimeError("converted dataset is empty")
    indices = np.linspace(0, len(reloaded) - 1, min(10, len(reloaded)), dtype=int)
    print(f"reloaded frames={len(reloaded)} episodes={reloaded.num_episodes}")
    for index in indices:
        frame = reloaded[int(index)]
        print(
            index,
            {key: (tuple(value.shape), str(value.dtype)) for key, value in frame.items() if hasattr(value, "shape")},
        )
    # Small local evidence artifacts: three-view preview and state/action curves.
    writer = None
    try:
        for index in range(len(reloaded)):
            frame = reloaded[index]
            views = []
            for name in ("front", "side", "top"):
                image = np.asarray(frame[f"observation.images.{name}"])
                if image.shape[0] == 3:
                    image = np.moveaxis(image, 0, -1)
                image = (
                    (np.clip(image, 0, 1) * 255).astype(np.uint8)
                    if np.issubdtype(image.dtype, np.floating)
                    else image.astype(np.uint8)
                )
                views.append(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            canvas = np.concatenate(views, axis=1)
            if writer is None:
                writer = cv2.VideoWriter(
                    str(root / "three_camera_preview.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (canvas.shape[1], canvas.shape[0]),
                )
            writer.write(canvas)
    finally:
        if writer is not None:
            writer.release()
    states = np.stack([np.asarray(reloaded[i]["observation.state"]) for i in range(len(reloaded))])
    actions = np.stack([np.asarray(reloaded[i]["action"]) for i in range(len(reloaded))])
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(states)
    axes[0].set_title("state")
    axes[1].plot(actions)
    axes[1].set_title("action")
    fig.tight_layout()
    fig.savefig(root / "state_action.png")
    plt.close(fig)
    return root


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--repo-id", default="local/ur5_twinwrist")
    p.add_argument("--output-root")
    a = p.parse_args()
    print(convert(a.source, a.repo_id, a.output_root))


if __name__ == "__main__":
    main()
