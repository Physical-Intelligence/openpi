"""Build a *combined* LeRobot dataset = DROID-100 (RLDS) + extra successful trajectories.

This is `convert_droid_rlds_to_lerobot.py` plus a second source: a directory of "tamp-vla" style
successful trajectories (each a folder with a `tiptop_plan.json` dense plan + raw camera mp4s). Both
sources are written into ONE dataset with the standard DROID **joint** schema (8-dim: 7 joints +
gripper), so the result loads directly through `LeRobotDROIDDataConfig` (see `pi05_droid100_lerobot`)
and norm stats can be recomputed with the same config.

Why we read `tiptop_plan.json` (NOT the trajectory's `_lerobot_raw.json` / `le-robot/`):
  The pre-baked `le-robot` export subsampled the dense plan to ~20 keyframes and set
  `action = diff(positions) * 15fps`, which yields joint velocities 15-70x larger than physical
  (mean ~2-7 rad/s, max ~72 vs DROID's ~0.13 mean / ~1.0 max) and a *constant* gripper. We instead
  read the real dense plan: each "trajectory" step carries `positions`+`velocities` at dt=0.02 (50 Hz)
  and each "gripper" step is an `open`/`close` event. We resample 50 Hz -> 15 Hz (DROID's rate), take
  the plan's own (physically valid) velocities as the action, and reconstruct the gripper channel from
  the open/close events (0 = open, 1 = closed, matching DROID; episodes start open).

Image alignment: the raw `external_cam.mp4` (exterior_1) and `hand_cam.mp4` (wrist) span the same
execution window as the plan at a consistent ~34 fps (verified: video_frames / (n_plan_pts*0.02) has
<2% spread across episodes), so we resize them to 180x320 and resample proportionally onto the same
15 Hz timeline. This is a time-proportional alignment, not frame-timestamp-exact (the plan timeline
does not model gripper-actuation pauses, so expect up to a few % drift mid-episode). exterior_2 is a
duplicate of exterior_1: the toys rig has no second exterior camera and pi05 masks that slot anyway.

Usage:
  uv run --group rlds examples/droid/convert_combined_droid_toys_to_lerobot.py \
      --rlds-data-dir /path/to/droid_rlds --builder-name r2d2_faceblur --version 1.0.0 \
      --toys-dir /home/samrat/tamp-vla/tamp-vla-data/toys-no-collision/success \
      --repo-id samratsahoo/droid_100_extended
"""

import json
import pathlib
import shutil

import numpy as np
import tyro

# DROID is 15 Hz; the tamp-vla dense plans are time-parameterized at dt=0.02 s (50 Hz).
_DROID_FPS = 15
_PLAN_DT = 0.02
# Target image size for the DROID LeRobot schema (H, W).
_IMG_HW = (180, 320)


def _instruction(step: dict) -> str:
    """Pick the first non-empty language instruction from an RLDS step, decoding bytes."""
    for key in ("language_instruction", "language_instruction_2", "language_instruction_3"):
        val = step.get(key, b"")
        if isinstance(val, bytes):
            val = val.decode("utf-8")
        val = val.strip()
        if val:
            return val
    return "do something"


def _dense_plan(plan: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten a tiptop plan into dense 50 Hz arrays (positions[M,7], velocities[M,7], gripper[M]).

    "trajectory" steps contribute their per-row positions/velocities; "gripper" steps are instantaneous
    open/close events that set the gripper state (0 = open, 1 = closed) for all following rows. The
    arm starts with the gripper open.
    """
    positions, velocities, gripper = [], [], []
    grip_state = 0.0  # open
    for step in plan["steps"]:
        if step.get("type") == "gripper":
            grip_state = 1.0 if step.get("action") == "close" else 0.0
        elif step.get("type") == "trajectory" and step.get("positions") is not None:
            pos = np.asarray(step["positions"], dtype=np.float32)  # (L, 7)
            vel = np.asarray(step["velocities"], dtype=np.float32)  # (L, 7)
            positions.append(pos)
            velocities.append(vel)
            gripper.append(np.full(len(pos), grip_state, dtype=np.float32))
    return np.concatenate(positions), np.concatenate(velocities), np.concatenate(gripper)


def _resample_indices(n_src: int, n_dst: int) -> np.ndarray:
    """Indices that evenly sample `n_dst` points from a sequence of length `n_src` (nearest)."""
    if n_dst <= 1 or n_src <= 1:
        return np.zeros(max(n_dst, 0), dtype=int)
    return np.clip(np.round(np.arange(n_dst) * (n_src - 1) / (n_dst - 1)), 0, n_src - 1).astype(int)


def _decode_resized(path: str, hw: tuple[int, int]) -> list[np.ndarray]:
    """Decode an mp4 into an ordered list of HWC uint8 RGB frames, resized to (H, W)."""
    import av
    import cv2

    h, w = hw
    container = av.open(path)
    frames = []
    for frame in container.decode(video=0):
        rgb = frame.to_ndarray(format="rgb24")
        if rgb.shape[:2] != (h, w):
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        frames.append(rgb)
    container.close()
    return frames


def _iter_toys_episodes(toys_dir: str):
    """Yield (name, task, frames) for each trajectory folder under `toys_dir`.

    `frames` is a list of per-frame dicts in the DROID joint schema, built from the dense plan
    resampled to DROID's 15 Hz and the raw camera videos resampled onto the same timeline.
    """
    for traj in sorted(pathlib.Path(toys_dir).iterdir()):
        plan_path = traj / "tiptop_plan.json"
        if not plan_path.exists():
            continue
        plan = json.loads(plan_path.read_text())
        task = json.loads((traj / "_lerobot_raw.json").read_text()).get("instruction", "do something")

        pos50, vel50, grip50 = _dense_plan(plan)
        m = len(pos50)
        n = int(round(m * _PLAN_DT * _DROID_FPS))  # 50 Hz -> 15 Hz
        sel = _resample_indices(m, n)
        pos, vel, grip = pos50[sel], vel50[sel], grip50[sel]

        # Decode + resize both raw cameras, then resample onto the same n-frame timeline.
        exterior = _decode_resized(str(traj / "external_cam.mp4"), _IMG_HW)
        wrist = _decode_resized(str(traj / "hand_cam.mp4"), _IMG_HW)
        ext_sel = _resample_indices(len(exterior), n)
        wr_sel = _resample_indices(len(wrist), n)

        frames = []
        for i in range(n):
            frames.append(
                {
                    "exterior_image_1_left": exterior[ext_sel[i]],
                    "exterior_image_2_left": exterior[ext_sel[i]],  # duplicate: pi05 masks the 3rd view
                    "wrist_image_left": wrist[wr_sel[i]],
                    "joint_position": pos[i],
                    "gripper_position": grip[i : i + 1],  # (1,)
                    "actions": np.concatenate([vel[i], grip[i : i + 1]]),  # joint_velocity[7] + gripper[1]
                    "task": task,
                }
            )
        yield traj.name, task, frames


def main(
    rlds_data_dir: str,
    repo_id: str,
    toys_dir: str,
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

    # --- Source 1: DROID-100 RLDS -------------------------------------------------------------
    builder = tfds.builder(builder_name, data_dir=rlds_data_dir, version=version)
    n_total = builder.info.splits["train"].num_examples
    rlds = builder.as_dataset(split="train")
    n = n_total if max_episodes is None else min(max_episodes, n_total)
    print(f"Converting {n}/{n_total} RLDS episodes from {builder_name}:{version}")

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
        print(f"  [droid {ep_idx + 1}/{n}] {len(steps)} steps | task: {task!r}")

    # --- Source 2: extra successful trajectories ---------------------------------------------
    n_toys = 0
    for name, task, frames in _iter_toys_episodes(toys_dir):
        for frame in frames:
            dataset.add_frame(frame)
        dataset.save_episode()
        n_toys += 1
        print(f"  [toys {n_toys}] {len(frames)} steps | {name} | task: {task!r}")

    print(f"Done. Wrote {n} DROID + {n_toys} trajectory episodes to {output_path}")
    if push_to_hub:
        dataset.push_to_hub(tags=["droid", "panda", "rlds"], private=False, push_videos=True, license="apache-2.0")


if __name__ == "__main__":
    tyro.cli(main)
