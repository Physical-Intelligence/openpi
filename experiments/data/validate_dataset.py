# validate_dataset.py: Validates a converted LeRobot v2.1 dataset against its v3.0 source.
# Checks episode/frame counts, action/state shapes, value ranges, and image integrity.

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def validate(repo_id: str, source_dirs: list[Path], lerobot_home: str | None = None):
    """Validate a converted dataset against source data."""
    # Set HF_LEROBOT_HOME before importing lerobot
    import os
    if lerobot_home:
        os.environ["HF_LEROBOT_HOME"] = lerobot_home

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    print(f"=== Validating: {repo_id} ===\n")
    errors = []

    # 1. Load converted dataset
    try:
        ds = LeRobotDataset(repo_id)
    except Exception as e:
        print(f"FAIL: Cannot load dataset: {e}")
        return False

    print(f"Converted dataset: {len(ds)} frames, {ds.meta.total_episodes} episodes")

    # 2. Load source data for comparison
    source_frames = 0
    source_episodes = 0
    source_actions = []
    source_states = []
    for src in source_dirs:
        with open(src / "meta" / "info.json") as f:
            info = json.load(f)
        source_episodes += info["total_episodes"]
        source_frames += info["total_frames"]
        t = pq.read_table(src / "data" / "chunk-000" / "file-000.parquet")
        df = t.to_pandas()
        source_actions.extend(df["action"].tolist())
        source_states.extend(df["observation.state"].tolist())

    print(f"Source data:       {source_frames} frames, {source_episodes} episodes\n")

    # 3. Check counts match
    if ds.meta.total_episodes != source_episodes:
        errors.append(f"Episode count mismatch: converted={ds.meta.total_episodes}, source={source_episodes}")
    else:
        print(f"[OK] Episode count: {ds.meta.total_episodes}")

    if len(ds) != source_frames:
        errors.append(f"Frame count mismatch: converted={len(ds)}, source={source_frames}")
    else:
        print(f"[OK] Frame count: {len(ds)}")

    # 4. Check action/state dimensions
    sample = ds[0]
    action_shape = sample["action"].shape
    state_shape = sample["observation.state"].shape
    if action_shape[-1] != 7:
        errors.append(f"Action dim wrong: {action_shape}")
    else:
        print(f"[OK] Action dim: {action_shape}")

    if state_shape[-1] != 7:
        errors.append(f"State dim wrong: {state_shape}")
    else:
        print(f"[OK] State dim: {state_shape}")

    # 5. Check image shapes and non-zero content
    for cam_key in ["observation.images.cam_high", "observation.images.cam_right_wrist"]:
        if cam_key in sample:
            img = sample[cam_key]
            if img.shape[0] == 3:  # CHW
                h, w = img.shape[1], img.shape[2]
            else:  # HWC
                h, w = img.shape[0], img.shape[1]
            if h < 100 or w < 100:
                errors.append(f"{cam_key}: suspiciously small {img.shape}")
            elif img.float().mean() < 0.001:
                errors.append(f"{cam_key}: all-black image (mean={img.float().mean():.4f})")
            else:
                print(f"[OK] {cam_key}: shape={tuple(img.shape)}, mean={img.float().mean():.3f}")
        else:
            errors.append(f"Missing camera: {cam_key}")

    # 6. Spot-check action/state values against source
    # Compare first and last episodes' first frames
    source_actions_arr = np.array(source_actions)
    source_states_arr = np.array(source_states)

    converted_first = ds[0]
    src_action_0 = source_actions_arr[0]
    conv_action_0 = converted_first["action"].numpy()

    action_diff = np.abs(src_action_0 - conv_action_0).max()
    if action_diff > 1e-4:
        errors.append(f"Action mismatch at frame 0: max_diff={action_diff:.6f}")
    else:
        print(f"[OK] Action values match source (frame 0, max_diff={action_diff:.6f})")

    state_diff = np.abs(source_states_arr[0] - converted_first["observation.state"].numpy()).max()
    if state_diff > 1e-4:
        errors.append(f"State mismatch at frame 0: max_diff={state_diff:.6f}")
    else:
        print(f"[OK] State values match source (frame 0, max_diff={state_diff:.6f})")

    # Check last frame too
    converted_last = ds[len(ds) - 1]
    src_action_last = source_actions_arr[-1]
    conv_action_last = converted_last["action"].numpy()
    action_diff_last = np.abs(src_action_last - conv_action_last).max()
    if action_diff_last > 1e-4:
        errors.append(f"Action mismatch at last frame: max_diff={action_diff_last:.6f}")
    else:
        print(f"[OK] Action values match source (last frame, max_diff={action_diff_last:.6f})")

    # 7. Check episode boundaries (random episode)
    mid_ep = ds.meta.total_episodes // 2
    ep_start = ds.episode_data_index["from"][mid_ep].item()
    ep_end = ds.episode_data_index["to"][mid_ep].item()
    ep_len = ep_end - ep_start
    if ep_len < 10:
        errors.append(f"Episode {mid_ep} suspiciously short: {ep_len} frames")
    else:
        print(f"[OK] Episode {mid_ep}: {ep_len} frames (sanity check)")

    # 8. Spot-check a mid-dataset image is not black
    mid_idx = len(ds) // 2
    mid_sample = ds[mid_idx]
    mid_img = mid_sample["observation.images.cam_high"]
    if mid_img.float().mean() < 0.001:
        errors.append(f"Mid-dataset image (idx={mid_idx}) is all black")
    else:
        print(f"[OK] Mid-dataset image (idx={mid_idx}): mean={mid_img.float().mean():.3f}")

    # Summary
    print(f"\n{'='*40}")
    if errors:
        print(f"FAILED: {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        return False
    else:
        print("ALL CHECKS PASSED")
        return True


def main():
    parser = argparse.ArgumentParser(description="Validate converted LeRobot dataset")
    parser.add_argument("--repo-id", required=True, help="Converted dataset repo ID")
    parser.add_argument("--source-dirs", nargs="+", type=Path, required=True,
                        help="Source v3.0 processed dataset directories")
    parser.add_argument("--lerobot-home", type=str, default=None,
                        help="HF_LEROBOT_HOME path")
    args = parser.parse_args()

    ok = validate(args.repo_id, args.source_dirs, args.lerobot_home)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
