# process_single_arm.py: Extracts single right-arm data from bimanual ALOHA LeRobot v3.0 datasets.
# Slices state/action to right arm indices (7:14), keeps cam_high + cam_right_wrist only.

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# Right arm occupies indices 7:14 in the 14-dim bimanual action/state vector.
# Layout: [left_joint_0..5, left_gripper, right_joint_0..5, right_gripper]
RIGHT_ARM_SLICE = slice(7, 14)

RIGHT_ARM_JOINT_NAMES = [
    "right_joint_0.pos",
    "right_joint_1.pos",
    "right_joint_2.pos",
    "right_joint_3.pos",
    "right_joint_4.pos",
    "right_joint_5.pos",
    "right_carriage_joint.pos",
]

KEEP_CAMERAS = ["observation.images.cam_high", "observation.images.cam_right_wrist"]
DROP_CAMERA = "observation.images.cam_left_wrist"


def slice_fixed_list_column(table: pa.Table, col_name: str, arm_slice: slice) -> pa.Table:
    """Slice a fixed_size_list column to extract a subset of elements."""
    col = table.column(col_name)
    values = col.combine_chunks()

    # Extract the raw flat array and reshape
    flat = values.values.to_numpy(zero_copy_only=False)
    original_dim = values.type.list_size
    n_rows = len(values)
    matrix = flat.reshape(n_rows, original_dim)

    # Slice to right arm
    sliced = matrix[:, arm_slice].copy()
    new_dim = sliced.shape[1]

    # Rebuild as fixed_size_list
    flat_arr = pa.array(sliced.ravel(), type=pa.float32())
    new_col = pa.FixedSizeListArray.from_arrays(flat_arr, new_dim)

    idx = table.column_names.index(col_name)
    table = table.set_column(idx, col_name, new_col)
    return table


def process_data_parquet(src_dir: Path, dst_dir: Path):
    """Process all data parquet files: slice action and state to right arm."""
    src_data = src_dir / "data"
    dst_data = dst_dir / "data"

    for parquet_file in sorted(src_data.rglob("*.parquet")):
        rel = parquet_file.relative_to(src_data)
        dst_file = dst_data / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        table = pq.read_table(parquet_file)
        table = slice_fixed_list_column(table, "action", RIGHT_ARM_SLICE)
        table = slice_fixed_list_column(table, "observation.state", RIGHT_ARM_SLICE)
        pq.write_table(table, dst_file)

    print(f"  Processed data parquets: action/state sliced to 7-dim (right arm)")


def copy_videos(src_dir: Path, dst_dir: Path):
    """Copy only cam_high and cam_right_wrist video directories."""
    src_videos = src_dir / "videos"
    dst_videos = dst_dir / "videos"

    for cam_key in KEEP_CAMERAS:
        src_cam = src_videos / cam_key
        dst_cam = dst_videos / cam_key
        if src_cam.exists():
            shutil.copytree(src_cam, dst_cam, dirs_exist_ok=True)
            print(f"  Copied video: {cam_key}")

    # Verify we skipped the left wrist
    skipped = src_videos / DROP_CAMERA
    if skipped.exists():
        print(f"  Skipped video: {DROP_CAMERA}")


def copy_meta(src_dir: Path, dst_dir: Path):
    """Copy episode and task metadata as-is."""
    src_meta = src_dir / "meta"
    dst_meta = dst_dir / "meta"
    dst_meta.mkdir(parents=True, exist_ok=True)

    # Copy episodes parquet
    episodes_dir = src_meta / "episodes"
    if episodes_dir.exists():
        shutil.copytree(episodes_dir, dst_meta / "episodes", dirs_exist_ok=True)

    # Copy tasks parquet
    tasks_file = src_meta / "tasks.parquet"
    if tasks_file.exists():
        shutil.copy2(tasks_file, dst_meta / "tasks.parquet")

    print("  Copied episode and task metadata")


def build_info_json(src_dir: Path, dst_dir: Path):
    """Build updated info.json with 7-dim features and only 2 cameras."""
    with open(src_dir / "meta" / "info.json") as f:
        info = json.load(f)

    # Update features for 7-dim action/state
    for key in ["action", "observation.state"]:
        info["features"][key]["shape"] = [7]
        info["features"][key]["names"] = RIGHT_ARM_JOINT_NAMES

    # Remove left wrist camera
    info["features"].pop(DROP_CAMERA, None)

    # Update robot type to reflect single arm
    info["robot_type"] = "single_arm_widowxai"

    dst_meta = dst_dir / "meta"
    dst_meta.mkdir(parents=True, exist_ok=True)
    with open(dst_meta / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    print("  Updated info.json: 7-dim features, 2 cameras, single_arm_widowxai")


def compute_stats(dst_dir: Path):
    """Recompute normalization stats for the processed single-arm dataset."""
    stats = {}

    # Compute stats for action and state from parquet data
    all_actions = []
    all_states = []
    for parquet_file in sorted((dst_dir / "data").rglob("*.parquet")):
        table = pq.read_table(parquet_file)

        actions_col = table.column("action")
        flat = actions_col.combine_chunks().values.to_numpy(zero_copy_only=False)
        all_actions.append(flat.reshape(-1, 7))

        states_col = table.column("observation.state")
        flat = states_col.combine_chunks().values.to_numpy(zero_copy_only=False)
        all_states.append(flat.reshape(-1, 7))

    all_actions = np.concatenate(all_actions, axis=0)
    all_states = np.concatenate(all_states, axis=0)

    for key, data in [("action", all_actions), ("observation.state", all_states)]:
        stats[key] = {
            "min": data.min(axis=0).tolist(),
            "max": data.max(axis=0).tolist(),
            "mean": data.mean(axis=0).tolist(),
            "std": data.std(axis=0).tolist(),
            "q01": np.quantile(data, 0.01, axis=0).tolist(),
            "q99": np.quantile(data, 0.99, axis=0).tolist(),
        }

    # Copy image stats from source (unchanged by slicing)
    raw_dir = dst_dir.parent.parent / "raw"  # experiments/data/raw (sibling of processed/)
    src_stats_file = raw_dir / dst_dir.name.replace("-right", "") / "meta" / "stats.json"
    if not src_stats_file.exists() and raw_dir.exists():
        for raw_link in raw_dir.iterdir():
            candidate = raw_link / "meta" / "stats.json"
            if candidate.exists():
                src_stats_file = candidate
                break

    if src_stats_file.exists():
        with open(src_stats_file) as f:
            orig_stats = json.load(f)
        for cam_key in KEEP_CAMERAS:
            if cam_key in orig_stats:
                stats[cam_key] = orig_stats[cam_key]

    with open(dst_dir / "meta" / "stats.json", "w") as f:
        json.dump(stats, f, indent=4)

    print(f"  Computed stats: action/state 7-dim (mean, std, q01, q99)")


def process_dataset(src_dir: Path, dst_dir: Path):
    """Process a single bimanual dataset into single right-arm format."""
    print(f"\nProcessing: {src_dir.name} -> {dst_dir.name}")
    dst_dir.mkdir(parents=True, exist_ok=True)

    process_data_parquet(src_dir, dst_dir)
    copy_videos(src_dir, dst_dir)
    copy_meta(src_dir, dst_dir)
    build_info_json(src_dir, dst_dir)
    compute_stats(dst_dir)

    print(f"  Done: {dst_dir}")


def merge_datasets(src_dirs: list[Path], dst_dir: Path):
    """Merge multiple processed single-arm datasets into one combined dataset."""
    print(f"\nMerging {len(src_dirs)} datasets into {dst_dir.name}")
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Merge data parquets: re-index episodes and frames across all datasets
    all_tables = []
    episode_offset = 0
    frame_offset = 0

    for src in src_dirs:
        for pf in sorted((src / "data").rglob("*.parquet")):
            table = pq.read_table(pf)
            df = table.to_pandas()
            df["episode_index"] += episode_offset
            df["index"] += frame_offset
            all_tables.append(pa.Table.from_pandas(df, preserve_index=False))

        # Count episodes and frames in this dataset
        info_path = src / "meta" / "info.json"
        with open(info_path) as f:
            info = json.load(f)
        episode_offset += info["total_episodes"]
        frame_offset += info["total_frames"]

    merged_table = pa.concat_tables(all_tables)
    data_dir = dst_dir / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(merged_table, data_dir / "file-000.parquet")
    print(f"  Merged data: {len(merged_table)} frames, {episode_offset} episodes")

    # Merge episode metadata
    episode_tables = []
    for src in src_dirs:
        for pf in sorted((src / "meta" / "episodes").rglob("*.parquet")):
            episode_tables.append(pq.read_table(pf))
    if episode_tables:
        merged_episodes = pa.concat_tables(episode_tables)
        ep_dir = dst_dir / "meta" / "episodes" / "chunk-000"
        ep_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(merged_episodes, ep_dir / "file-000.parquet")

    # Copy tasks from first dataset (same task across all)
    first_tasks = src_dirs[0] / "meta" / "tasks.parquet"
    if first_tasks.exists():
        meta_dir = dst_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(first_tasks, meta_dir / "tasks.parquet")

    # Merge videos: copy all video files with episode-based naming
    for cam_key in KEEP_CAMERAS:
        for src in src_dirs:
            src_cam = src / "videos" / cam_key
            dst_cam = dst_dir / "videos" / cam_key / "chunk-000"
            dst_cam.mkdir(parents=True, exist_ok=True)
            if src_cam.exists():
                for mp4 in sorted(src_cam.rglob("*.mp4")):
                    # Use source dataset name as prefix to avoid collisions
                    prefix = src.name.replace("-right", "")
                    dst_file = dst_cam / f"{prefix}_{mp4.name}"
                    shutil.copy2(mp4, dst_file)

    # Build merged info.json
    with open(src_dirs[0] / "meta" / "info.json") as f:
        info = json.load(f)
    info["total_episodes"] = episode_offset
    info["total_frames"] = frame_offset
    info["splits"] = {"train": f"0:{episode_offset}"}
    with open(dst_dir / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    # Compute merged stats
    compute_stats(dst_dir)
    print(f"  Merged dataset: {dst_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract right arm from bimanual ALOHA LeRobot datasets"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(__file__).parent / "raw",
        help="Directory containing raw bimanual dataset symlinks",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent / "processed",
        help="Output directory for processed single-arm datasets",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Dataset names to process (e.g. marker_pick marker_pick1 marker_pick2)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge all processed datasets into a single combined dataset",
    )
    parser.add_argument(
        "--merge-name",
        type=str,
        default=None,
        help="Name of the merged dataset (default: first_dataset-all-right)",
    )
    args = parser.parse_args()

    processed_dirs = []
    for name in args.datasets:
        src = args.raw_dir / name
        if not src.exists():
            print(f"WARNING: {src} not found, skipping")
            continue
        dst = args.out_dir / f"{name}-right"
        process_dataset(src.resolve(), dst)
        processed_dirs.append(dst)

    if args.merge and len(processed_dirs) > 1:
        merge_name = args.merge_name or f"{args.datasets[0]}-all-right"
        merged_dir = args.out_dir / merge_name
        merge_datasets(processed_dirs, merged_dir)

    print(f"\nAll datasets processed. Output: {args.out_dir}")
    print("Set LEROBOT_HOME to the processed directory for training.")


if __name__ == "__main__":
    main()
