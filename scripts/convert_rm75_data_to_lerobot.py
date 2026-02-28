"""Convert RM75 HDF5 robot demonstration data to LeRobot dataset format.

This script reads HDF5 episode files recorded from the RM75 7-DoF arm platform
and writes them into a LeRobot dataset compatible with the openpi training pipeline
and the `LeRobotRM75DataConfig` data config factory.

NOTE ON FEATURE KEY NAMING
---------------------------
LeRobot (v0.1.0) prohibits "/" in feature key names (it uses "/" as a separator for
its stats computation).  We therefore store dataset keys using dot notation:

    observation.wrist_image        (instead of  observation/wrist_image)
    observation.joint_position     (instead of  observation/joint_position)
    observation.joint_velocity     (instead of  observation/joint_velocity)
    observation.gripper_position   (instead of  observation/gripper_position)
    actions

The `LeRobotRM75DataConfig.repack_transform` in `src/openpi/research/shared/rm75_policy.py`
maps these dataset keys to the slash-namespace expected by the training pipeline.
If the RepackTransform currently uses an identity mapping, update it to:

    {
        "observation/wrist_image":      "observation.wrist_image",
        "observation/joint_position":   "observation.joint_position",
        "observation/joint_velocity":   "observation.joint_velocity",
        "observation/gripper_position": "observation.gripper_position",
        "actions":                      "actions",
        "prompt":                       "prompt",
    }

Usage:
    uv run scripts/convert_rm75_data_to_lerobot.py \\
        --raw-dir /path/to/hdf5/episodes \\
        --repo-id myorg/spacecil_payload \\
        --task-label "pick up the object"

    # Dry run — inspect schema without writing:
    uv run scripts/convert_rm75_data_to_lerobot.py \\
        --raw-dir /path/to/hdf5/episodes \\
        --repo-id myorg/spacecil_payload \\
        --task-label "pick up the object" \\
        --dry-run

    # Custom HDF5 key names:
    uv run scripts/convert_rm75_data_to_lerobot.py \\
        --raw-dir /path/to/hdf5/episodes \\
        --repo-id myorg/spacecil_payload \\
        --task-label "pick up the object" \\
        --key-map '{"wrist_image": "images/wrist_rgb"}'

HDF5 Input Schema (default key names, overridable via --key-map):
    joint_position    (N, 7)          float32
    joint_velocity    (N, 7)          float32
    gripper_position  (N, 1)          float32
    wrist_image       (N, H, W, 3)    uint8
    action_joint      (N, 7)          float32
    action_gripper    (N, 1)          float32

LeRobot Output Features (dot-separated to comply with LeRobot v0.1.0 constraint):
    observation.wrist_image       image   (H, W, 3)    — uint8 wrist camera RGB
    observation.joint_position    float32 (7,)          — absolute joint angles (rad)
    observation.joint_velocity    float32 (7,)          — joint velocities (rad/s)
    observation.gripper_position  float32 (1,)          — gripper opening (normalised)
    actions                       float32 (8,)          — concat(action_joint, action_gripper)
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Default HDF5 key mapping (RM75 platform convention)
# Maps logical field name → HDF5 dataset path inside the file
# ---------------------------------------------------------------------------
DEFAULT_KEY_MAP: dict[str, str] = {
    "joint_position": "joint_position",
    "joint_velocity": "joint_velocity",
    "gripper_position": "gripper_position",
    "wrist_image": "wrist_image",
    "action_joint": "action_joint",
    "action_gripper": "action_gripper",
}

# LeRobot dataset keys (dot-separated — LeRobot v0.1.0 disallows "/" in keys).
# The RepackTransform in LeRobotRM75DataConfig should map these to "observation/..." namespace.
LEROBOT_KEY_WRIST_IMAGE = "observation.wrist_image"
LEROBOT_KEY_JOINT_POS = "observation.joint_position"
LEROBOT_KEY_JOINT_VEL = "observation.joint_velocity"
LEROBOT_KEY_GRIPPER_POS = "observation.gripper_position"
LEROBOT_KEY_ACTIONS = "actions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RM75 HDF5 episode files to LeRobot dataset format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--raw-dir",
        required=True,
        type=Path,
        help="Directory containing HDF5 episode files (*.hdf5 or *.h5).",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        type=str,
        help="LeRobot dataset repository ID (e.g. myorg/spacecil_payload).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Recording frequency in frames per second (default: 30).",
    )
    parser.add_argument(
        "--task-label",
        required=True,
        type=str,
        help="Language instruction string describing the task.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read the first episode and print the schema without writing anything.",
    )
    parser.add_argument(
        "--key-map",
        type=str,
        default=None,
        help=(
            "JSON string to override default HDF5 key names. "
            "Only the keys you want to override need to be specified. "
            'Example: \'{"wrist_image": "images/wrist_rgb"}\''
        ),
    )
    return parser.parse_args()


def discover_hdf5_files(raw_dir: Path) -> list[Path]:
    """Discover all HDF5 files in raw_dir, sorted by name."""
    hdf5 = sorted(raw_dir.glob("*.hdf5"))
    h5 = sorted(raw_dir.glob("*.h5"))
    # Combine, de-duplicate, re-sort by name
    seen: set[Path] = set()
    result: list[Path] = []
    for f in hdf5 + h5:
        if f not in seen:
            seen.add(f)
            result.append(f)
    return sorted(result)


def load_episode(ep_path: Path, key_map: dict[str, str]) -> dict[str, np.ndarray]:
    """Load all arrays from an HDF5 episode file using the given key mapping."""
    with h5py.File(ep_path, "r") as f:
        data = {
            "joint_position": f[key_map["joint_position"]][:],
            "joint_velocity": f[key_map["joint_velocity"]][:],
            "gripper_position": f[key_map["gripper_position"]][:],
            "wrist_image": f[key_map["wrist_image"]][:],
            "action_joint": f[key_map["action_joint"]][:],
            "action_gripper": f[key_map["action_gripper"]][:],
        }
    return data


def print_dry_run_schema(ep_path: Path, key_map: dict[str, str]) -> None:
    """Print dataset schema from the first episode without writing anything."""
    print(f"\n=== DRY RUN: Schema inspection of {ep_path} ===\n")
    data = load_episode(ep_path, key_map)

    print("HDF5 Input Arrays:")
    print(f"  {'Logical key':<25} {'HDF5 key':<30} {'Shape':<20} {'Dtype'}")
    print(f"  {'-' * 25} {'-' * 30} {'-' * 20} {'-' * 10}")
    for logical_key, hdf5_key in key_map.items():
        arr = data[logical_key]
        print(f"  {logical_key:<25} {hdf5_key:<30} {str(arr.shape):<20} {arr.dtype}")

    # Compute derived shapes
    n_frames = data["joint_position"].shape[0]
    img_shape = data["wrist_image"].shape[1:]  # (H, W, 3)
    action_dim = data["action_joint"].shape[-1] + data["action_gripper"].shape[-1]

    print(f"\nEpisode length: {n_frames} frames")
    print()
    print("LeRobot Output Features (will be written):")
    print(f"  {'Feature Key':<40} {'Dtype':<10} {'Shape'}")
    print(f"  {'-' * 40} {'-' * 10} {'-' * 20}")
    features_preview = [
        (LEROBOT_KEY_WRIST_IMAGE, "image", str(img_shape)),
        (LEROBOT_KEY_JOINT_POS, "float32", str(data["joint_position"].shape[1:])),
        (LEROBOT_KEY_JOINT_VEL, "float32", str(data["joint_velocity"].shape[1:])),
        (LEROBOT_KEY_GRIPPER_POS, "float32", str(data["gripper_position"].shape[1:])),
        (LEROBOT_KEY_ACTIONS, "float32", f"({action_dim},)"),
    ]
    for key, dtype, shape in features_preview:
        print(f"  {key:<40} {dtype:<10} {shape}")

    print()
    print("NOTE: Keys use dot-notation ('observation.X') because LeRobot v0.1.0 disallows")
    print("      '/' in feature names.  The RepackTransform in LeRobotRM75DataConfig must")
    print("      map 'observation.X' → 'observation/X' for the training pipeline.")
    print("\n[DRY RUN] No files written. Remove --dry-run to convert.\n")


def build_features(img_shape: tuple[int, int, int]) -> dict:
    """Build the LeRobot feature spec for an RM75 dataset.

    Uses dot-separated keys because LeRobot v0.1.0 prohibits '/' in feature names.
    The RepackTransform in LeRobotRM75DataConfig should remap these to the
    'observation/...' namespace that the training pipeline and policy inputs expect.
    """
    h, w, c = img_shape
    return {
        LEROBOT_KEY_WRIST_IMAGE: {
            "dtype": "image",
            "shape": (h, w, c),
            "names": ["height", "width", "channel"],
        },
        LEROBOT_KEY_JOINT_POS: {
            "dtype": "float32",
            "shape": (7,),
            "names": ["joint_position"],
        },
        LEROBOT_KEY_JOINT_VEL: {
            "dtype": "float32",
            "shape": (7,),
            "names": ["joint_velocity"],
        },
        LEROBOT_KEY_GRIPPER_POS: {
            "dtype": "float32",
            "shape": (1,),
            "names": ["gripper_position"],
        },
        LEROBOT_KEY_ACTIONS: {
            "dtype": "float32",
            "shape": (8,),
            "names": ["actions"],
        },
    }


def convert(
    raw_dir: Path,
    repo_id: str,
    fps: int,
    task_label: str,
    key_map: dict[str, str],
) -> None:
    """Main conversion routine: HDF5 files → LeRobot dataset."""
    # Late import: only needed for actual conversion (not --dry-run or --help)
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

    hdf5_files = discover_hdf5_files(raw_dir)
    if not hdf5_files:
        print(f"ERROR: No HDF5 files found in {raw_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(hdf5_files)} episode file(s) in {raw_dir}")

    # Read image shape from first frame of first episode
    first_data = load_episode(hdf5_files[0], key_map)
    img_shape: tuple[int, int, int] = first_data["wrist_image"].shape[1:]  # (H, W, 3)
    print(f"Image shape: {img_shape}  (H, W, C)")

    features = build_features(img_shape)

    # Clean up any existing dataset at the output path
    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)

    # Create the LeRobot dataset
    print(f"Creating LeRobot dataset: {repo_id}  (fps={fps})")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type="rm75",
        features=features,
        use_videos=False,
    )

    # Populate episodes
    for ep_idx, ep_path in enumerate(hdf5_files):
        print(f"  [{ep_idx + 1}/{len(hdf5_files)}] Processing {ep_path.name} ...", end=" ", flush=True)

        ep_data = load_episode(ep_path, key_map)
        n_frames: int = ep_data["joint_position"].shape[0]

        # Validate consistent lengths across all arrays
        for logical_key in (
            "joint_velocity",
            "gripper_position",
            "wrist_image",
            "action_joint",
            "action_gripper",
        ):
            if ep_data[logical_key].shape[0] != n_frames:
                raise ValueError(
                    f"Inconsistent frame counts in {ep_path}: "
                    f"joint_position has {n_frames} frames but "
                    f"{logical_key} has {ep_data[logical_key].shape[0]} frames."
                )

        for i in range(n_frames):
            # Concatenate 7D joint action + 1D gripper action → 8D
            action_vec = np.concatenate(
                [
                    ep_data["action_joint"][i],
                    ep_data["action_gripper"][i],
                ],
                axis=0,
            ).astype(np.float32)

            frame = {
                # Images are stored as uint8 (H, W, 3) — LeRobot handles PNG encoding.
                LEROBOT_KEY_WRIST_IMAGE: ep_data["wrist_image"][i],
                LEROBOT_KEY_JOINT_POS: ep_data["joint_position"][i].astype(np.float32),
                LEROBOT_KEY_JOINT_VEL: ep_data["joint_velocity"][i].astype(np.float32),
                LEROBOT_KEY_GRIPPER_POS: ep_data["gripper_position"][i].astype(np.float32),
                LEROBOT_KEY_ACTIONS: action_vec,
                # 'task' is required by LeRobot v0.1.0 add_frame validation.
                "task": task_label,
            }
            dataset.add_frame(frame)

        dataset.save_episode()
        print(f"{n_frames} frames saved.")

    # Finalize the dataset.
    # (LeRobot v0.1.0 persists data incrementally via save_episode().
    #  No separate consolidate() step is required in this version.)

    # Verification: reload and print summary
    print("\n=== Verification: Reloading dataset ===")
    loaded = LeRobotDataset(repo_id=repo_id)
    print(f"  Episodes   : {loaded.num_episodes}")
    print(f"  Frames     : {loaded.num_frames}")
    print(f"  FPS        : {loaded.fps}")
    print(f"  Features   : {list(loaded.features.keys())}")
    print(f"  Saved to   : {output_path}")
    print("\nConversion complete.\n")


def main() -> None:
    args = parse_args()

    # Build key map: start from defaults, overlay user overrides
    key_map = dict(DEFAULT_KEY_MAP)
    if args.key_map is not None:
        try:
            overrides = json.loads(args.key_map)
        except json.JSONDecodeError as exc:
            print(f"ERROR: --key-map is not valid JSON: {exc}", file=sys.stderr)
            sys.exit(1)
        if not isinstance(overrides, dict):
            print("ERROR: --key-map must be a JSON object (dict).", file=sys.stderr)
            sys.exit(1)
        key_map.update(overrides)

    # Validate raw-dir
    if not args.raw_dir.exists():
        print(f"ERROR: --raw-dir does not exist: {args.raw_dir}", file=sys.stderr)
        sys.exit(1)
    if not args.raw_dir.is_dir():
        print(f"ERROR: --raw-dir is not a directory: {args.raw_dir}", file=sys.stderr)
        sys.exit(1)

    hdf5_files = discover_hdf5_files(args.raw_dir)
    if not hdf5_files:
        print(f"ERROR: No HDF5 files found in {args.raw_dir}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print_dry_run_schema(hdf5_files[0], key_map)
        return

    convert(
        raw_dir=args.raw_dir,
        repo_id=args.repo_id,
        fps=args.fps,
        task_label=args.task_label,
        key_map=key_map,
    )


if __name__ == "__main__":
    main()
