#!/usr/bin/env python3
"""Convert recorder RGB-D kinesthetic Coke demonstrations into LeRobot splits.

The source recordings contain exact applied upper-body Teach commands, but the
Dex3 values are measured observations.  The historical ``arm14`` contract
exports waist, both arms, and the observed hand.  The ``left-arm7`` contract
strictly removes the right arm from both observation and action: its 17D state
is waist, left arm, and observed physical-left hand; its 7D action is the exact
applied left-arm target.  Neither contract relabels observed hand motion as a
policy action.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
from typing import Any

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from PIL import Image

BATCH_SCHEMA = "wendy.g1.mujoco-trial-batch.v1"
TRIAL_SCHEMA_VERSION = 4
OUTPUT_FPS = 15
BODY_DIM = 17
HAND_DIM = 7
IMAGE_HEIGHT = 240
IMAGE_WIDTH = 320
# One source episode contains a single roughly 0.26 s camera dropout.  Offline
# resampling may repeat its nearest real RGB-D pair across that gap, while the
# original camera-to-robot synchronization gate remains 20 ms and live
# inference retains its independent stale-frame rejection.
MAX_CAMERA_GRID_SKEW_S = 0.135
MAX_ROBOT_CAMERA_SYNC_SKEW_MS = 20.0
MAX_ABS_WAIST_ROLL_PITCH_RAD = math.radians(5.0)
NEAR_DEPTH_M = 0.20
FAR_DEPTH_M = 2.00
TASK = "Grasp the Coke can, lift it, and present it in front of the robot."
DEFAULT_HOLDOUT_NAMES = ("RCokeGrabbing4", "RCokeGrabbing9", "RCokeGrabbing15")

BODY_JOINT_NAMES = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
LEFT_HAND_JOINT_NAMES = (
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
)


@dataclass(frozen=True)
class TaskContract:
    name: str
    output_schema: str
    state_body_indices: tuple[int, ...]
    action_body_indices: tuple[int, ...]
    state_names: tuple[str, ...]
    action_names: tuple[str, ...]

    @property
    def state_dim(self) -> int:
        return len(self.state_names)

    @property
    def action_dim(self) -> int:
        return len(self.action_names)


ARM14_CONTRACT = TaskContract(
    name="arm14",
    output_schema="wendy.g1.coke-rgbd-arm14-conversion.v1",
    state_body_indices=tuple(range(BODY_DIM)),
    action_body_indices=tuple(range(3, BODY_DIM)),
    state_names=BODY_JOINT_NAMES + LEFT_HAND_JOINT_NAMES,
    action_names=BODY_JOINT_NAMES[3:],
)
LEFT_ARM7_CONTRACT = TaskContract(
    name="left-arm7",
    output_schema="wendy.g1.coke-rgbd-left-arm7-conversion.v1",
    state_body_indices=tuple(range(10)),
    action_body_indices=tuple(range(3, 10)),
    state_names=BODY_JOINT_NAMES[:10] + LEFT_HAND_JOINT_NAMES,
    action_names=BODY_JOINT_NAMES[3:10],
)
CONTRACTS = {contract.name: contract for contract in (ARM14_CONTRACT, LEFT_ARM7_CONTRACT)}


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
    parser.add_argument("--train-repo-id", type=_repo_id, required=True)
    parser.add_argument("--eval-repo-id", type=_repo_id, required=True)
    parser.add_argument(
        "--contract",
        choices=tuple(CONTRACTS),
        default=ARM14_CONTRACT.name,
        help="Explicit state/action embodiment contract; use left-arm7 for the physical-left task",
    )
    parser.add_argument(
        "--holdout-name",
        action="append",
        dest="holdout_names",
        help="Complete source episode name reserved for evaluation; repeat for multiple episodes",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _finite_vector(value: Any, length: int, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    if vector.shape != (length,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must be one finite {length}-value vector")
    return vector


def _load_batch(raw_dir: Path) -> dict[str, Any]:
    manifest_path = raw_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != BATCH_SCHEMA:
        raise ValueError(f"unsupported recorder batch schema {manifest.get('schema')!r}")
    if manifest.get("read_only_gets_only") is not True or manifest.get("robot_commands_sent") != 0:
        raise ValueError("source batch does not preserve its read-only pull provenance")
    trials = manifest.get("trials")
    if not isinstance(trials, list) or not trials:
        raise ValueError("recorder batch contains no trials")
    return manifest


def _load_trial(raw_dir: Path, entry: dict[str, Any]) -> dict[str, Any]:
    path = raw_dir / str(entry["file"])
    if _sha256(path) != entry.get("sha256"):
        raise ValueError(f"{path} does not match its recorded SHA-256")
    trial = json.loads(path.read_text(encoding="utf-8"))
    name = trial.get("name") or trial.get("id")
    if trial.get("schema_version") != TRIAL_SCHEMA_VERSION:
        raise ValueError(f"{name} is not recorder schema v{TRIAL_SCHEMA_VERSION}")
    if trial.get("id") != entry.get("id") or trial.get("name") != entry.get("name"):
        raise ValueError(f"{path} identity does not match the batch manifest")
    if trial.get("sample_hz") != 25.0 or trial.get("joint_count") != BODY_DIM:
        raise ValueError(f"{name} does not use the expected 25 Hz 17D body contract")

    hand = trial.get("hand")
    if not isinstance(hand, dict):
        raise ValueError(f"{name} is missing its hand contract")
    if (
        hand.get("semantic_hand") != "physical_left"
        or hand.get("unitree_socket") != "left"
        or hand.get("joint_count") != HAND_DIM
    ):
        raise ValueError(f"{name} does not use the physical-left Dex3 on the left socket")
    if hand.get("policy_action_eligible") is not False:
        raise ValueError(f"{name} unexpectedly marks measured hand pose as a policy action")

    frames = trial.get("frames")
    if not isinstance(frames, list) or len(frames) != trial.get("frame_count"):
        raise ValueError(f"{name} has a mismatched robot frame count")
    timestamps = [float(frame["t_s"]) for frame in frames]
    if any(not math.isfinite(value) for value in timestamps) or any(
        right <= left for left, right in itertools.pairwise(timestamps)
    ):
        raise ValueError(f"{name} robot timestamps are not finite and strictly increasing")
    for frame_index, frame in enumerate(frames):
        body_state = _finite_vector(
            frame.get("measured_q_rad"), BODY_DIM, f"{name} frame {frame_index} body state"
        )
        if float(np.max(np.abs(body_state[1:3]))) > MAX_ABS_WAIST_ROLL_PITCH_RAD:
            raise ValueError(f"{name} frame {frame_index} is not an upright waist observation")
        hand_frame = frame.get("hand")
        if not isinstance(hand_frame, dict):
            raise ValueError(f"{name} frame {frame_index} is missing measured hand state")
        _finite_vector(
            hand_frame.get("measured_q_rad"),
            HAND_DIM,
            f"{name} frame {frame_index} hand state",
        )
        teach = frame.get("teach_command")
        if not isinstance(teach, dict) or teach.get("topic") != "rt/arm_sdk":
            raise ValueError(f"{name} frame {frame_index} lacks an applied rt/arm_sdk Teach command")
        _finite_vector(teach.get("q_rad"), BODY_DIM, f"{name} frame {frame_index} Teach command")

    video = trial.get("video")
    depth = trial.get("depth")
    if not isinstance(video, dict) or not isinstance(depth, dict):
        raise ValueError(f"{name} does not contain RGB-D metadata")
    if video.get("encoding") != "jpeg_sequence" or depth.get("encoding") != "png_z16_sequence":
        raise ValueError(f"{name} uses an unsupported RGB-D encoding")
    if depth.get("aligned_to") != "color":
        raise ValueError(f"{name} depth is not aligned to RGB")
    if float(video.get("maximum_sync_skew_ms", math.inf)) > MAX_ROBOT_CAMERA_SYNC_SKEW_MS:
        raise ValueError(f"{name} RGB-to-robot synchronization exceeds 20 ms")
    if float(depth.get("maximum_sync_skew_ms", math.inf)) > MAX_ROBOT_CAMERA_SYNC_SKEW_MS:
        raise ValueError(f"{name} depth-to-robot synchronization exceeds 20 ms")
    rgb_frames = video.get("frames")
    depth_frames = depth.get("frames")
    if (
        not isinstance(rgb_frames, list)
        or not isinstance(depth_frames, list)
        or len(rgb_frames) != len(depth_frames)
        or len(rgb_frames) != video.get("frame_count")
        or len(depth_frames) != depth.get("frame_count")
    ):
        raise ValueError(f"{name} RGB and depth frame counts do not match")
    for pair_index, (rgb_frame, depth_frame) in enumerate(zip(rgb_frames, depth_frames, strict=True)):
        if (
            abs(float(rgb_frame["t_s"]) - float(depth_frame["t_s"])) > 1.0e-9
            or rgb_frame.get("source_frame_id") != depth_frame.get("source_frame_id")
            or rgb_frame.get("nearest_robot_frame_index")
            != depth_frame.get("nearest_robot_frame_index")
        ):
            raise ValueError(f"{name} RGB-D pair {pair_index} is not aligned")
        if not (raw_dir / str(rgb_frame["image"])).is_file():
            raise FileNotFoundError(raw_dir / str(rgb_frame["image"]))
        if not (raw_dir / str(depth_frame["image"])).is_file():
            raise FileNotFoundError(raw_dir / str(depth_frame["image"]))
    return trial


def _interpolate_robot(
    trial: dict[str, Any], source_time_s: float, contract: TaskContract = ARM14_CONTRACT
) -> tuple[np.ndarray, np.ndarray]:
    frames = trial["frames"]
    timestamps = [float(frame["t_s"]) for frame in frames]
    right_index = bisect_left(timestamps, source_time_s)
    if right_index == 0:
        left_index = right_index = 0
        ratio = 0.0
    elif right_index == len(frames):
        left_index = right_index = len(frames) - 1
        ratio = 0.0
    else:
        left_index = right_index - 1
        interval = timestamps[right_index] - timestamps[left_index]
        ratio = (source_time_s - timestamps[left_index]) / interval

    left = frames[left_index]
    right = frames[right_index]

    def lerp(left_value: Any, right_value: Any, length: int, label: str) -> np.ndarray:
        left_vector = _finite_vector(left_value, length, label)
        right_vector = _finite_vector(right_value, length, label)
        return left_vector + (right_vector - left_vector) * np.float32(ratio)

    body = lerp(left["measured_q_rad"], right["measured_q_rad"], BODY_DIM, "body state")
    hand = lerp(
        left["hand"]["measured_q_rad"],
        right["hand"]["measured_q_rad"],
        HAND_DIM,
        "hand state",
    )
    applied_teach = lerp(
        left["teach_command"]["q_rad"],
        right["teach_command"]["q_rad"],
        BODY_DIM,
        "Teach command",
    )
    state = np.concatenate((body[list(contract.state_body_indices)], hand)).astype(
        np.float32, copy=False
    )
    action = applied_teach[list(contract.action_body_indices)].astype(np.float32, copy=False)
    if state.shape != (contract.state_dim,) or action.shape != (contract.action_dim,):
        raise RuntimeError("internal G1 state/action slicing error")
    return state, action


def _camera_pairs(trial: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "t_s": float(rgb["t_s"]),
            "rgb": rgb["image"],
            "depth": depth["image"],
        }
        for rgb, depth in zip(trial["video"]["frames"], trial["depth"]["frames"], strict=True)
    ]


def _nearest_camera_pair(pairs: list[dict[str, Any]], source_time_s: float) -> tuple[dict[str, Any], float]:
    timestamps = [pair["t_s"] for pair in pairs]
    right_index = bisect_left(timestamps, source_time_s)
    candidates = []
    if right_index < len(pairs):
        candidates.append(pairs[right_index])
    if right_index > 0:
        candidates.append(pairs[right_index - 1])
    pair = min(candidates, key=lambda item: abs(item["t_s"] - source_time_s))
    skew_s = abs(pair["t_s"] - source_time_s)
    if skew_s > MAX_CAMERA_GRID_SKEW_S:
        raise ValueError(
            f"nearest RGB-D frame is {skew_s * 1000.0:.1f} ms from the {OUTPUT_FPS} Hz grid"
        )
    return pair, skew_s


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize(
            (IMAGE_WIDTH, IMAGE_HEIGHT),
            resample=Image.Resampling.BILINEAR,
        )
        result = np.asarray(rgb, dtype=np.uint8)
    if result.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 3):
        raise ValueError(f"invalid RGB image {path}: {result.shape}")
    return result


def _load_depth(path: Path, depth_scale_m: float) -> tuple[np.ndarray, float]:
    with Image.open(path) as image:
        raw = np.asarray(image)
    if raw.shape != (480, 640) or raw.dtype != np.uint16:
        raise ValueError(f"invalid Z16 depth image {path}: {raw.shape} {raw.dtype}")
    depth_m = raw.astype(np.float32) * np.float32(depth_scale_m)
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    valid_fraction = float(np.mean(valid))
    if valid_fraction < 0.10:
        raise ValueError(f"depth image {path} has only {valid_fraction:.3f} valid pixels")
    clipped = np.clip(depth_m, NEAR_DEPTH_M, FAR_DEPTH_M)
    inverse = (FAR_DEPTH_M - clipped) / (FAR_DEPTH_M - NEAR_DEPTH_M)
    encoded = np.where(valid, np.rint(inverse * 255.0), 0.0).astype(np.uint8)
    resized = Image.fromarray(encoded, mode="L").resize(
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        resample=Image.Resampling.NEAREST,
    )
    depth_gray = np.asarray(resized, dtype=np.uint8)
    depth_rgb = np.repeat(depth_gray[..., None], 3, axis=-1)
    return depth_rgb, valid_fraction


def _samples(
    raw_dir: Path, trial: dict[str, Any], contract: TaskContract = ARM14_CONTRACT
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    pairs = _camera_pairs(trial)
    first_time_s = pairs[0]["t_s"]
    last_time_s = min(pairs[-1]["t_s"], float(trial["frames"][-1]["t_s"]))
    count = math.floor((last_time_s - first_time_s) * OUTPUT_FPS + 1.0e-9) + 1
    if count < OUTPUT_FPS:
        raise ValueError(f"{trial['name']} is shorter than one second after RGB-D alignment")
    depth_scale_m = float(trial["depth"]["calibration"]["depth_scale_m"])
    if not 0.0 < depth_scale_m < 0.1:
        raise ValueError(f"{trial['name']} has an invalid depth scale {depth_scale_m}")

    samples = []
    maximum_camera_grid_skew_s = 0.0
    minimum_depth_valid_fraction = 1.0
    for frame_index in range(count):
        source_time_s = first_time_s + frame_index / OUTPUT_FPS
        pair, camera_grid_skew_s = _nearest_camera_pair(pairs, source_time_s)
        maximum_camera_grid_skew_s = max(maximum_camera_grid_skew_s, camera_grid_skew_s)
        state, action = _interpolate_robot(trial, source_time_s, contract)
        rgb = _load_rgb(raw_dir / pair["rgb"])
        depth, valid_fraction = _load_depth(raw_dir / pair["depth"], depth_scale_m)
        minimum_depth_valid_fraction = min(minimum_depth_valid_fraction, valid_fraction)
        samples.append(
            {
                "observation.images.head": rgb,
                "observation.images.depth": depth,
                "observation.state": state,
                "action": action,
                "task": TASK,
            }
        )
    return samples, {
        "maximum_camera_grid_skew_ms": maximum_camera_grid_skew_s * 1000.0,
        "minimum_depth_valid_fraction": minimum_depth_valid_fraction,
    }


def _create_dataset(repo_id: str, contract: TaskContract, *, overwrite: bool) -> LeRobotDataset:
    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")
        shutil.rmtree(output_path)
    return LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="unitree_g1_upper_body_physical_left_dex3",
        fps=OUTPUT_FPS,
        features={
            "observation.images.head": {
                "dtype": "image",
                "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.depth": {
                "dtype": "image",
                "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (contract.state_dim,),
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (contract.action_dim,),
                "names": ["action"],
            },
        },
        use_videos=True,
        image_writer_threads=8,
        image_writer_processes=0,
    )


def _write_conversion_metadata(repo_id: str, metadata: dict[str, Any]) -> None:
    path = HF_LEROBOT_HOME / repo_id / "meta" / "wendy-rgbd-conversion.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    contract = CONTRACTS[args.contract]
    raw_dir = args.raw_dir.expanduser().resolve()
    if args.train_repo_id == args.eval_repo_id:
        raise ValueError("train and held-out evaluation repo ids must differ")
    if contract is LEFT_ARM7_CONTRACT and any(
        "left_only" not in repo_id for repo_id in (args.train_repo_id, args.eval_repo_id)
    ):
        raise ValueError("left-arm7 datasets must use distinct repo ids containing 'left_only'")
    manifest = _load_batch(raw_dir)
    holdout_names = set(args.holdout_names or DEFAULT_HOLDOUT_NAMES)
    available_names = {str(entry["name"]) for entry in manifest["trials"]}
    missing_holdouts = holdout_names - available_names
    if missing_holdouts:
        raise ValueError(f"holdout episodes are absent: {sorted(missing_holdouts)}")
    if len(holdout_names) >= len(available_names):
        raise ValueError("holdout split leaves no training episodes")

    train_dataset = _create_dataset(args.train_repo_id, contract, overwrite=args.overwrite)
    eval_dataset = _create_dataset(args.eval_repo_id, contract, overwrite=args.overwrite)
    split_records: dict[str, list[dict[str, Any]]] = {"train": [], "eval": []}
    for entry in manifest["trials"]:
        trial = _load_trial(raw_dir, entry)
        split = "eval" if trial["name"] in holdout_names else "train"
        samples, quality = _samples(raw_dir, trial, contract)
        dataset = eval_dataset if split == "eval" else train_dataset
        for sample in samples:
            dataset.add_frame(sample)
        dataset.save_episode()
        split_records[split].append(
            {
                "id": trial["id"],
                "name": trial["name"],
                "source_json": entry["file"],
                "source_json_sha256": entry["sha256"],
                "frames": len(samples),
                **quality,
            }
        )

    metadata = {
        "schema": contract.output_schema,
        "contract": contract.name,
        "source_batch_manifest_sha256": _sha256(raw_dir / "manifest.json"),
        "fps": OUTPUT_FPS,
        "task": TASK,
        "image": {
            "rgb": "320x240 uint8 RGB",
            "depth": {
                "representation": "three-channel uint8 inverse depth",
                "near_m": NEAR_DEPTH_M,
                "far_m": FAR_DEPTH_M,
                "invalid_value": 0,
            },
        },
        "state": {
            "width": contract.state_dim,
            "names": list(contract.state_names),
            "source": (
                "measured waist and left arm plus measured physical-left Dex3"
                if contract is LEFT_ARM7_CONTRACT
                else "17D measured upper body plus 7D measured physical-left Dex3"
            ),
            "right_arm_included": contract is ARM14_CONTRACT,
        },
        "action": {
            "width": contract.action_dim,
            "names": list(contract.action_names),
            "source": (
                "applied rt/arm_sdk Teach q_rad indices 3 through 9"
                if contract is LEFT_ARM7_CONTRACT
                else "applied rt/arm_sdk Teach q_rad indices 3 through 16"
            ),
            "waist_excluded": True,
            "right_arm_included": contract is ARM14_CONTRACT,
            "measured_hand_as_action": False,
        },
        "splits": split_records,
    }
    _write_conversion_metadata(args.train_repo_id, metadata)
    _write_conversion_metadata(args.eval_repo_id, metadata)
    print(
        json.dumps(
            {
                "status": "passed",
                "contract": contract.name,
                "train_repo_id": args.train_repo_id,
                "eval_repo_id": args.eval_repo_id,
                "train_episodes": len(split_records["train"]),
                "eval_episodes": len(split_records["eval"]),
                "train_frames": sum(record["frames"] for record in split_records["train"]),
                "eval_frames": sum(record["frames"] for record in split_records["eval"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
