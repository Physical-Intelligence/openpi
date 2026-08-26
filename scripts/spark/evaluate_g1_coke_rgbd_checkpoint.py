#!/usr/bin/env python3
"""Evaluate a G1 RGB-D LoRA checkpoint on held-out demonstrations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import av
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from openpi.policies import policy_config
from openpi.training import config as training_config

PROMPT = "grasp the Coke can, lift it, and present it"
ARM_MIN_RAD = np.asarray(
    [
        -3.0892,
        -1.5882,
        -2.618,
        -1.0472,
        -1.97222,
        -1.61443,
        -1.61443,
        -3.0892,
        -2.2515,
        -2.618,
        -1.0472,
        -1.97222,
        -1.61443,
        -1.61443,
    ],
    dtype=np.float32,
)
ARM_MAX_RAD = np.asarray(
    [
        2.6704,
        2.2515,
        2.618,
        2.0944,
        1.97222,
        1.61443,
        1.61443,
        2.6704,
        1.5882,
        2.618,
        2.0944,
        1.97222,
        1.61443,
        1.61443,
    ],
    dtype=np.float32,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _image(value) -> np.ndarray:
    image = _numpy(value)
    if image.ndim != 3:
        raise ValueError(f"expected one image, got {image.shape}")
    if image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0, 255)
    image = image.astype(np.uint8)
    if image.shape[-1] != 3:
        raise ValueError(f"expected three image channels, got {image.shape}")
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--config-name", default="pi05_spark_g1_coke_rgbd_arm14")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--denoise-steps", type=int, default=5)
    args = parser.parse_args()
    if args.samples < 1 or args.denoise_steps < 1:
        raise ValueError("samples and denoise steps must be positive")

    metadata = LeRobotDatasetMetadata(args.repo_id)
    dataset = LeRobotDataset(
        args.repo_id,
        delta_timestamps={"action": [step / metadata.fps for step in range(10)]},
    )
    if len(dataset) < 1:
        raise ValueError("held-out dataset is empty")

    config = training_config.get_config(args.config_name)
    policy = policy_config.create_trained_policy(
        config,
        args.checkpoint_dir,
        default_prompt=PROMPT,
        sample_kwargs={"num_steps": args.denoise_steps},
        pytorch_device="cuda:0",
    )
    sample_ids = np.unique(np.linspace(0, len(dataset) - 1, min(args.samples, len(dataset)), dtype=np.int64))
    noise_rng = np.random.default_rng(20260825)
    maes: list[float] = []
    max_errors: list[float] = []
    inference_ms: list[float] = []
    temporal_steps: list[float] = []
    joint_error_sum = np.zeros(14, dtype=np.float64)
    joint_error_count = 0
    hard_limit_violations = 0

    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = av.open(str(args.output_video), mode="w")
    stream = writer.add_stream("mpeg4", rate=4)
    stream.width = 640
    stream.height = 240
    stream.pix_fmt = "yuv420p"
    stream.bit_rate = 1_500_000
    try:
        font = ImageFont.load_default(size=13)
    except TypeError:
        font = ImageFont.load_default()

    try:
        for frame_index in sample_ids:
            sample = dataset[int(frame_index)]
            rgb = _image(sample["observation.images.head"])
            depth = _image(sample["observation.images.depth"])
            state = _numpy(sample["observation.state"]).astype(np.float32)
            teacher = _numpy(sample["action"]).astype(np.float32)
            result = policy.infer(
                {
                    "head_image": rgb,
                    "depth_image": depth,
                    "state": state,
                    "prompt": PROMPT,
                },
                noise=noise_rng.standard_normal((10, 32), dtype=np.float32),
            )
            predicted = np.asarray(result["actions"], dtype=np.float32)
            if predicted.shape != (10, 14) or not np.isfinite(predicted).all():
                raise ValueError(f"checkpoint returned invalid actions {predicted.shape}")
            if teacher.shape != (10, 14) or not np.isfinite(teacher).all():
                raise ValueError(f"held-out sample has invalid actions {teacher.shape}")

            error = np.abs(predicted - teacher)
            maes.append(float(np.mean(error)))
            max_errors.append(float(np.max(error)))
            joint_error_sum += np.sum(error, axis=0)
            joint_error_count += error.shape[0]
            temporal_steps.append(float(np.max(np.abs(np.diff(predicted, axis=0)))))
            hard_limit_violations += int(
                np.count_nonzero((predicted < ARM_MIN_RAD[None, :]) | (predicted > ARM_MAX_RAD[None, :]))
            )
            inference_ms.append(float(result.get("policy_timing", {}).get("infer_ms", np.nan)))

            canvas = Image.new("RGB", (640, 240))
            canvas.paste(Image.fromarray(rgb).resize((320, 240)), (0, 0))
            canvas.paste(Image.fromarray(depth).resize((320, 240)), (320, 0))
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((0, 0, 640, 43), fill=(0, 0, 0))
            draw.text((8, 5), "held-out RGB-D checkpoint evaluation", font=font, fill=(255, 255, 255))
            draw.text(
                (8, 24),
                f"frame {int(frame_index)} | 10-step action MAE {maes[-1]:.4f} rad",
                font=font,
                fill=(80, 220, 255),
            )
            frame = av.VideoFrame.from_ndarray(np.asarray(canvas), format="rgb24")
            for packet in stream.encode(frame):
                writer.mux(packet)
    finally:
        for packet in stream.encode():
            writer.mux(packet)
        writer.close()

    report = {
        "status": "passed" if hard_limit_violations == 0 else "failed",
        "kind": "g1_coke_rgbd_heldout_policy_evaluation_v1",
        "checkpoint": str(args.checkpoint_dir),
        "checkpoint_model_sha256": _sha256(args.checkpoint_dir / "model.safetensors"),
        "repo_id": args.repo_id,
        "dataset_frames": len(dataset),
        "samples": len(maes),
        "denoise_steps": args.denoise_steps,
        "action_mae_rad_mean": float(np.mean(maes)),
        "action_mae_rad_p95": float(np.quantile(maes, 0.95)),
        "action_abs_error_rad_max": float(np.max(max_errors)),
        "per_joint_action_mae_rad": (joint_error_sum / joint_error_count).tolist(),
        "predicted_chunk_step_rad_max": float(np.max(temporal_steps)),
        "hard_limit_violations": hard_limit_violations,
        "inference_ms_mean": float(np.nanmean(inference_ms)),
        "video": str(args.output_video),
        "video_sha256": _sha256(args.output_video),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)
    if report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
