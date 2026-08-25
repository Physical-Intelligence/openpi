#!/usr/bin/env python3
"""Render an offline RGB/action comparison video for a pi0.5 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from openpi.policies import policy_config
from openpi.training import config as training_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config-name", default="pi05_spark_g1_coke_pickup")
    parser.add_argument("--prompt", default="pick up the Coke can and hold it upright")
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--denoise-steps", type=int, default=5)
    args = parser.parse_args()

    if args.samples < 1 or args.denoise_steps < 1:
        raise ValueError("samples and denoise steps must be positive")
    with np.load(args.episode, allow_pickle=False) as archive:
        rgb = np.asarray(archive["observation_images_head"], dtype=np.uint8)
        state = np.asarray(archive["observation_state"], dtype=np.float32)
        teacher = np.asarray(archive["action"], dtype=np.float32)
    if rgb.ndim != 4 or state.shape != (len(rgb), 24) or teacher.shape != (len(rgb), 21):
        raise ValueError("episode does not match the G1 Coke RGB/state/action contract")

    config = training_config.get_config(args.config_name)
    policy = policy_config.create_trained_policy(
        config,
        args.checkpoint_dir,
        default_prompt=args.prompt,
        sample_kwargs={"num_steps": args.denoise_steps},
        pytorch_device="cuda:0",
    )
    sample_ids = np.unique(np.linspace(0, len(rgb) - 1, min(args.samples, len(rgb)), dtype=np.int64))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = av.open(str(args.output), mode="w")
    stream = writer.add_stream("mpeg4", rate=4)
    stream.width = int(rgb.shape[2])
    stream.height = int(rgb.shape[1])
    stream.pix_fmt = "yuv420p"
    stream.bit_rate = 1_000_000
    try:
        font = ImageFont.load_default(size=13)
    except TypeError:
        font = ImageFont.load_default()

    maes: list[float] = []
    inference_ms: list[float] = []
    try:
        for frame_index in sample_ids:
            result = policy.infer(
                {
                    "head_image": rgb[frame_index],
                    "state": state[frame_index],
                    "prompt": args.prompt,
                }
            )
            actions = np.asarray(result["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != 21 or not np.isfinite(actions).all():
                raise ValueError(f"checkpoint returned invalid actions {actions.shape}")
            mae = float(np.mean(np.abs(actions[0] - teacher[frame_index])))
            maes.append(mae)
            inference_ms.append(float(result.get("policy_timing", {}).get("infer_ms", np.nan)))
            frame = Image.fromarray(rgb[frame_index], mode="RGB")
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, frame.width, 51), fill=(0, 0, 0))
            draw.text((8, 5), "pi0.5 offline checkpoint evaluation", font=font, fill=(255, 255, 255))
            draw.text(
                (8, 29),
                f"sim frame {int(frame_index)} | action MAE {mae:.4f} rad",
                font=font,
                fill=(80, 220, 255),
            )
            video_frame = av.VideoFrame.from_ndarray(np.asarray(frame), format="rgb24")
            for packet in stream.encode(video_frame):
                writer.mux(packet)
    finally:
        for packet in stream.encode():
            writer.mux(packet)
        writer.close()

    report = {
        "status": "passed",
        "kind": "offline_sim_rgb_action_comparison",
        "checkpoint": str(args.checkpoint_dir),
        "episode": str(args.episode),
        "video": str(args.output),
        "samples": len(maes),
        "action_mae_rad_mean": float(np.mean(maes)),
        "action_mae_rad_max": float(np.max(maes)),
        "inference_ms_mean": float(np.nanmean(inference_ms)),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
