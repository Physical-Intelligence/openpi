#!/usr/bin/env python3
"""Run a real pi0.5 checkpoint through GB10 GPU inference and emit evidence."""

import argparse
import dataclasses
import hashlib
import json
import math
import pathlib
import platform
import time

import safetensors.torch
import torch

from openpi.models import model as model_api
from openpi.models_pytorch import pi0_pytorch
from openpi.training import config as training_config


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        while chunk := checkpoint_file.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _observation(config, device: torch.device) -> model_api.Observation:
    images = {key: torch.zeros((1, 3, 224, 224), dtype=torch.float32, device=device) for key in model_api.IMAGE_KEYS}
    image_masks = {key: torch.ones((1,), dtype=torch.bool, device=device) for key in model_api.IMAGE_KEYS}
    tokenized_prompt = torch.ones((1, config.max_token_len), dtype=torch.int64, device=device)
    tokenized_prompt_mask = torch.zeros((1, config.max_token_len), dtype=torch.bool, device=device)
    tokenized_prompt_mask[:, :8] = True
    return model_api.Observation(
        images=images,
        image_masks=image_masks,
        state=torch.zeros((1, config.action_dim), dtype=torch.float32, device=device),
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=pathlib.Path, required=True)
    parser.add_argument("--config-name", default="pi05_spark_smoke")
    parser.add_argument("--num-steps", type=int, default=2)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to report a CPU-only pi0.5 pass")

    device = torch.device("cuda:0")
    capability = torch.cuda.get_device_capability(device)
    if capability < (12, 1):
        raise RuntimeError(f"Expected a GB10-class sm_121 GPU, found sm_{capability[0]}{capability[1]}")

    train_config = training_config.get_config(args.config_name)
    model_config = dataclasses.replace(train_config.model, pytorch_compile_mode=None)
    if not model_config.pi05:
        raise ValueError(f"{args.config_name} is not a pi0.5 config")

    weight_path = args.checkpoint_dir / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError(weight_path)

    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats(device)
    model = pi0_pytorch.PI0Pytorch(model_config)
    safetensors.torch.load_model(model, weight_path)
    model.to(device).eval()

    observation = _observation(model_config, device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        actions = model.sample_actions(device, observation, num_steps=args.num_steps)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    actions_cpu = actions.float().cpu()
    finite = bool(torch.isfinite(actions_cpu).all().item())
    if not finite:
        raise RuntimeError("pi0.5 produced non-finite actions")

    report = {
        "status": "passed",
        "model": "pi0.5",
        "config": args.config_name,
        "checkpoint": {
            "path": str(weight_path),
            "bytes": weight_path.stat().st_size,
            "sha256": _sha256(weight_path),
        },
        "runtime": {
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": list(capability),
        },
        "inference": {
            "denoise_steps": args.num_steps,
            "elapsed_seconds": elapsed,
            "action_shape": list(actions_cpu.shape),
            "all_finite": finite,
            "action_mean": actions_cpu.mean().item(),
            "action_std": actions_cpu.std().item(),
            "action_l2": math.sqrt(torch.square(actions_cpu).sum().item()),
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(device),
        },
    }

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n")


if __name__ == "__main__":
    main()
