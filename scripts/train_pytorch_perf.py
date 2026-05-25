"""
PyTorch single-GPU training with optional performance profiling (standalone).

Launch commands match `train_pytorch.py` (single GPU only). Do not use torchrun.

  python scripts/train_pytorch_perf.py <config_name> --exp_name <run_name>
  python scripts/train_pytorch_perf.py debug --exp_name my_run --perf

Perf flags:
  --perf                    Enable profiling: warmup -> capture -> continue training
  --perf-warmup-steps N     Warmup steps before profiler (default: 5)
  --perf-profile-steps N    Steps to record trace (default: 5)
  --perf-start-step N       Global step to start perf window (default: 0)
  --perf-trace-dir PATH     Trace output directory (default: ./perf_traces)

With `--perf`: warmup_steps normal training, then profile_steps with torch.profiler
(Chrome trace for Perfetto), then continues until num_train_steps.
"""

from __future__ import annotations

import dataclasses
import gc
import json
import logging
import os
import pathlib
import platform
import shutil
import sys
import time
from typing import Any

_scripts_dir = pathlib.Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import jax
import numpy as np
import safetensors.torch
import torch
import tqdm
import wandb

import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.training.config as _config
import openpi.training.data_loader as _data
import train_pytorch as _base


@dataclasses.dataclass(frozen=True)
class PerfConfig:
    enabled: bool = False
    warmup_steps: int = 5
    profile_steps: int = 5
    start_step: int = 0
    trace_output_dir: str = "./perf_traces"


def _assert_single_gpu() -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        raise RuntimeError(
            "train_pytorch_perf.py only supports single-GPU training. "
            "Do not use torchrun; run: python scripts/train_pytorch_perf.py ..."
        )


def _setup_device() -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        return torch.device("cuda:0")
    return torch.device("cpu")


def parse_perf_from_argv(argv: list[str] | None = None) -> tuple[PerfConfig, list[str]]:
    if argv is None:
        argv = sys.argv
    perf = PerfConfig()
    cleaned: list[str] = [argv[0]]
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "--perf":
            perf = dataclasses.replace(perf, enabled=True)
            i += 1
        elif arg == "--no-perf":
            perf = dataclasses.replace(perf, enabled=False)
            i += 1
        elif arg == "--perf-warmup-steps" and i + 1 < len(argv):
            perf = dataclasses.replace(perf, warmup_steps=int(argv[i + 1]))
            i += 2
        elif arg == "--perf-profile-steps" and i + 1 < len(argv):
            perf = dataclasses.replace(perf, profile_steps=int(argv[i + 1]))
            i += 2
        elif arg == "--perf-start-step" and i + 1 < len(argv):
            perf = dataclasses.replace(perf, start_step=int(argv[i + 1]))
            i += 2
        elif arg == "--perf-trace-dir" and i + 1 < len(argv):
            perf = dataclasses.replace(perf, trace_output_dir=argv[i + 1])
            i += 2
        else:
            cleaned.append(arg)
            i += 1
    return perf, cleaned


def _perf_phase(global_step: int, perf: PerfConfig) -> str | None:
    offset = global_step - perf.start_step
    if offset < 0:
        return None
    if offset < perf.warmup_steps:
        return "warmup"
    if offset < perf.warmup_steps + perf.profile_steps:
        return "profile"
    return None


def _perf_trace_dir(perf: PerfConfig, config: _config.TrainConfig) -> pathlib.Path:
    base = pathlib.Path(perf.trace_output_dir) / config.exp_name
    base.mkdir(parents=True, exist_ok=True)
    return base


def _profiler_activities() -> list[torch.profiler.ProfilerActivity]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return activities


def _event_cpu_time_us(event: Any) -> int:
    return int(getattr(event, "cpu_time_total", 0) or 0)


def _event_device_time_us(event: Any) -> int:
    # CUDA builds may expose cuda_time_total; ROCm/HIP (DCU) uses device_time_total.
    for attr in ("device_time_total", "cuda_time_total"):
        val = getattr(event, attr, None)
        if val:
            return int(val)
    return 0


def _finalize_profiler(
    profiler: torch.profiler.profile,
    *,
    perf: PerfConfig,
    config: _config.TrainConfig,
    global_step: int,
) -> None:
    trace_dir = _perf_trace_dir(perf, config)
    trace_path = trace_dir / f"trace_step{global_step:06d}.json"
    profiler.export_chrome_trace(str(trace_path))
    logging.info(f"Exported Perfetto Chrome trace -> {trace_path}")

    try:
        key_averages = profiler.key_averages()
        top_ops: list[dict[str, Any]] = []
        for event in sorted(key_averages, key=_event_device_time_us, reverse=True)[:20]:
            device_us = _event_device_time_us(event)
            cpu_us = _event_cpu_time_us(event)
            top_ops.append(
                {
                    "name": event.key,
                    "device_time_ms": device_us / 1000.0,
                    "cpu_time_ms": cpu_us / 1000.0,
                    "count": event.count,
                }
            )
            logging.info(
                f"  op {event.key}: device={device_us / 1000:.2f}ms "
                f"cpu={cpu_us / 1000:.2f}ms count={event.count}"
            )

        summary_path = trace_dir / "perf_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "exp_name": config.exp_name,
                    "global_step": global_step,
                    "perf": dataclasses.asdict(perf),
                    "trace_file": str(trace_path),
                    "top_operators": top_ops,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        logging.info(f"Wrote perf summary -> {summary_path}")
    except Exception as e:
        logging.warning(f"Failed to write perf_summary.json (trace already saved): {e!s}")

    logging.info("Open https://ui.perfetto.dev/ and load the trace JSON to analyze DCU/GPU utilization.")


def train_loop(config: _config.TrainConfig, perf: PerfConfig) -> None:
    """Single-GPU training loop aligned with train_pytorch.py."""
    device = _setup_device()
    _base.set_seed(config.seed, local_rank=0)

    if perf.enabled:
        logging.info(
            f"Perf enabled: {perf.warmup_steps} warmup step(s) from global_step={perf.start_step}, "
            f"then {perf.profile_steps} profile step(s), then continue to num_train_steps={config.num_train_steps}"
        )

    resuming = False
    if config.resume:
        exp_checkpoint_dir = config.checkpoint_dir
        if exp_checkpoint_dir.exists():
            latest_step = _base.get_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is not None:
                resuming = True
                logging.info(
                    f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}"
                )
            else:
                raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir} for resume")
        else:
            raise FileNotFoundError(f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume")
    elif config.overwrite and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
        logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")

    if not resuming:
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created experiment checkpoint directory: {config.checkpoint_dir}")
    else:
        logging.info(f"Using existing experiment checkpoint directory: {config.checkpoint_dir}")

    _base.init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    logging.info(f"Single-GPU training, batch_size={config.batch_size}, device={device}")
    loader, data_config = _base.build_datasets(config)

    if config.wandb_enabled and not resuming:
        sample_data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
        sample_batch = next(iter(sample_data_loader))
        observation, actions = sample_batch
        sample_batch = observation.to_dict()
        sample_batch["actions"] = actions
        images_to_log = []
        batch_size = next(iter(sample_batch["image"].values())).shape[0]
        for i in range(min(5, batch_size)):
            img_concatenated = torch.cat([img[i].permute(1, 2, 0) for img in sample_batch["image"].values()], axis=1)
            images_to_log.append(wandb.Image(img_concatenated.cpu().numpy()))
        wandb.log({"camera_views": images_to_log}, step=0)
        del sample_batch, observation, actions, images_to_log, sample_data_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Cleared sample batch and data loader from memory")

    if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = config.model
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)

    if hasattr(model, "gradient_checkpointing_enable"):
        enable_gradient_checkpointing = True
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing for memory optimization")
    else:
        enable_gradient_checkpointing = False
        logging.info("Gradient checkpointing is not supported for this model")

    if torch.cuda.is_available():
        _base.log_memory_usage(device, 0, "after_model_creation")

    if config.pytorch_weight_path is not None:
        logging.info(f"Loading weights from: {config.pytorch_weight_path}")
        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        safetensors.torch.load_model(model, model_path)
        logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")

    lr_warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    global_step = 0
    if resuming:
        global_step = _base.load_checkpoint(model, optim, config.checkpoint_dir, device)
        logging.info(f"Resumed training from step {global_step}")

    def lr_schedule(step: int):
        if step < lr_warmup_steps:
            init_lr = peak_lr / (lr_warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / lr_warmup_steps
        progress = min(1.0, (step - lr_warmup_steps) / max(1, decay_steps - lr_warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    model.train()
    start_time = time.time()
    infos: list[dict[str, Any]] = []

    logging.info(f"Running on: {platform.node()} | single GPU")
    logging.info(
        f"Training config: batch_size={config.batch_size}, num_train_steps={config.num_train_steps}"
    )
    logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
    logging.info(
        f"LR schedule: warmup={lr_warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}"
    )
    logging.info(
        f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, "
        f"clip_norm={config.optimizer.clip_gradient_norm}"
    )
    logging.info("EMA is not supported for PyTorch training")
    logging.info(f"Training precision: {model_cfg.dtype}")

    pbar = tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training")

    torch_profiler: torch.profiler.profile | None = None
    last_perf_phase: str | None = None

    def _maybe_start_profiler(step: int) -> None:
        nonlocal torch_profiler
        if not perf.enabled or torch_profiler is not None:
            return
        if step < perf.start_step + perf.warmup_steps:
            return
        if step >= perf.start_step + perf.warmup_steps + perf.profile_steps:
            return
        torch_profiler = torch.profiler.profile(
            activities=_profiler_activities(),
            record_shapes=True,
            profile_memory=True,
        )
        torch_profiler.__enter__()
        logging.info(f"Started torch.profiler capture at step {step}")

    def _maybe_stop_profiler(step: int) -> None:
        nonlocal torch_profiler
        if torch_profiler is None:
            return
        if step < perf.start_step + perf.warmup_steps + perf.profile_steps - 1:
            return
        torch_profiler.__exit__(None, None, None)
        _finalize_profiler(torch_profiler, perf=perf, config=config, global_step=step)
        torch_profiler = None
        logging.info(f"Finished torch.profiler capture at step {step}, resuming normal training")

    while global_step < config.num_train_steps:
        for observation, actions in loader:
            if global_step >= config.num_train_steps:
                break

            if perf.enabled:
                current_perf_phase = _perf_phase(global_step, perf)
                if current_perf_phase != last_perf_phase and current_perf_phase is not None:
                    logging.info(f"Perf phase -> {current_perf_phase} (global_step={global_step})")
                    last_perf_phase = current_perf_phase
                _maybe_start_profiler(global_step)

            observation = jax.tree.map(lambda x: x.to(device), observation)  # noqa: PLW2901
            actions = actions.to(torch.float32)  # noqa: PLW2901
            actions = actions.to(device)  # noqa: PLW2901

            for pg in optim.param_groups:
                pg["lr"] = lr_schedule(global_step)

            losses = model(observation, actions)
            if isinstance(losses, list | tuple):
                losses = torch.stack(losses)
            elif not isinstance(losses, torch.Tensor):
                losses = torch.tensor(losses, device=device, dtype=torch.float32)

            loss = losses.mean()
            loss.backward()

            if global_step < 5 and torch.cuda.is_available():
                _base.log_memory_usage(device, global_step, "after_backward")

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.optimizer.clip_gradient_norm
            )
            optim.step()
            optim.zero_grad(set_to_none=True)

            for param in model.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad = None

            if torch_profiler is not None:
                torch_profiler.step()

            infos.append(
                {
                    "loss": loss.item(),
                    "learning_rate": optim.param_groups[0]["lr"],
                    "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                }
            )

            if global_step % config.log_interval == 0:
                elapsed = time.time() - start_time
                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)
                avg_grad_norm = None
                if any("grad_norm" in info for info in infos):
                    vals = [
                        info["grad_norm"] for info in infos if "grad_norm" in info and info["grad_norm"] is not None
                    ]
                    if vals:
                        avg_grad_norm = sum(vals) / len(vals)
                logging.info(
                    f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} grad_norm={avg_grad_norm:.2f} time={elapsed:.1f}s"
                    if avg_grad_norm is not None
                    else f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} time={elapsed:.1f}s"
                )
                if config.wandb_enabled and infos:
                    log_payload = {
                        "loss": avg_loss,
                        "learning_rate": avg_lr,
                        "step": global_step,
                        "time_per_step": elapsed / max(config.log_interval, 1),
                    }
                    if avg_grad_norm is not None:
                        log_payload["grad_norm"] = avg_grad_norm
                    wandb.log(log_payload, step=global_step)
                start_time = time.time()
                infos = []

            if perf.enabled:
                _maybe_stop_profiler(global_step)

            global_step += 1
            _base.save_checkpoint(model, optim, global_step, config, is_main=True, data_config=data_config)

            pbar.update(1)
            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "lr": f"{optim.param_groups[0]['lr']:.2e}", "step": global_step}
            )

    pbar.close()

    if torch_profiler is not None:
        _maybe_stop_profiler(global_step - 1)

    if config.wandb_enabled:
        wandb.finish()


def main() -> None:
    _assert_single_gpu()
    _base.init_logging()
    perf, cleaned_argv = parse_perf_from_argv()
    sys.argv = cleaned_argv
    config = _config.cli()
    train_loop(config, perf)


if __name__ == "__main__":
    main()
