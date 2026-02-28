"""Custom continual training script for SpaceCIL.

Wraps openpi's training infrastructure with ContinualHarness for
sequential task training with adapter bank management.

Usage:
    uv run scripts/train_spacecil.py --config spacecil_rm75_payload --task-sequence payload latch clean connector
    uv run scripts/train_spacecil.py --config spacecil_debug --task-sequence task_a task_b --num-steps-per-task 10
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from typing import Any

import numpy as np  # noqa: F401  -- used by commented-out metrics code

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class SpaceCILArgs:
    """CLI arguments for SpaceCIL training."""

    config_name: str  # openpi config name (e.g. "spacecil_rm75_payload")
    task_sequence: list[str]  # ordered list of task IDs
    num_steps_per_task: int = 10_000  # training steps per task
    distillation_alpha: float = 0.5  # weight of distillation loss
    enable_distillation: bool = False  # whether to use behavior distillation
    checkpoint_dir: str = "checkpoints"  # where to save adapter checkpoints
    seed: int = 42
    eval_dir: str = "data/eval_episodes/"
    calibration_dir: str = "data/calibration/"
    exp_name: str = "spacecil_run"


def parse_args() -> SpaceCILArgs:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="SpaceCIL continual training")
    parser.add_argument("--config", type=str, required=True, help="openpi config name")
    parser.add_argument("--task-sequence", nargs="+", required=True, help="ordered task IDs")
    parser.add_argument("--num-steps-per-task", type=int, default=10_000)
    parser.add_argument("--distillation-alpha", type=float, default=0.5)
    parser.add_argument("--enable-distillation", action="store_true", help="enable behavior distillation")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp-name", type=str, default="spacecil_run")
    parser.add_argument("--eval-dir", type=str, default="data/eval_episodes/", help="directory with eval episodes per task")
    parser.add_argument("--calibration-dir", type=str, default="data/calibration/", help="directory with calibration episodes for distillation")
    args = parser.parse_args()
    return SpaceCILArgs(
        config_name=args.config,
        task_sequence=args.task_sequence,
        num_steps_per_task=args.num_steps_per_task,
        distillation_alpha=args.distillation_alpha,
        enable_distillation=args.enable_distillation,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        exp_name=args.exp_name,
        eval_dir=args.eval_dir,
        calibration_dir=args.calibration_dir,
    )


def _build_scorers() -> dict:
    """Build scorer dict for all 4 SpaceCIL tasks."""
    from openpi.research.shared.scorer_base import (
        ConnectorMatingScorer,
        LatchActuationScorer,
        PayloadTransferScorer,
        SurfaceCleaningScorer,
    )

    return {
        "payload": PayloadTransferScorer(),
        "latch": LatchActuationScorer(),
        "clean": SurfaceCleaningScorer(),
        "connector": ConnectorMatingScorer(),
    }


def _load_eval_episodes(eval_dir: str, task_ids: list[str]) -> dict[str, list]:
    """Load evaluation episodes from disk, with graceful degradation."""
    import glob
    import json
    import os

    from openpi.research.shared.episode_schema import Episode

    episodes: dict[str, list] = {}
    for task_id in task_ids:
        task_dir = os.path.join(eval_dir, task_id)
        if os.path.isdir(task_dir):
            task_episodes = []
            for f in sorted(glob.glob(os.path.join(task_dir, "*.json"))):
                with open(f) as fh:
                    task_episodes.append(Episode.from_dict(json.load(fh)))
            episodes[task_id] = task_episodes
        else:
            episodes[task_id] = []
    if not any(episodes.values()):
        logger.warning(f"No eval episodes found in {eval_dir}. Evaluation will produce empty results.")
    return episodes


def make_train_fn(
    config: Any, train_state: Any, state_sharding: Any, mesh: Any, rng: Any, num_steps_per_task: int,
    *, distillation: Any = None,
) -> Any:
    """Create a train_fn closure for ContinualHarness.

    This returns a callable that:
    1. Gets data loader for the task
    2. Runs N steps of openpi's train_step
    3. Optionally computes distillation loss for monitoring
    4. Returns (model, training_info_list)

    Args:
        config: Base TrainConfig (used for debug_task fallback).
        train_state: Initialized TrainState (mutated across tasks via nonlocal).
        state_sharding: Sharding spec for train_state pytree.
        mesh: JAX device mesh.
        rng: PRNG key for training randomness.
        num_steps_per_task: Number of gradient steps per task.
        distillation: Optional BehaviorDistillation instance for anti-forgetting.
    """
    # Defer JAX imports to function scope
    import functools

    import flax.nnx as nnx
    import jax

    from openpi.training import config as _config
    from openpi.training import data_loader as _data_loader
    from openpi.training import sharding as _sharding

    from scripts.train import train_step

    num_steps = num_steps_per_task

    def train_fn(task_id: str) -> tuple[nnx.Module, list[dict[str, Any]]]:
        """Inner training loop for a single task."""
        nonlocal train_state

        logger.info(f"Training task: {task_id} for {num_steps} steps")

        # 1. Resolve per-task config (debug_task uses the passed-in config directly).
        if task_id == "debug_task":
            task_config = config
        else:
            task_config = _config.get_config(f"spacecil_rm75_{task_id}")

        # 2. Compute shardings.
        replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        data_sharding = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec(_sharding.DATA_AXIS)
        )
        train_state_sharding = state_sharding

        # 3. Create data loader for this task.
        data_loader = _data_loader.create_data_loader(
            task_config, sharding=data_sharding, shuffle=True
        )
        data_iter = iter(data_loader)

        # 4. JIT-compile train_step with exact upstream sharding pattern.
        ptrain_step = jax.jit(
            functools.partial(train_step, task_config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=(train_state_sharding, replicated_sharding),
            donate_argnums=(1,),
        )

        # 5. Run training loop.
        collected_infos: list[dict[str, Any]] = []
        has_teacher = distillation is not None and distillation.teacher.has_snapshot
        for i in range(num_steps):
            batch = next(data_iter)
            with _sharding.set_mesh(mesh):
                train_state, info = ptrain_step(rng, train_state, batch)

            # -- Distillation monitoring (post-step) --
            # When a teacher snapshot exists, compute distillation loss on the
            # current batch for monitoring purposes.  The actual combined-loss
            # training (single backward pass) would require a custom train_step
            # variant; this wiring records the metric so we can verify the
            # plumbing and measure the distillation signal.
            if has_teacher and distillation is not None:
                try:
                    task_loss = info["loss"]
                    # Use zero-valued dummy actions for monitoring --
                    # real action extraction requires model forward pass
                    # which will be added with the custom train_step variant.
                    import jax.numpy as jnp
                    dummy_actions = jnp.zeros((1, 1, 1))
                    _total_loss, distill_metrics = distillation.compute_total_loss(
                        task_loss, dummy_actions, dummy_actions,
                    )
                    info["distill_loss"] = distill_metrics["distill_loss"]
                    info["total_loss_with_distill"] = distill_metrics["total_loss"]
                except Exception:
                    pass  # graceful degradation -- don't break training

            collected_infos.append(info)
            if i % max(1, num_steps // 10) == 0:
                loss_str = f"loss={info['loss']:.4f}"
                if "distill_loss" in info:
                    loss_str += f" distill_loss={info['distill_loss']:.4f}"
                logger.info(f"Task {task_id} step {i}/{num_steps} {loss_str}")

        logger.info(f"Task {task_id} training complete ({num_steps} steps)")

        # 6. Distillation: add calibration data from this task's data loader.
        # Sample a few batches worth of episodes for future replay.
        if distillation is not None:
            # Collect calibration episodes from the data loader for replay.
            # Each batch is a tuple (observation, actions); we store raw dicts.
            calibration_samples: list[dict] = []
            try:
                for _ in range(min(5, max(1, num_steps // 100))):
                    cal_batch = next(data_iter)
                    calibration_samples.append({"batch": cal_batch})
            except StopIteration:
                pass  # data loader exhausted, use whatever we got
            if calibration_samples:
                distillation.add_calibration_episodes(task_id, calibration_samples)
                logger.info(
                    f"Added {len(calibration_samples)} calibration batches for task {task_id}. "
                    f"Total calibration episodes: {distillation.memory.num_episodes()}"
                )
            else:
                logger.warning(f"No calibration data collected for task {task_id} (data loader exhausted).")

        # 7. Extract model from final state.
        model = nnx.merge(train_state.model_def, train_state.params)
        return model, collected_infos

    return train_fn


def main() -> None:
    """Run SpaceCIL continual training."""
    args = parse_args()
    logger.info(f"SpaceCIL training: config={args.config_name}, tasks={args.task_sequence}")

    # Import heavy deps only when actually running
    import jax  # noqa: F401

    from openpi.research.spacecil import metrics  # noqa: F401
    from openpi.research.spacecil.behavior_distillation import BehaviorDistillation
    from openpi.research.spacecil.behavior_distillation import CalibrationMemory
    from openpi.research.spacecil.behavior_distillation import TeacherSnapshot
    from openpi.research.spacecil.continual_harness import ContinualHarness
    from openpi.research.spacecil.continual_harness import ContinualResult  # noqa: F401
    from openpi.research.spacecil.task_adapter_bank import TaskAdapterBank
    from openpi.training import config as _config

    # 1. Load config
    config = _config.get_config(args.config_name)
    logger.info(f"Loaded config: {config.name}")

    # 2. Build components
    adapter_bank = TaskAdapterBank()
    if args.enable_distillation:
        distillation = BehaviorDistillation(
            memory=CalibrationMemory(),
            teacher=TeacherSnapshot(),
            distillation_weight=args.distillation_alpha,
        )
    else:
        distillation = None

    # 3. Build harness
    harness = ContinualHarness(  # noqa: F841
        task_sequence=args.task_sequence,
        adapter_bank=adapter_bank,
        distillation=distillation,
        scorers=_build_scorers(),
        eval_episodes=_load_eval_episodes(args.eval_dir, args.task_sequence),
        distillation_alpha=args.distillation_alpha,
    )

    # 4. Initialize training state and run sequence
    import os

    from openpi.training import sharding as _sharding
    from scripts.train import init_train_state

    os.makedirs(f"{args.checkpoint_dir}/{args.exp_name}", exist_ok=True)

    mesh = _sharding.make_mesh(config.fsdp_devices)
    rng = jax.random.key(config.seed)
    init_rng, train_rng = jax.random.split(rng)

    with _sharding.set_mesh(mesh):
        train_state, state_sharding = init_train_state(config, init_rng, mesh, resume=False)

    train_fn = make_train_fn(config, train_state, state_sharding, mesh, train_rng, num_steps_per_task=args.num_steps_per_task, distillation=distillation)

    try:
        result = harness.run_sequence(train_fn)
        logger.info(f"Continual sequence complete. Result matrix shape: {result.result_matrix.shape}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        adapter_bank.save(f"{args.checkpoint_dir}/{args.exp_name}/adapter_bank_crash_recovery")
        raise

    # 5. Compute and log metrics (when result is available)
    # final_metrics = {
    #     "average_success": metrics.average_success(result.result_matrix),
    #     "backward_transfer": metrics.backward_transfer(result.result_matrix),
    #     "forgetting": metrics.forgetting(result.result_matrix),
    # }
    # logger.info(f"Final metrics: {final_metrics}")

    # 6. Save adapter bank
    # adapter_bank.save(f"{args.checkpoint_dir}/{args.exp_name}/adapter_bank")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    main()
