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
    enable_distillation: bool = True  # whether to use behavior distillation
    checkpoint_dir: str = "checkpoints"  # where to save adapter checkpoints
    seed: int = 42
    exp_name: str = "spacecil_run"


def parse_args() -> SpaceCILArgs:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="SpaceCIL continual training")
    parser.add_argument("--config", type=str, required=True, help="openpi config name")
    parser.add_argument("--task-sequence", nargs="+", required=True, help="ordered task IDs")
    parser.add_argument("--num-steps-per-task", type=int, default=10_000)
    parser.add_argument("--distillation-alpha", type=float, default=0.5)
    parser.add_argument("--no-distillation", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp-name", type=str, default="spacecil_run")
    args = parser.parse_args()
    return SpaceCILArgs(
        config_name=args.config,
        task_sequence=args.task_sequence,
        num_steps_per_task=args.num_steps_per_task,
        distillation_alpha=args.distillation_alpha,
        enable_distillation=not args.no_distillation,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        exp_name=args.exp_name,
    )


def make_train_fn(config: Any, train_state: Any, state_sharding: Any, mesh: Any, rng: Any) -> Any:
    """Create a train_fn closure for ContinualHarness.

    This returns a callable that:
    1. Gets data loader for the task
    2. Runs N steps of openpi's train_step
    3. Returns (model, training_info_list)
    """
    # Defer JAX imports to function scope
    import flax.nnx as nnx
    import jax  # noqa: F401

    from openpi.training import data_loader as _data_loader  # noqa: F401

    from scripts.train import init_train_state, train_step  # noqa: F401

    def train_fn(task_id: str) -> tuple[nnx.Module, list[dict[str, Any]]]:
        """Inner training loop for a single task."""
        logger.info(f"Training task: {task_id}")

        # TODO: In real usage, this would:
        # 1. Get the data config for this task
        # 2. Create data loader
        # 3. Run train_step for N steps
        # 4. Return (model_from_state, info_list)

        # For now, this is a skeleton that shows the intended structure
        raise NotImplementedError(
            f"Real training for task '{task_id}' requires GPU and data. "
            "Use ContinualHarness directly with a custom train_fn for testing."
        )

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
        scorers={},  # TODO: wire up real scorers per task
        eval_episodes={},  # TODO: load eval episodes per task
        distillation_alpha=args.distillation_alpha,
    )

    # 4. Run sequence
    # NOTE: Real training requires make_train_fn with GPU, mesh, etc.
    # This is the intended wiring pattern:
    #
    # mesh = jax.sharding.Mesh(jax.devices(), ("fsdp",))
    # rng = jax.random.PRNGKey(args.seed)
    # train_state, state_sharding = init_train_state(config, rng, mesh, resume=False)
    # train_fn = make_train_fn(config, train_state, state_sharding, mesh, rng)
    # result = harness.run_sequence(train_fn)

    logger.info("SpaceCIL training script ready. Provide train_fn for actual training.")
    logger.info(f"Task sequence: {args.task_sequence}")
    logger.info(f"Adapter bank tasks: {adapter_bank.registered_tasks}")

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
