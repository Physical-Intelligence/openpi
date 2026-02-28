"""Custom factorized training script for LunarCompose.

Wraps openpi's training infrastructure with MissingCornerHarness for
factorized task × environment training.

Usage:
    uv run scripts/train_lunarcompose.py --config lunarcompose_factorized --rotation 0
    uv run scripts/train_lunarcompose.py --config lunarcompose_debug --rotation 0 --num-steps-per-cell 10
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from typing import Any

import numpy as np  # noqa: F401

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class LunarComposeArgs:
    """CLI arguments for LunarCompose training."""

    config_name: str  # openpi config name (e.g. "lunarcompose_factorized")
    rotation: int = 0  # missing-corner split rotation (0, 1, or 2)
    num_steps_per_cell: int = 10_000  # training steps per cell
    enable_env_adapters: bool = True  # whether to use separate env adapters
    checkpoint_dir: str = "checkpoints"  # where to save adapter checkpoints
    seed: int = 42
    exp_name: str = "lunarcompose_run"


def parse_args() -> LunarComposeArgs:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="LunarCompose factorized training")
    parser.add_argument("--config", type=str, required=True, help="openpi config name")
    parser.add_argument("--rotation", type=int, default=0, choices=[0, 1, 2], help="missing-corner split rotation")
    parser.add_argument("--num-steps-per-cell", type=int, default=10_000)
    parser.add_argument("--no-env-adapters", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp-name", type=str, default="lunarcompose_run")
    args = parser.parse_args()
    return LunarComposeArgs(
        config_name=args.config,
        rotation=args.rotation,
        num_steps_per_cell=args.num_steps_per_cell,
        enable_env_adapters=not args.no_env_adapters,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        exp_name=args.exp_name,
    )


def make_train_fn(config: Any, train_state: Any, state_sharding: Any, mesh: Any, rng: Any) -> Any:
    """Create a train_fn closure for MissingCornerHarness.

    Returns a callable that:
    1. Gets data loader for the task-env cell
    2. Runs N steps of openpi's train_step
    3. Returns (model, training_info_list)
    """
    # Defer JAX imports to function scope
    import flax.nnx as nnx
    import jax  # noqa: F401

    from openpi.training import data_loader as _data_loader  # noqa: F401
    from scripts.train import train_step  # noqa: F401

    def train_fn(task_id: str, env_id: str) -> tuple[nnx.Module, list[dict[str, Any]]]:
        """Inner training loop for a single task-env cell."""
        logger.info(f"Training cell: ({task_id}, {env_id})")

        # TODO: In real usage, this would:
        # 1. Get the data config for this task-env cell
        # 2. Apply task adapter from TaskAdapterBank
        # 3. Apply env adapter from EnvAdapterBank
        # 4. Create data loader
        # 5. Run train_step for N steps
        # 6. Return (model_from_state, info_list)

        raise NotImplementedError(
            f"Real training for cell ({task_id!r}, {env_id!r}) requires GPU and data. "
            "Use MissingCornerHarness directly with a custom train_fn for testing."
        )

    return train_fn


def main() -> None:
    """Run LunarCompose factorized training."""
    args = parse_args()
    logger.info(f"LunarCompose training: config={args.config_name}, rotation={args.rotation}")

    # Import heavy deps only when actually running
    import jax  # noqa: F401

    from openpi.research.lunarcompose.dual_head_router import DualHeadRouter  # noqa: F401
    from openpi.research.lunarcompose.env_adapter_bank import EnvAdapterBank
    from openpi.research.lunarcompose.factorization_diagnostics import seen_unseen_gap  # noqa: F401
    from openpi.research.lunarcompose.missing_corner_harness import MissingCornerHarness
    from openpi.research.spacecil.task_adapter_bank import TaskAdapterBank
    from openpi.training import config as _config

    # 1. Load config
    config = _config.get_config(args.config_name)
    logger.info(f"Loaded config: {config.name}")

    # 2. Build components
    task_ids = ["payload", "latch", "clean", "connector"]
    env_ids = ["nominal", "shadow", "contamination"]

    task_bank = TaskAdapterBank()
    env_bank = EnvAdapterBank()

    # 3. Build harness
    harness = MissingCornerHarness(
        task_ids=task_ids,
        env_ids=env_ids,
        scorers={},  # TODO: wire up real scorers per task
        eval_episodes={},  # TODO: load eval episodes per cell
    )

    # 4. Generate split
    train_cells, test_cells = harness.generate_split(rotation=args.rotation)
    logger.info(f"Split rotation {args.rotation}: {len(train_cells)} train, {len(test_cells)} test cells")

    # 5. Training loop (skeleton)
    # NOTE: Real training requires GPU, mesh, etc.
    # This is the intended wiring pattern:
    #
    # mesh = jax.sharding.Mesh(jax.devices(), ("fsdp",))
    # rng = jax.random.PRNGKey(args.seed)
    # train_state, state_sharding = init_train_state(config, rng, mesh, resume=False)
    # train_fn = make_train_fn(config, train_state, state_sharding, mesh, rng)
    #
    # for task_id, env_id in train_cells:
    #     cell_config = _config.get_config(f"lunarcompose_{task_id}_{env_id}")
    #     model, infos = train_fn(task_id, env_id)
    #     task_bank.register_adapter(task_id, model)
    #     if args.enable_env_adapters:
    #         env_bank.register_env(env_id, model)

    logger.info("LunarCompose training script ready. Provide train_fn for actual training.")
    logger.info(f"Task bank tasks: {task_bank.registered_tasks}")
    logger.info(f"Env bank envs: {env_bank.registered_envs}")

    # 6. Evaluate (when implemented)
    # result = harness.evaluate_all_cells()
    # gap = seen_unseen_gap(result)
    # logger.info(f"Seen-unseen gap: {gap:.4f}")

    # 7. Save banks
    # task_bank.save(f"{args.checkpoint_dir}/{args.exp_name}/task_bank")
    # env_bank.save(f"{args.checkpoint_dir}/{args.exp_name}/env_bank")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    main()
