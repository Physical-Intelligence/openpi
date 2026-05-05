# run_train.py: Entrypoint for training pi0.5 on single-arm lipbalm dataset.
# Usage: python experiments/run_train.py --config experiments/configs/lipbalm.yaml [--exp_name NAME] [--resume]

import argparse
import logging
import sys
from pathlib import Path

# Ensure experiments/ is on the path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from config import build_config_from_yaml, build_train_config
from scripts.train_pytorch import init_logging, train_loop


def main():
    parser = argparse.ArgumentParser(description="Train pi0.5 on single-arm lipbalm data")
    parser.add_argument("--config", type=str, default="experiments/configs/lipbalm.yaml",
                        help="Path to YAML experiment config")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Override experiment name")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging")
    args = parser.parse_args()

    init_logging()

    if Path(args.config).exists():
        train_config = build_config_from_yaml(args.config)
    else:
        logging.warning(f"Config file {args.config} not found, using defaults")
        train_config = build_train_config()

    # Apply CLI overrides
    if args.exp_name:
        object.__setattr__(train_config, "exp_name", args.exp_name)
    if args.resume:
        object.__setattr__(train_config, "resume", True)
    if args.no_wandb:
        object.__setattr__(train_config, "wandb_enabled", False)

    logging.info(f"Training config: {train_config.name}")
    logging.info(f"Experiment: {train_config.exp_name}")
    logging.info(f"Steps: {train_config.num_train_steps}, Batch: {train_config.batch_size}")
    logging.info(f"Checkpoint dir: {train_config.checkpoint_dir}")

    train_loop(train_config)


if __name__ == "__main__":
    main()
