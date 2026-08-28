# run_train.py: Entrypoint for training pi0.5 on single-arm datasets.
# Usage: python experiments/train/run_train.py --config experiments/configs/<task>.yaml [--exp_name NAME] [--resume]

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure experiments/ and project root are on the path for local imports
_EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _EXPERIMENTS_DIR.parent
sys.path.insert(0, str(_EXPERIMENTS_DIR))
sys.path.insert(0, str(_PROJECT_ROOT))

# Patch LeRobot to skip Hub version check for local datasets.
# LeRobot always calls get_safe_version() which hits HuggingFace Hub,
# even when the dataset exists locally under HF_LEROBOT_HOME.
import lerobot.common.datasets.utils as _lerobot_utils

_original_get_safe_version = _lerobot_utils.get_safe_version

def _patched_get_safe_version(repo_id, version=None):
    """Skip Hub check if dataset exists locally under HF_LEROBOT_HOME."""
    from lerobot.common.constants import HF_LEROBOT_HOME
    local_path = Path(HF_LEROBOT_HOME) / repo_id
    if local_path.exists() and (local_path / "meta" / "info.json").exists():
        return version or "main"
    return _original_get_safe_version(repo_id, version)

_lerobot_utils.get_safe_version = _patched_get_safe_version

from config import build_config_from_yaml, build_train_config

# Import both JAX and PyTorch trainers
import importlib.util

def _load_trainer(name, filename):
    spec = importlib.util.spec_from_file_location(name, _PROJECT_ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Lazy-load trainers to avoid importing both frameworks upfront
_jax_trainer = None
_pytorch_trainer = None

def get_jax_trainer():
    global _jax_trainer
    if _jax_trainer is None:
        _jax_trainer = _load_trainer("train_jax", "train.py")
    return _jax_trainer

def get_pytorch_trainer():
    global _pytorch_trainer
    if _pytorch_trainer is None:
        _pytorch_trainer = _load_trainer("train_pytorch", "train_pytorch.py")
    return _pytorch_trainer


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
    parser.add_argument("--backend", type=str, default="jax", choices=["jax", "pytorch"],
                        help="Training backend: jax (supports LoRA) or pytorch (full finetune only)")
    args = parser.parse_args()

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

    logging.basicConfig(level=logging.INFO)
    logging.info(f"Training config: {train_config.name}")
    logging.info(f"Experiment: {train_config.exp_name}")
    logging.info(f"Backend: {args.backend}")
    logging.info(f"LoRA: {train_config.freeze_filter is not None}")
    logging.info(f"Steps: {train_config.num_train_steps}, Batch: {train_config.batch_size}")
    logging.info(f"Checkpoint dir: {train_config.checkpoint_dir}")

    if args.backend == "jax":
        trainer = get_jax_trainer()
        trainer.main(train_config)
    else:
        trainer = get_pytorch_trainer()
        trainer.init_logging()
        trainer.train_loop(train_config)


if __name__ == "__main__":
    main()
