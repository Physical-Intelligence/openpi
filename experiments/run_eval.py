# run_eval.py: Entrypoint for serving/evaluating a trained pi0.5 single-arm lipbalm checkpoint.
# Usage: python experiments/run_eval.py --config experiments/configs/lipbalm.yaml [--checkpoint_step STEP]

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import build_config_from_yaml, build_train_config

import openpi.policies.policy_config as policy_config


def main():
    parser = argparse.ArgumentParser(description="Evaluate/serve pi0.5 single-arm lipbalm checkpoint")
    parser.add_argument("--config", type=str, default="experiments/configs/lipbalm.yaml",
                        help="Path to YAML experiment config")
    parser.add_argument("--checkpoint_step", type=int, default=None,
                        help="Checkpoint step to load (default: latest)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Server port")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if Path(args.config).exists():
        train_config = build_config_from_yaml(args.config)
    else:
        logging.warning(f"Config file {args.config} not found, using defaults")
        train_config = build_train_config()

    # Resolve checkpoint directory
    checkpoint_dir = train_config.checkpoint_dir
    if args.checkpoint_step:
        checkpoint_dir = checkpoint_dir / str(args.checkpoint_step)
    else:
        # Find latest checkpoint
        steps = [int(d.name) for d in checkpoint_dir.iterdir()
                 if d.is_dir() and d.name.isdigit()]
        if not steps:
            logging.error(f"No checkpoints found in {checkpoint_dir}")
            sys.exit(1)
        latest = max(steps)
        checkpoint_dir = checkpoint_dir / str(latest)
        logging.info(f"Using latest checkpoint: step {latest}")

    logging.info(f"Loading checkpoint from: {checkpoint_dir}")

    # Create policy from checkpoint
    policy = policy_config.create_trained_policy(train_config, str(checkpoint_dir))

    logging.info(f"Policy loaded. Serving on port {args.port}")
    logging.info("Use openpi-client or scripts/serve_policy.py for full server setup")

    # Run a quick sanity check with dummy input
    from transforms import make_single_arm_example
    example = make_single_arm_example()
    result = policy.infer(example)
    logging.info(f"Sanity check passed. Action shape: {result['actions'].shape}")


if __name__ == "__main__":
    main()
