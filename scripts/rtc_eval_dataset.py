#!/usr/bin/env python

import logging
import pathlib
import random
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tyro

from openpi.policies import policy_config
from openpi.training import config as train_config


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    logging.info(f"Random seed set to: {seed}")


class RTCDatasetEvaluator:
    """Evaluator for RTC on dataset samples."""

    def __init__(self, cfg: "Args"):
        self.cfg = cfg

        # Load the training config to get model configuration
        logging.info(f"Loading policy from {cfg.checkpoint_path}")
        train_cfg = train_config.get_config(cfg.train_config_name)

        # Load policy using the openpi policy_config
        self.policy = policy_config.create_trained_policy(
            train_config=train_cfg,
            checkpoint_dir=pathlib.Path(cfg.checkpoint_path),
            sample_kwargs=cfg.sample_kwargs or {},
        )

        logging.info(f"Policy loaded successfully")
        logging.info(f"Model config: {train_cfg.model}")

        # Load dataset - using lerobot if available
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
            logging.info(f"Loading dataset: {cfg.dataset_repo_id}")
            self.dataset = LeRobotDataset(
                cfg.dataset_repo_id,
                delta_timestamps={"action": np.arange(cfg.action_horizon) / 30}
            )
            logging.info(f"Dataset loaded: {len(self.dataset)} samples, {self.dataset.num_episodes} episodes")
        except ImportError:
            raise ImportError(
                "lerobot package not found. Please install it with: pip install lerobot"
            )

    def run_evaluation(self) -> dict:
        """Run full evaluation on dataset.

        Returns:
            Dictionary with aggregated metrics and detailed results
        """
        logging.info(f"Starting evaluation on {self.cfg.num_samples} samples")

        prev_chunk_left_over = None

        for i in range(self.cfg.num_samples):
            # Get a random sample from the dataset
            idx = np.random.randint(0, len(self.dataset))
            sample = self.dataset[idx]

            # Convert sample to proper format
            # Note: This assumes the dataset returns observations in a specific format
            # You may need to adjust this based on your dataset structure
            obs = {
                "observation/image": np.array(sample.get("observation.images.top", sample.get("observation/image"))),
                "observation/state": np.array(sample.get("observation.state", [])),
            }

            # Add prompt if available
            if "prompt" in sample or "task" in sample:
                obs["prompt"] = sample.get("prompt", sample.get("task", ""))

            if i % 2 == 0:
                # Store actions from this sample for comparison
                if "action" in sample:
                    actions = np.array(sample["action"])
                    prev_chunk_left_over = actions[:self.cfg.action_horizon // 2]
                continue

            if prev_chunk_left_over is None:
                continue

            # Generate noise for both runs (same noise for fair comparison)
            noise = np.random.randn(self.cfg.action_horizon, self.cfg.action_dim).astype(np.float32)

            # Inference using the policy
            # Note: The pi0 model's inference is handled through the Policy.infer method
            result = self.policy.infer(obs, noise=noise)
            actions = result["actions"]

            # Create visualization
            fig, axs = plt.subplots(min(6, self.cfg.action_dim), 1, figsize=(12, 12))
            if self.cfg.action_dim == 1:
                axs = [axs]
            fig.suptitle(f"Sample {i} - Action Prediction", fontsize=16)

            # Plot actions
            self.axs = axs
            self.plot_waypoints(prev_chunk_left_over, label="Previous Actions", color="green")
            self.plot_waypoints(actions, label="Predicted Actions", color="blue")

            plt.tight_layout()
            plt.savefig(f"actions_sample_{i}.png", dpi=150)
            logging.info(f"Saved actions for sample {i} to actions_sample_{i}.png")
            plt.close(fig)

        logging.info("Evaluation completed")
        return {}

    def plot_waypoints(self, chunk, start_from: int = 0, color: str | None = None, label: str | None = None):
        for j in range(chunk.shape[-1]):
            self.axs[j].plot(
                np.arange(start_from, start_from + chunk.shape[0]),
                chunk[:, j],
                color=color,
                label=label,
            )
            self.axs[j].set_ylabel("Joint angle", fontsize=14)
            self.axs[j].grid()
            plt.tick_params(labelsize=14)
            self.axs[j].legend(loc="upper right", fontsize=14)
            if j == 2:
                self.axs[j].set_xlabel("Step #", fontsize=16)

@dataclass
class Args:
    """Arguments for RTC dataset evaluation."""

    # Training config name (e.g., "pi0_libero", "pi05_droid", etc.)
    train_config_name: str = field(
        metadata={"help": "Name of the training config to use (from openpi.training.config)"}
    )

    # Path to the checkpoint directory
    checkpoint_path: str = field(
        metadata={"help": "Path to the checkpoint directory"}
    )

    # Dataset configuration
    dataset_repo_id: str = field(
        metadata={"help": "HuggingFace repo ID for the LeRobot dataset"}
    )

    # Number of samples to evaluate
    num_samples: int = field(
        default=10,
        metadata={"help": "Number of samples to evaluate"},
    )

    # Action dimensions
    action_dim: int = field(
        default=7,
        metadata={"help": "Action dimension"},
    )

    action_horizon: int = field(
        default=50,
        metadata={"help": "Action horizon (chunk size)"},
    )

    # If provided, will be used as default prompt
    default_prompt: str | None = field(
        default=None,
        metadata={"help": "Default prompt to use if not present in the data"},
    )

    # Additional sample kwargs to pass to the model
    sample_kwargs: dict[str, Any] | None = field(
        default=None,
        metadata={"help": "Additional kwargs to pass to sample_actions"},
    )

    # Seed configuration
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for reproducibility"},
    )
    
def main(args: Args):
    """Main entry point for RTC dataset evaluation."""
    # Set random seed for reproducibility
    set_seed(args.seed)

    logging.info("=" * 80)
    logging.info("Pi0 Dataset Evaluation with JAX")
    logging.info("=" * 80)
    logging.info(f"Training config: {args.train_config_name}")
    logging.info(f"Checkpoint: {args.checkpoint_path}")
    logging.info(f"Dataset: {args.dataset_repo_id}")
    logging.info(f"Num samples: {args.num_samples}")
    logging.info(f"Action dim: {args.action_dim}")
    logging.info(f"Action horizon: {args.action_horizon}")
    logging.info(f"Seed: {args.seed}")
    logging.info("=" * 80)

    evaluator = RTCDatasetEvaluator(args)
    evaluator.run_evaluation()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
