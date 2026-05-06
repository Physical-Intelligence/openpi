# run_eval.py: Evaluate a trained pi0.5 single-arm checkpoint against dataset episodes.
# Runs inference on dataset frames and reports action prediction errors.

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np

_EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _EXPERIMENTS_DIR.parent
sys.path.insert(0, str(_EXPERIMENTS_DIR))
sys.path.insert(0, str(_PROJECT_ROOT))

# Patch LeRobot Hub check (same as run_train.py)
import lerobot.common.datasets.utils as _lerobot_utils
_original_get_safe_version = _lerobot_utils.get_safe_version
def _patched_get_safe_version(repo_id, version=None):
    from lerobot.common.constants import HF_LEROBOT_HOME
    local_path = Path(HF_LEROBOT_HOME) / repo_id
    if local_path.exists() and (local_path / "meta" / "info.json").exists():
        return version or "main"
    return _original_get_safe_version(repo_id, version)
_lerobot_utils.get_safe_version = _patched_get_safe_version

from config import build_config_from_yaml, build_train_config
from transforms import AlohaSingleArmOutputs

import openpi.policies.policy_config as policy_config


def evaluate_on_dataset(policy, repo_id, num_episodes=5, frames_per_episode=10):
    """Run inference on dataset episodes and report prediction errors."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id)
    total_episodes = ds.meta.total_episodes
    eval_episodes = min(num_episodes, total_episodes)

    # Sample episodes evenly across the dataset
    episode_indices = np.linspace(0, total_episodes - 1, eval_episodes, dtype=int)

    all_errors = []
    per_episode_errors = []

    for ep_idx in episode_indices:
        ep_start = ds.episode_data_index["from"][ep_idx].item()
        ep_end = ds.episode_data_index["to"][ep_idx].item()
        ep_len = ep_end - ep_start

        # Sample frames evenly within episode
        sample_indices = np.linspace(ep_start, ep_end - 1, min(frames_per_episode, ep_len), dtype=int)
        ep_errors = []

        for frame_idx in sample_indices:
            sample = ds[int(frame_idx)]

            # Build inference input matching our transform pipeline
            import torch
            images = {}
            for key in ["observation.images.cam_high", "observation.images.cam_right_wrist"]:
                img = sample[key].numpy()
                # CHW -> keep as CHW (transform handles conversion)
                images[key.split(".")[-1]] = img

            state = sample["observation.state"].numpy()
            gt_action = sample["action"].numpy()

            inference_input = {
                "state": state,
                "images": {
                    "cam_high": images["cam_high"],
                    "cam_right_wrist": images["cam_right_wrist"],
                },
                "prompt": "pick lipbalm",
            }

            result = policy.infer(inference_input)
            pred_actions = result["actions"]  # [horizon, action_dim]

            # Compare first predicted action to ground truth
            pred_first = pred_actions[0, :7]  # first step, 7-dim
            error = np.abs(pred_first - gt_action)
            all_errors.append(error)
            ep_errors.append(error)

        ep_mean = np.mean(ep_errors, axis=0)
        per_episode_errors.append((ep_idx, ep_mean, np.mean(ep_mean)))
        logging.info(f"  Episode {ep_idx:3d}: mean_abs_error={np.mean(ep_mean):.6f} "
                     f"per_joint={np.round(ep_mean, 5)}")

    all_errors = np.array(all_errors)
    mean_error = np.mean(all_errors, axis=0)
    std_error = np.std(all_errors, axis=0)

    return {
        "mean_abs_error": mean_error,
        "std_abs_error": std_error,
        "overall_mae": float(np.mean(mean_error)),
        "per_episode": per_episode_errors,
        "num_samples": len(all_errors),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate pi0.5 single-arm lipbalm checkpoint")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML experiment config")
    parser.add_argument("--checkpoint_step", type=int, default=None,
                        help="Checkpoint step (default: latest)")
    parser.add_argument("--num_episodes", type=int, default=10,
                        help="Number of episodes to evaluate on")
    parser.add_argument("--frames_per_episode", type=int, default=20,
                        help="Frames to sample per episode")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if Path(args.config).exists():
        train_config = build_config_from_yaml(args.config)
    else:
        train_config = build_train_config()

    # Resolve checkpoint
    checkpoint_dir = train_config.checkpoint_dir
    if args.checkpoint_step:
        checkpoint_dir = checkpoint_dir / str(args.checkpoint_step)
    else:
        steps = [int(d.name) for d in checkpoint_dir.iterdir()
                 if d.is_dir() and d.name.isdigit()]
        if not steps:
            logging.error(f"No checkpoints found in {checkpoint_dir}")
            sys.exit(1)
        latest = max(steps)
        checkpoint_dir = checkpoint_dir / str(latest)

    logging.info(f"Checkpoint: {checkpoint_dir}")

    # Load policy
    logging.info("Loading policy...")
    policy = policy_config.create_trained_policy(train_config, str(checkpoint_dir))
    logging.info("Policy loaded.")

    # Sanity check with dummy input
    from transforms import make_single_arm_example
    example = make_single_arm_example()
    result = policy.infer(example)
    logging.info(f"Sanity check: action shape={result['actions'].shape}")

    # Evaluate on dataset
    repo_id = train_config.data.repo_id
    logging.info(f"\n{'='*60}")
    logging.info(f"Evaluating on dataset: {repo_id}")
    logging.info(f"Episodes: {args.num_episodes}, Frames/episode: {args.frames_per_episode}")
    logging.info(f"{'='*60}")

    results = evaluate_on_dataset(
        policy, repo_id,
        num_episodes=args.num_episodes,
        frames_per_episode=args.frames_per_episode,
    )

    # Summary
    joint_names = ["j0", "j1", "j2", "j3", "j4", "j5", "grip"]
    logging.info(f"\n{'='*60}")
    logging.info(f"EVALUATION RESULTS ({results['num_samples']} samples)")
    logging.info(f"{'='*60}")
    logging.info(f"Overall MAE: {results['overall_mae']:.6f}")
    logging.info(f"\nPer-joint mean absolute error:")
    for name, mean, std in zip(joint_names, results["mean_abs_error"], results["std_abs_error"]):
        logging.info(f"  {name:6s}: {mean:.6f} +/- {std:.6f}")

    # Interpret results
    logging.info(f"\nInterpretation:")
    mae = results["overall_mae"]
    if mae < 0.01:
        logging.info(f"  Excellent — MAE {mae:.4f} rad, model has learned the task well")
    elif mae < 0.05:
        logging.info(f"  Good — MAE {mae:.4f} rad, reasonable for fine-tuning")
    elif mae < 0.1:
        logging.info(f"  Fair — MAE {mae:.4f} rad, may need more training or data")
    else:
        logging.info(f"  Poor — MAE {mae:.4f} rad, check data pipeline or config")


if __name__ == "__main__":
    main()
