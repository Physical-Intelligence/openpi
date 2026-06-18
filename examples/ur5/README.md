# UR5 Example

This example shows how to fine-tune a pi0 model on a UR5 dataset and serve the resulting policy.
The UR5 has 6 revolute joints and a parallel-jaw gripper (7-dimensional state and action space),
one base (third-person) camera, and one wrist camera.

## Overview

The key components live in two places:

- **`src/openpi/policies/ur5_policy.py`** — `UR5Inputs` and `UR5Outputs` transform classes
  that map between the UR5 environment and the model's expected format.
- **`src/openpi/training/config.py`** — `LeRobotUR5DataConfig` and three ready-to-use training
  configs: `pi0_ur5` (full fine-tune), `pi0_ur5_low_mem_finetune` (LoRA), and `pi0_fast_ur5`.

## Step 1 — Prepare your dataset

Convert your raw UR5 recordings to LeRobot format using the provided script. Adapt the data
loading logic in the script to match your actual data format (rosbag, HDF5, etc.).

```bash
uv run examples/ur5/convert_ur5_data_to_lerobot.py --data_dir /path/to/ur5_data
```

To push the converted dataset to the Hugging Face Hub:

```bash
uv run examples/ur5/convert_ur5_data_to_lerobot.py --data_dir /path/to/ur5_data --push_to_hub
```

The script expects each episode to be a sub-directory containing:

| File | Description |
|---|---|
| `states.npy` | Shape `(T, 7)` — 6 joint angles + 1 gripper position |
| `actions.npy` | Shape `(T, 7)` — 6 joint targets + 1 gripper command |
| `base_rgb_*.png` | Third-person camera frames |
| `wrist_rgb_*.png` | Wrist camera frames |
| `task.txt` | Language instruction for the episode |

## Step 2 — Compute normalization statistics

```bash
uv run scripts/compute_norm_stats.py --config pi0_ur5
```

## Step 3 — Fine-tune

Full fine-tuning with pi0:

```bash
uv run scripts/train.py --config pi0_ur5
```

LoRA fine-tuning (lower GPU memory):

```bash
uv run scripts/train.py --config pi0_ur5_low_mem_finetune
```

Fine-tuning with pi0-FAST:

```bash
uv run scripts/train.py --config pi0_fast_ur5
```

Before running, set `repo_id` in the chosen config inside `src/openpi/training/config.py` to
point to your dataset.

## Step 4 — Serve the policy

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config pi0_ur5 --policy.dir ./checkpoints/pi0_ur5/<run_name>/
```

## Adapting to a different robot

`UR5Inputs`, `UR5Outputs`, and `LeRobotUR5DataConfig` are designed to be easy to copy and
modify for other single-arm robots. See `src/openpi/policies/libero_policy.py` for detailed
inline comments explaining each part of the transform interface.
