---
license: apache-2.0
tags:
  - robotics
  - pi0
  - so101
  - openpi
  - lerobot
---

# pi0.5 SO101 Stacking Rings

Fine-tuned [pi0.5](https://github.com/Physical-Intelligence/openpi) checkpoint for the SO101 stacking-rings task (6D joint-space actions with delta joints + absolute gripper).

**Note:** Training was in progress when step 5000 was published. Later steps may be added to this repo under `checkpoints/<step>/params/`.

## Experiment

- **Objective:** Replication run — verify pi0.5 fine-tuning on `lorenzouttini/so101_stacking_rings`.
- **Weight init:** `weights/pi05_base/params` (pi0.5 base weights).
- **Published step:** 5,000 (of 50,000 planned).
- **Loss at step 5,000:** 0.0252

## Config

- **Config name:** `pi05_so101_stacking_rings`
- **Model:** pi0.5 (`pi05=True`, `action_horizon=30`)
- **Batch size:** 32
- **Learning rate:** 2.5e-5 cosine decay (1k warmup, decay to 2.5e-6)
- **EMA decay:** 0.999
- **Delta actions:** mask `[T,T,T,T,T,F]` (5 joints delta, gripper absolute)
- **Norm:** quantile normalization (pi0.5 default) + per-timestep action norm
- **Default prompt:** `stack the rings`

## Dataset

- [lorenzouttini/so101_stacking_rings](https://huggingface.co/datasets/lorenzouttini/so101_stacking_rings) — 101 episodes, ~34k frames

## Checkpoint Hashes

Verify integrity with:

```bash
cd checkpoints/<step> && find params -type f | sort | xargs sha256sum | sha256sum
```

| Step | Loss | SHA-256 |
|------|------|---------|
| 5,000 | 0.0252 | `3e772c819c5e0233b939e5f739f4de74b1ca8224e4fcf9499d59a9bf603cdb7c` |

## W&B

- [Training dashboard](https://wandb.ai/pravsels/so101_stacking_rings/runs/tmrsnyct)

## Repo Structure

```
assets/                         # Norm stats, valid_indices.txt
checkpoints/5000/params/        # Model weights (params only)
README.md                       # This file
TRAINING_LOG.md                 # Training log
```

## Usage

```python
from openpi.training.config import get_config
from openpi.serving.policy_server import PolicyServer

config = get_config("pi05_so101_stacking_rings")
server = PolicyServer(config, checkpoint_path="checkpoints/5000/params")
```
