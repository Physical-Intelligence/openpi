---
license: apache-2.0
tags:
  - robotics
  - LeRobot
  - pi0
  - so101
  - openpi
---

# pi0.5 SO101 Stacking Big Rings (Isambard)

Fine-tuned pi0.5 checkpoint for SO101 big-ring stacking. Trained on Isambard GH200 (4× GPU node). **Training is in progress** — this repo publishes interim checkpoints under `step_<N>/params/`; final 50k may be added as `step_49999/params/`.

## Published checkpoints

| Directory           | Step   | Loss (approx) | Notes                                             |
| ------------------- | ------ | ------------- | ------------------------------------------------- |
| _(none yet)_        | —      | —             | First checkpoint published at step 5,000          |

## Experiment

- **Objective:** Fine-tune pi0.5 on `lorenzouttini/so101_stacking_big_rings`.
- **Weight init:** pi0.5 base (`pi05_base`).
- **Training target:** 50,000 steps (`save_interval=5000`).

## Config

- **Config name:** `pi05_so101_stacking_big_rings`
- **Experiment name:** `so101_stacking_big_rings`
- **Model:** pi0.5 (`pi05=True`, `action_horizon=30`)
- **Batch size:** 32 · **LR:** 2.5e-5 cosine (1k warmup) · **EMA:** 0.999
- **Delta actions:** yes · **Default prompt:** `stack the big rings`

## Dataset

- [lorenzouttini/so101_stacking_big_rings](https://huggingface.co/datasets/lorenzouttini/so101_stacking_big_rings)

## W&B

- Training run (to be synced after training completes)

## Repo structure

```
README.md
TRAINING_LOG.md
MODEL_PASSPORT.json
SIGNOFF.json
assets/
step_5000/params/
step_10000/params/
...
```

## Usage

Point `checkpoint_path` at the published step directory:

```python
from openpi.training.config import get_config
from openpi.serving.policy_server import PolicyServer

config = get_config("pi05_so101_stacking_big_rings")
server = PolicyServer(config, checkpoint_path="step_5000/params")
```

Verify integrity before load:

```bash
validate-checkpoint . --require-signoff
```
