# Lipbalm Pick — Single Right-Arm Pi0.5 Fine-tuning

Fine-tunes a pi0.5 model on the lipbalm pick task using only the right arm from bimanual ALOHA (WidowX) recordings.

## Data

Three raw bimanual datasets under `data/raw/`:
- `mobileai-lipbalm` (10 episodes, 1738 frames)
- `mobileai-lipbalm1` (10 episodes)
- `mobileai-lipbalm2` (10 episodes)

Each has 14-dim actions/state (left + right arm), 3 cameras. The processing script extracts:
- **Right arm only** (indices 7:14 → 7-dim: 6 joints + gripper)
- **2 cameras**: `cam_high` + `cam_right_wrist`

## Setup

Process the data first:

```bash
python experiments/data/process_single_arm.py --merge
```

This creates individual processed datasets (`*-right`) plus a merged `mobileai-lipbalm-all-right` under `data/processed/`.

## Training

Edit `configs/lipbalm.yaml` to set your target server and hyperparams, then:

```bash
# Launch on a remote server from ~/.ssh/config
./experiments/train.sh aws-L40S-48gb

# Or run locally
LEROBOT_HOME=experiments/data/processed python experiments/run_train.py \
    --config experiments/configs/lipbalm.yaml
```

The script rsyncs code and processed data to the server, then starts training in a tmux session.

## Evaluation

```bash
# Launch on a remote server
./experiments/eval.sh aws-L40S-48gb

# Or run locally with a specific checkpoint step
LEROBOT_HOME=experiments/data/processed python experiments/run_eval.py \
    --config experiments/configs/lipbalm.yaml --checkpoint_step 20000
```

## Structure

```
experiments/
├── configs/lipbalm.yaml          # Server, hyperparams, data paths
├── data/
│   ├── raw/                      # Symlinks to bimanual recordings
│   ├── processed/                # Single right-arm datasets (generated)
│   └── process_single_arm.py     # Bimanual → right arm extraction
├── transforms.py                 # Single-arm input/output transforms
├── config.py                     # Builds TrainConfig from YAML
├── run_train.py                  # Training entrypoint
├── run_eval.py                   # Eval entrypoint
├── train.sh                      # Remote training launcher
└── eval.sh                       # Remote eval launcher
```

## Config

All training parameters live in `configs/lipbalm.yaml`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training.num_train_steps` | 20000 | Total training steps |
| `training.batch_size` | 64 | Batch size |
| `training.peak_lr` | 5e-5 | Peak learning rate |
| `server.name` | `aws-L40S-48gb` | SSH config host for remote jobs |
| `server.gpu_count` | 1 | Number of GPUs (>1 uses torchrun DDP) |

## Notes

- No existing openpi source files are modified. Everything is self-contained here.
- The `adapt_to_pi` conversion is **not applied** since this is a WidowX robot, not Trossen.
- Camera mapping follows the DROID pattern: `cam_right_wrist` maps to the `left_wrist_0_rgb` model slot, with `right_wrist_0_rgb` zero-padded.
