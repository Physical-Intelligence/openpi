# OpenPI Pi0.5 LoRA Fine-tuning

Fine-tunes a Pi0.5 (3B param VLA) model via LoRA on single right-arm ALOHA (WidowX) recordings. Extracts right arm data from bimanual datasets, converts to LeRobot v2.1 format, trains with JAX backend, and evaluates action prediction accuracy.

## Directory Structure

```
experiments/
├── train.sh                          # Remote training launcher
├── eval.sh                           # Remote eval launcher
├── config.py                         # Builds TrainConfig from YAML
├── transforms.py                     # Single-arm input/output transforms
├── configs/
│   ├── lipbalm.yaml                  # Lipbalm pick task config
│   └── marker_pick.yaml              # Marker pick task config
├── data/
│   ├── process_single_arm.py         # Bimanual → right arm extraction
│   ├── convert_to_lerobot_v21.py     # v3.0 → v2.1 format conversion
│   └── validate_dataset.py           # Pre-training data validation
├── train/
│   └── run_train.py                  # Training entrypoint
├── eval/
│   ├── run_eval.py                   # Evaluation entrypoint
│   └── export_results.py             # Results export + Excel tracker
└── results/                          # Exported metrics, plots (gitignored)
```

## Workflow

### 1. Data Preparation

Source datasets are bimanual ALOHA recordings with 14-dim actions/state and 3 cameras. The pipeline extracts right arm only (indices 7:14 → 7-dim) with cam_high + cam_right_wrist.

```bash
# Symlink raw datasets
ln -s /path/to/dataset experiments/data/raw/dataset_name

# Extract right arm and merge
python experiments/data/process_single_arm.py \
    --raw-dir experiments/data/raw \
    --out-dir experiments/data/processed \
    --datasets dataset1 dataset2 dataset3 \
    --merge --merge-name dataset-all-right
```

### 2. Format Conversion

OpenPI requires LeRobot v2.1 format. Convert individual datasets and merge:

```bash
HF_LEROBOT_HOME=/path/to/output python experiments/data/convert_to_lerobot_v21.py \
    --src-dirs processed/dataset1-right processed/dataset2-right ... \
    --repo-id dataset-all-right \
    --task-name "pick object"
```

### 3. Validation

Verify the converted dataset before training:

```bash
python experiments/data/validate_dataset.py \
    --repo-id dataset-all-right \
    --source-dirs processed/dataset1-right processed/dataset2-right ... \
    --lerobot-home /path/to/output
```

Checks: episode/frame counts, action/state dimensions (7-dim), image integrity, value accuracy vs source.

### 4. Configuration

Create a YAML config in `configs/`. Required fields:

```yaml
experiment:
  name: task_pi05_lora          # Used to derive prompt and config name
  project_name: openpi-task

data:
  merged_name: dataset-all-right  # repo_id under HF_LEROBOT_HOME
  # default_prompt: "pick object" # Optional — auto-derived from experiment name

training:
  num_train_steps: 10000
  batch_size: 16
  use_lora: true
  peak_lr: 5.0e-5

server:
  name: ssh-host-name
  remote_dir: ~/openpi
  remote_data_dir: ~/data/task
```

### 5. Training

```bash
# Remote (rsyncs code + data, launches in tmux)
./experiments/train.sh <server_name> experiments/configs/<task>.yaml

# Or directly on a GPU machine
HF_LEROBOT_HOME=/path/to/data python experiments/train/run_train.py \
    --config experiments/configs/<task>.yaml
```

LoRA fine-tuning uses `gemma_2b_lora` (PaliGemma) + `gemma_300m_lora` (action expert) with frozen base weights. Checkpoints saved every `save_interval` steps.

### 6. Evaluation

```bash
# Remote
./experiments/eval.sh <server_name> experiments/configs/<task>.yaml [checkpoint_step]

# Or directly
HF_LEROBOT_HOME=/path/to/data python experiments/eval/run_eval.py \
    --config experiments/configs/<task>.yaml \
    --num_episodes 10 --frames_per_episode 20
```

Reports per-joint MAE (mean absolute error in radians) across sampled episodes.

### 7. Export Results

```bash
python experiments/eval/export_results.py \
    --server <server_name> \
    --run-name task_pi05_lora \
    --config experiments/configs/<task>.yaml \
    --comments "first run"
```

Fetches logs from server, generates loss plots, updates `experiment_tracker.xlsx`.

## Adding a New Task

1. Symlink raw datasets into `experiments/data/raw/`
2. Run `process_single_arm.py` with `--datasets` and `--merge`
3. Convert to v2.1 with `convert_to_lerobot_v21.py --src-dirs ... --task-name "your task"`
4. Validate with `validate_dataset.py`
5. Create `experiments/configs/<task>.yaml` (copy an existing one, update names and data paths)
6. Train and evaluate

## Config Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training.num_train_steps` | 10000 | Total training steps |
| `training.batch_size` | 16 | Batch size (16 fits L40S 48GB with LoRA) |
| `training.use_lora` | true | LoRA vs full fine-tuning |
| `training.peak_lr` | 5e-5 | Peak learning rate (cosine decay) |
| `training.warmup_steps` | 1000 | LR warmup steps |
| `training.save_interval` | 2000 | Checkpoint save frequency |
| `server.name` | — | SSH config host for remote jobs |
| `server.gpu_count` | 1 | Number of GPUs (>1 uses torchrun DDP) |

## Architecture Notes

- No existing openpi source files are modified. All code is self-contained in `experiments/`.
- Camera mapping follows the DROID pattern: `cam_right_wrist` → `left_wrist_0_rgb` slot, `right_wrist_0_rgb` zero-padded.
- The `adapt_to_pi` conversion is not applied (WidowX robot, not Trossen).
- LeRobot Hub checks are monkey-patched to skip for local datasets.
- JAX backend is used for training (supports LoRA + freeze_filter). PyTorch trainer does not support LoRA.
