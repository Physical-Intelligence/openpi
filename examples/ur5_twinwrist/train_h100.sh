#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
: "${DATASET_REPO_ID:=local/ur5_twinwrist}"
: "${TRAIN_STEPS:=10000}" "${BATCH_SIZE:=16}" "${FSDP_DEVICES:=4}"
: "${LEROBOT_DATA_ROOT:?set LEROBOT_DATA_ROOT to the directory containing local/ur5_twinwrist}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a visible_devices <<< "$CUDA_VISIBLE_DEVICES"
  gpu_count=${#visible_devices[@]}
else
  gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
fi
(( gpu_count >= FSDP_DEVICES )) || { echo "need $FSDP_DEVICES GPUs, found $gpu_count" >&2; exit 2; }
(( BATCH_SIZE % FSDP_DEVICES == 0 )) || { echo "batch size must be divisible by devices" >&2; exit 2; }
dataset_path="$LEROBOT_DATA_ROOT/$DATASET_REPO_ID"
test -d "$dataset_path" || { echo "dataset not found: $dataset_path" >&2; exit 2; }
current_commit=$(git rev-parse HEAD)
test -f uv.lock; test -n "$current_commit"
if [[ -n "${EXPECTED_GIT_COMMIT:-}" && "$current_commit" != "$EXPECTED_GIT_COMMIT" ]]; then
  echo "git commit mismatch: expected $EXPECTED_GIT_COMMIT, found $current_commit" >&2
  exit 2
fi
current_lock_hash=$(sha256sum uv.lock | awk '{print $1}')
if [[ -n "${EXPECTED_UV_LOCK_SHA256:-}" && "$current_lock_hash" != "$EXPECTED_UV_LOCK_SHA256" ]]; then
  echo "uv.lock hash mismatch" >&2
  exit 2
fi
mapfile -t config_files < <(find examples/ur5_twinwrist/config -maxdepth 1 -type f -name '*.yaml' -print | sort)
(( ${#config_files[@]} == 5 )) || { echo "expected five project YAML files" >&2; exit 2; }
sha256sum uv.lock pyproject.toml docs/wrist/05_动作空间与数据格式.md "${config_files[@]}"
uv run python -m examples.ur5_twinwrist.config_loader
export HF_LEROBOT_HOME="$LEROBOT_DATA_ROOT"
export UR5_TWINWRIST_DATASET_REPO_ID="$DATASET_REPO_ID"
mkdir -p reports
{
  git status --short --branch
  git rev-parse HEAD
  uv --version
  .venv/bin/python --version
  sha256sum uv.lock pyproject.toml docs/wrist/05_动作空间与数据格式.md "${config_files[@]}"
  nvidia-smi
  uv run python -c 'import jax; print(jax.__version__); print(jax.devices())'
  uv pip freeze
} > "reports/h100_environment_$(date +%Y%m%dT%H%M%S).txt"
uv run scripts/compute_norm_stats.py --config-name pi05_ur5_twinwrist
uv run scripts/train.py pi05_ur5_twinwrist --data.repo-id "$DATASET_REPO_ID" --num-train-steps "$TRAIN_STEPS" --batch-size "$BATCH_SIZE" --fsdp-devices "$FSDP_DEVICES"
