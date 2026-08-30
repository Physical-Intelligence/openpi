#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
: "${DATASET_REPO_ID:=local/ur5_twinwrist}"
: "${TRAIN_STEPS:=10000}" "${BATCH_SIZE:=16}" "${FSDP_DEVICES:=4}"
gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
(( gpu_count >= FSDP_DEVICES )) || { echo "need $FSDP_DEVICES GPUs, found $gpu_count" >&2; exit 2; }
(( BATCH_SIZE % FSDP_DEVICES == 0 )) || { echo "batch size must be divisible by devices" >&2; exit 2; }
test -f uv.lock; test -n "$(git rev-parse HEAD)"
sha256sum uv.lock pyproject.toml examples/ur5_twinwrist/config.example.toml examples/ur5_twinwrist/ACTION_SPACE.md
mkdir -p reports; { git status --short --branch; uv --version; .venv/bin/python --version; nvidia-smi; } > "reports/h100_environment_$(date +%Y%m%dT%H%M%S).txt"
uv run scripts/compute_norm_stats.py --config-name pi05_ur5_twinwrist
uv run scripts/train.py pi05_ur5_twinwrist --data.repo-id "$DATASET_REPO_ID" --num-train-steps "$TRAIN_STEPS" --batch-size "$BATCH_SIZE" --fsdp-devices "$FSDP_DEVICES"
