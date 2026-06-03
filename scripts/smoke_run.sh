#!/usr/bin/env bash
# Smoke-test runner for a single training config on Slurm.
# Usage:  scripts/smoke_run.sh <config_name> [train_script]
#   train_script defaults to scripts/train.py; pass scripts/train_trace_vla.py for trace/target configs.
# Runs a 2-step training smoke (small batch, wandb disabled) to validate the pipeline.
# Cleans nothing — caller is responsible for `rm -rf checkpoints/<config>/smoke` between runs.
set -euo pipefail

CONFIG="${1:?usage: smoke_run.sh <config_name> [train_script]}"
TRAIN_SCRIPT="${2:-scripts/train.py}"

REPO=/u/jcpeng2/workspace/mujoco_test/pace/openpi
PY=/u/jcpeng2/workspace/mujoco_test/mujoco_playground/.venv/bin/python
cd "$REPO"

export WANDB_MODE=disabled
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax"

echo "[smoke] config=$CONFIG script=$TRAIN_SCRIPT host=$(hostname) start=$(date -Is)"
"$PY" "$TRAIN_SCRIPT" "$CONFIG" \
  --exp-name smoke \
  --num-train-steps 2 \
  --batch-size "${BATCH:-8}" \
  --fsdp-devices "${FSDP:-2}" \
  --save-interval 1000 \
  --no-wandb-enabled \
  --overwrite
echo "[smoke] config=$CONFIG exit=$? end=$(date -Is)"
