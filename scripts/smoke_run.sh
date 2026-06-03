#!/usr/bin/env bash
# Smoke-test runner for a single training config on Slurm.
# Usage:  scripts/smoke_run.sh <config_name> [train_script]
#   train_script defaults to scripts/train.py; pass scripts/train_trace_vla.py for trace/target configs.
# Runs a 2-step training smoke (small batch, wandb disabled) to validate the pipeline.
#
# train.py always saves a (~50 GB full-FT) checkpoint at the final step. The /work/hdd/bgtb group
# quota is tight (check with `quota -s`, not `df`), so we (a) write checkpoints to a HOME scratch
# dir, which has space, and (b) self-clean via a trap that fires even if the run/save crashes.
set -euo pipefail

CONFIG="${1:?usage: smoke_run.sh <config_name> [train_script]}"
TRAIN_SCRIPT="${2:-scripts/train.py}"

REPO=/u/jcpeng2/workspace/mujoco_test/pace/openpi
PY=/u/jcpeng2/workspace/mujoco_test/mujoco_playground/.venv/bin/python
CKPT_BASE="$HOME/.smoke_checkpoints"   # on /u (HOME, ~PB free) — avoids the /work group quota
cd "$REPO"

export WANDB_MODE=disabled
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax"

# Always remove this run's checkpoint on exit (success, failure, or a save crash).
trap 'rm -rf "${CKPT_BASE:?}/${CONFIG}/smoke"' EXIT

echo "[smoke] config=$CONFIG script=$TRAIN_SCRIPT host=$(hostname) start=$(date -Is)"
rc=0
"$PY" "$TRAIN_SCRIPT" "$CONFIG" \
  --exp-name smoke \
  --checkpoint-base-dir "$CKPT_BASE" \
  --num-train-steps 2 \
  --batch-size "${BATCH:-8}" \
  --fsdp-devices "${FSDP:-2}" \
  --save-interval 1000 \
  --no-wandb-enabled \
  --overwrite || rc=$?
echo "[smoke] config=$CONFIG exit=$rc end=$(date -Is)"
exit "$rc"
