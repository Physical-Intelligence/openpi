#!/usr/bin/env bash
# Smoke-test runner: a 2-step training step (small batch, wandb off) to validate the pipeline end
# to end. Run from the repo root.
#
# Usage:      scripts/smoke_run.sh <config_name> [train_script]   (train_script defaults to scripts/train.py)
# Env knobs:  BATCH (default 8), FSDP (default 2; set 4 for the heavy full-FT MoE configs)
#
# Examples:
#   bash scripts/smoke_run.sh atomic_libero
#   FSDP=4 bash scripts/smoke_run.sh trace_vla_moe
#
# train.py always saves a (~50 GB full-FT) checkpoint at the final step (to the default
# ./checkpoints/<config>/smoke), and the group disk quota is tight (check with `quota -s`, not
# `df`), so remove each run's checkpoint afterward and don't run many heavy smokes at once:
#   rm -rf checkpoints/<config>/smoke
set -euo pipefail

CONFIG="${1:?usage: smoke_run.sh <config_name> [train_script]}"
TRAIN_SCRIPT="${2:-scripts/train.py}"

# Resolve the repo root from this script's own location, so it is path-independent.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

export WANDB_MODE=disabled
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax"

echo "[smoke] config=$CONFIG script=$TRAIN_SCRIPT host=$(hostname) start=$(date -Is)"
rc=0
python "$TRAIN_SCRIPT" "$CONFIG" \
  --exp-name smoke \
  --num-train-steps 2 \
  --batch-size "${BATCH:-8}" \
  --fsdp-devices "${FSDP:-2}" \
  --save-interval 1000 \
  --no-wandb-enabled \
  --overwrite || rc=$?
echo "[smoke] config=$CONFIG exit=$rc end=$(date -Is)"
exit "$rc"
