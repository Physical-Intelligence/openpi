#!/usr/bin/env bash
# Test 5 (submit): GPU smoke run of the REAL sbatch via env overrides — 2 datasets,
# 20 steps each, tiny batch, capped norm-stats frames. Validates the full pipeline:
# download -> norm stats -> train (--overwrite) -> scratch cleanup -> train (--resume).
# Run this ON A LOGIN NODE (it submits with sbatch). Then run 06_check_smoke.sh after it ends.
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 2

export LIST="examples/babel/tests/smoke_datasets.txt"
export EXP_NAME="${EXP_NAME:-yam_smoke}"
export STEPS_PER_DATASET="${STEPS_PER_DATASET:-20}"
export NORM_STATS_MAX_FRAMES="${NORM_STATS_MAX_FRAMES:-200}"
export EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:---batch-size=8}"   # 8 is divisible by 1 GPU

# Smaller footprint than the real run: 1 GPU, short wall time, debug-ish partition.
PARTITION="${PARTITION:-debug}"
GRES="${GRES:-gpu:1}"
TIME="${TIME:-00:40:00}"

echo "Submitting smoke run:"
echo "  list=$LIST exp=$EXP_NAME steps/dataset=$STEPS_PER_DATASET batch via '$EXTRA_TRAIN_ARGS'"
echo "  partition=$PARTITION gres=$GRES time=$TIME"
echo "  -> checkpoints under: /data/user_data/\$USER/openpi_checkpoints/pi05_BiYAMMolmoAct2_loravlm/$EXP_NAME"
echo

sbatch \
    --job-name=yam_smoke \
    --partition="$PARTITION" \
    --gres="$GRES" \
    --time="$TIME" \
    --export=ALL \
    examples/babel/train_foldclo_loravlm.sbatch

echo
echo "Watch with:  squeue --me   and   tail -f yam_smoke-<jobid>.out"
echo "When it finishes, run:  bash examples/babel/tests/06_check_smoke.sh <jobid>"
