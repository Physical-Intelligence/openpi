#!/usr/bin/env bash
set -euo pipefail

cd /home/jiajun/SA-WM/vlas/openpi
source examples/libero/.venv/bin/activate

export PYTHONPATH=/home/jiajun/SA-WM/simulations/LIBERO:${PYTHONPATH:-}
export LIBERO_CONFIG_PATH=/home/jiajun/SA-WM/configs/libero
export MPLCONFIGDIR=/tmp/matplotlib
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

OUT_DIR=/home/jiajun/SA-WM/experiments/openpi_pi05_libero10_50trial
mkdir -p "$OUT_DIR"

python examples/libero/main.py \
  --args.host 127.0.0.1 \
  --args.port 8000 \
  --args.task-suite-name libero_10 \
  --args.task-start 0 \
  --args.num-tasks 0 \
  --args.num-trials-per-task 50 \
  --args.video-out-path "$OUT_DIR" \
  --args.results-out-path "$OUT_DIR/results.jsonl" \
  --args.summary-out-path "$OUT_DIR/summary.json" \
  2>&1 | tee "$OUT_DIR/eval.log"
