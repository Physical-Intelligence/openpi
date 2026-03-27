#!/usr/bin/env bash
set -euo pipefail

cd /home/ziyang10/openpi
# Avoid Xet-backed download path, which can hit stricter shared-IP rate limits.
export HF_HUB_DISABLE_XET=1

python scripts/compute_norm_stats.py --config-name pi05_libero
python scripts/train_pytorch.py pi05_libero \
  --exp-name pi05_test \
  --num-train-steps 20 \
  --batch-size 2 \
  --log-interval 1 \
  --save-interval 20 \
  --model.pytorch-compile-mode None \
  --pytorch-training-precision bfloat16 \
  --model.paligemma-variant gemma_2b \
  --model.action-expert-variant gemma_300m \
  --no-wandb-enabled \
  --pytorch-weight-path ./pi05_base
