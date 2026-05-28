#!/bin/bash
#SBATCH --job-name=foldclo
#SBATCH --partition=general          # general: max 8 GPUs, up to 2 days, sbatch-only
#SBATCH --gres=gpu:4                 # ask for 4 GPUs on ONE machine
#SBATCH --cpus-per-task=32           # CPUs for the data loader (8 per GPU is plenty)
#SBATCH --mem=128G                   # system RAM (not GPU RAM)
#SBATCH --time=2-00:00:00            # 2 days (the max for the 'general' partition)
#SBATCH --output=%x-%j.out           # log file: <job-name>-<job-id>.out
#SBATCH --error=%x-%j.out            # send errors to the same file

# ===========================================================================
#  Fold-cloth pi05 fine-tune on 4 GPUs (Babel / CMU).
#  Submit from the repo root with:   sbatch examples/babel/train_foldclo.sbatch
#  Everything below runs ON the compute node Slurm gives you, NOT the login node.
# ===========================================================================

set -euo pipefail

# --- Where big files live. /home is only 100GB; use your 500GB user_data. -----
export OPENPI_USER_DATA="/data/user_data/${USER}"
export HF_HOME="${OPENPI_USER_DATA}/hf"               # HuggingFace dataset/model cache
export OPENPI_DATA_HOME="${OPENPI_USER_DATA}/openpi_cache"   # base-checkpoint cache
export UV_CACHE_DIR="${OPENPI_USER_DATA}/uv_cache"    # uv package cache
mkdir -p "$HF_HOME" "$OPENPI_DATA_HOME" "$UV_CACHE_DIR"

# --- Secrets (set these in your shell before sbatch, or hard-code here) -------
#   export HF_TOKEN=hf_xxx        # needed to pull the private/gated dataset
#   export WANDB_API_KEY=xxx      # needed for Weights & Biases logging
# If you don't want W&B at all, add `--no-wandb-enabled` to the train.py command
# below, or run `export WANDB_MODE=offline`.

# --- Let JAX use almost all of each GPU's memory. -----------------------------
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

cd "$SLURM_SUBMIT_DIR"   # the directory you ran `sbatch` from (your repo root)

echo "Running on host: $(hostname)"
echo "GPUs visible to this job:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# --- 1) Normalization statistics (only needs to run once per config) ----------
# Safe to re-run; it is quick relative to training. Skips nothing but is cheap.
uv run scripts/compute_norm_stats.py --config-name pi05_foldclo_babel

# --- 2) Training --------------------------------------------------------------
# The config already sets batch_size=64 and fsdp_devices=4 (4-GPU model sharding).
uv run scripts/train.py pi05_foldclo_babel \
    --exp-name="foldclo_prompts_4gpu" \
    --checkpoint-base-dir="${OPENPI_USER_DATA}/openpi_checkpoints" \
    --overwrite

echo "DONE."
