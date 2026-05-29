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

# --- Make sure `uv` is on PATH. The compute node starts with a clean environment, ---
# so re-source uv's env (installed via the standalone installer into ~/.local/bin).
# If you used `module load uv` instead, replace this line with that module load.
[ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env"
command -v uv >/dev/null || { echo "ERROR: uv not found on PATH"; exit 1; }

# --- Where big files live. /home is only 100GB; use your 500GB user_data. -----
# NOTE: /data/user_data only exists ON A COMPUTE NODE during an active job (it is
# an on-demand AutoFS mount). It is invisible on the login node -- that's normal.
# We stat the full path first to trigger the mount before using it.
export OPENPI_USER_DATA="/data/user_data/${USER}"
ls -ld "$OPENPI_USER_DATA" >/dev/null 2>&1 || true   # trigger AutoFS mount
if [ ! -d "$OPENPI_USER_DATA" ]; then
    echo "ERROR: $OPENPI_USER_DATA not available on this compute node."
    echo "Your user_data may not be provisioned -- ask in #babel-babble (@help-babel)."
    exit 1
fi
export HF_HOME="${OPENPI_USER_DATA}/hf"               # HuggingFace dataset/model cache
export OPENPI_DATA_HOME="${OPENPI_USER_DATA}/openpi_cache"   # base-checkpoint cache
# Babel compute nodes can reach huggingface.co but NOT the "Xet" CDN hosts
# (transfer.xethub.hf.co / *.aws.cdn.hf.co) -> DNS failures. Force the plain
# LFS download path instead, which resolves fine.
export HF_HUB_DISABLE_XET=1
# uv cache stays on /home so it's shared with the login-node `uv sync` (the .venv
# already exists there); no need to put it on /data.
export UV_CACHE_DIR="${HOME}/uv_cache"
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
