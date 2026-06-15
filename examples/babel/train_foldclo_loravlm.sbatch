#!/bin/bash
#SBATCH --job-name=yam_continual
#SBATCH --partition=general          # general: max 8 GPUs, up to 2 days, sbatch-only
#SBATCH --gres=gpu:4                 # ask for 4 GPUs on ONE machine (data-parallel: 1 replica each)
#SBATCH --cpus-per-task=32           # CPUs for the data loader (8 per GPU is plenty)
#SBATCH --mem=128G                   # system RAM (not GPU RAM)
#SBATCH --time=2-00:00:00            # 2 days (the max for the 'general' partition)
#SBATCH --output=%x-%j.out           # log file: <job-name>-<job-id>.out
#SBATCH --error=%x-%j.out            # send errors to the same file
#SBATCH --mail-type=END,FAIL,TIME_LIMIT   # native SLURM email (confirmed working on Babel)
#SBATCH --mail-user=leokswang@gmail.com

# ===========================================================================
#  CONTINUAL pi05 LoRA-VLM fine-tune across MANY YAM datasets (Babel / CMU).
#
#  Walks every dataset listed in examples/babel/converted_yam_data.txt IN ORDER
#  and finetunes ONE evolving model through them (each dataset continues from the
#  previous dataset's checkpoint). For each dataset:
#    1. wait until it exists on https://huggingface.co/leokswang (re-poll if not),
#    2. compute its per-dataset normalization stats (this also downloads it to /scratch),
#    3. train STEPS_PER_DATASET more steps (continuing the same model),
#    4. delete the local /scratch copy before moving to the next dataset.
#
#  Finetuning scheme (config pi05_BiYAMMolmoAct2_loravlm):
#    - LoRA on the VLM (paligemma) -> base VLM weights frozen, only adapters train
#    - FULL finetuning of the action expert
#  Checkpoints: kept in user_data, only the 3 most recent retained (config max_to_keep=3).
#  Submit from the repo root with:
#      sbatch examples/babel/train_foldclo_loravlm.sbatch
#  Everything below runs ON the compute node Slurm gives you, NOT the login node.
# ===========================================================================

set -uo pipefail   # not -e: we want the dataset loop to handle errors itself

# All of these are env-overridable so the smoke test can reuse this exact script
# (e.g. LIST=... STEPS_PER_DATASET=20 EXP_NAME=yam_smoke sbatch ... this_script).
CONFIG_NAME="${CONFIG_NAME:-pi05_BiYAMMolmoAct2_loravlm}"
EXP_NAME="${EXP_NAME:-yam_continual_4gpu}"
LIST="${LIST:-examples/babel/converted_yam_data.txt}"
STEPS_PER_DATASET="${STEPS_PER_DATASET:-2000}"   # additional train steps per dataset
POLL_SECONDS="${POLL_SECONDS:-300}"              # re-check interval when caught up to the upload frontier
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"         # extra flags appended to train.py (e.g. --batch-size=4 for smoke)
NORM_STATS_MAX_FRAMES="${NORM_STATS_MAX_FRAMES:-}"   # cap norm-stats frames (smoke); empty = full dataset

# --- Make sure `uv` is on PATH. The compute node starts with a clean environment. ---
[ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env"
command -v uv >/dev/null || { echo "ERROR: uv not found on PATH"; exit 1; }

# --- Checkpoints + base-weight cache live on user_data (persists across jobs). -------
# /data/user_data only exists ON A COMPUTE NODE (on-demand AutoFS mount); stat to trigger it.
export OPENPI_USER_DATA="/data/user_data/${USER}"
ls -ld "$OPENPI_USER_DATA" >/dev/null 2>&1 || true   # trigger AutoFS mount
if [ ! -d "$OPENPI_USER_DATA" ]; then
    echo "ERROR: $OPENPI_USER_DATA not available on this compute node."
    echo "Your user_data may not be provisioned -- ask in #babel-babble (@help-babel)."
    exit 1
fi
export OPENPI_DATA_HOME="${OPENPI_USER_DATA}/openpi_cache"   # base-checkpoint cache
CKPT_DIR="${OPENPI_USER_DATA}/openpi_checkpoints"

# --- Datasets are staged on /scratch (5.5T, temporary), NOT user_data/hf. -----------
# /scratch/$USER (e.g. /scratch/leokingw) does not exist yet; mkdir -p is idempotent.
export HF_LEROBOT_HOME="/scratch/${USER}/lerobot"   # lerobot downloads datasets here
export HF_HOME="/scratch/${USER}/hf"                # hub blob cache also on /scratch
# Babel compute nodes can reach huggingface.co but NOT the "Xet" CDN hosts -> force plain LFS.
export HF_HUB_DISABLE_XET=1
# uv cache stays on /home so it's shared with the login-node `uv sync` (the .venv exists there).
export UV_CACHE_DIR="${HOME}/uv_cache"
mkdir -p "/scratch/${USER}" "$HF_LEROBOT_HOME" "$HF_HOME" "$OPENPI_DATA_HOME" "$UV_CACHE_DIR"

# --- Secrets (set these in your shell before sbatch, or hard-code here) -------
#   export HF_TOKEN=hf_xxx        # needed to pull private/gated datasets
#   export WANDB_API_KEY=xxx      # needed for Weights & Biases logging

# --- Let JAX use almost all of each GPU's memory. -----------------------------
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

cd "$SLURM_SUBMIT_DIR"   # the directory you ran `sbatch` from (your repo root)

echo "Running on host: $(hostname)"
echo "Config: ${CONFIG_NAME}   Exp: ${EXP_NAME}"
echo "Dataset list: ${LIST}   Steps/dataset: ${STEPS_PER_DATASET}"
echo "Checkpoints:  ${CKPT_DIR}/${CONFIG_NAME}/${EXP_NAME} (keeping 3 most recent)"
echo "Scratch:      ${HF_LEROBOT_HOME}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# --- Does a dataset repo exist on HF yet? (exit 0 = yes, 1 = no) ---------------
dataset_exists() {  # $1 = repo_id
    uv run --no-project --with huggingface_hub python - "$1" <<'PY'
import sys
from huggingface_hub import HfApi
sys.exit(0 if HfApi().repo_exists(sys.argv[1], repo_type="dataset") else 1)
PY
}

# --- Main loop: continual finetune through every dataset in order --------------
step_total=0
first=1
while IFS= read -r repo || [ -n "$repo" ]; do
    repo="${repo%%[[:space:]]*}"   # strip trailing whitespace/CR (repo ids have no spaces)
    [ -z "$repo" ] && continue

    # 1) Wait until the dataset is uploaded (training caught up to the upload frontier).
    until dataset_exists "$repo"; do
        echo "[wait] $(date -Iseconds) $repo not on HF yet; re-poll in ${POLL_SECONDS}s"
        sleep "$POLL_SECONDS"
    done
    echo "[data] $(date -Iseconds) $repo is available"

    # 2) Per-dataset norm stats. First access downloads the dataset into $HF_LEROBOT_HOME.
    norm_args=(--config-name "$CONFIG_NAME" --repo-id "$repo")
    [ -n "$NORM_STATS_MAX_FRAMES" ] && norm_args+=(--max-frames "$NORM_STATS_MAX_FRAMES")
    if ! uv run scripts/compute_norm_stats.py "${norm_args[@]}"; then
        echo "[FAIL] norm stats failed for $repo -- aborting so SLURM emails FAIL"
        exit 1
    fi

    # 3) Continual train: bump the cumulative step budget; --overwrite seeds the very first
    #    dataset from the config's base weight_loader, --resume continues thereafter.
    step_total=$(( step_total + STEPS_PER_DATASET ))
    resume_flag="--resume"; [ "$first" = 1 ] && resume_flag="--overwrite"
    echo "[train] $(date -Iseconds) $repo  (${resume_flag}, num_train_steps=${step_total})"
    if ! uv run scripts/train.py "$CONFIG_NAME" \
        --exp-name="$EXP_NAME" \
        --checkpoint-base-dir="$CKPT_DIR" \
        --data.repo-id="$repo" \
        --num-train-steps="$step_total" \
        "$resume_flag" $EXTRA_TRAIN_ARGS; then
        echo "[FAIL] training failed for $repo -- aborting so SLURM emails FAIL"
        exit 1
    fi
    first=0

    # 4) Free /scratch before the next dataset (Hub copy is the source of truth).
    rm -rf "${HF_LEROBOT_HOME:?}/${repo}"
    echo "[clean] removed ${HF_LEROBOT_HOME}/${repo}"
done < "$LIST"

echo "ALL DATASETS DONE. $(date -Iseconds)"
