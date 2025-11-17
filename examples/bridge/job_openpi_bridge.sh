#!/bin/bash

# Set up common env variables. Always keep this line in your job script!
# Note: Don't use set -e before sourcing env.sh as it may fail on non-critical operations
set +e  # Temporarily allow failures
source ${0%/*}/env.sh
set -e  # Exit on any command failure from this point forward

# env.sh will export three variables:
# _JOB_DIR - the root dir that contains all files related to this job
# _DEV_DIR - the directory which hosts the user uploaded code dirs
# _TB_DIR  - the output dir for storing the job's results

#SBATCH --job-name=openpi-bridge
#SBATCH --output=slurm/logs/%A_openpi_bridge.out
#SBATCH --error=slurm/logs/%A_openpi_bridge.err
#SBATCH --time=71:59:59
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=104
#SBATCH --mem=500G

## Entering the dev directory
cd ${_DEV_DIR}
rm -f core.python.*  # remove any previous core files

# Create logs directory if it doesn't exist
# mkdir -p slurm/logs

# ======================
# Environment Setup
# ======================

# Set up environment variables for HuggingFace and dataset paths
# Needed for lerobot dataset
export HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-${DATASET_ROOT}/datasets/openx}
export HF_DATASETS_CACHE=${DATASET_ROOT}/datasets/openx/bridge_lerobot_cache/datasets
# export HF_HOME=${HF_HOME:-/data/jerry/datasets/openx/cache}

# Set OpenPI cache directory to use uploaded weights/tokenizer
# This prevents downloading from gs://big_vision
export OPENPI_DATA_HOME=/data/cache/openpi

# PyTorch memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ======================
# GPU Discovery
# ======================
# echo "=========================================="
# echo "SLURM Job Information"
# echo "=========================================="
# echo "Job ID: $SLURM_JOB_ID"
# echo "Node: $SLURMD_NODENAME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
NUM_GPU="$(nvidia-smi --list-gpus | wc -l)"
echo "Number of GPUs: $NUM_GPU"
# nvidia-smi --query-gpu=name,memory.total --format=csv
# echo "=========================================="

# Note: --standalone mode handles distributed setup automatically
# No need to manually set MASTER_ADDR/MASTER_PORT for single-node training

# ======================
# Training Configuration
# ======================

# Checkpoint directory - save to cluster output directory
# This prevents filling up the code directory and ensures checkpoints are preserved
export CHECKPOINT_DIR=${_TB_DIR}/checkpoints

# Experiment name - modify as needed
EXP_NAME="default"

# Total batch size across all GPUs (will be divided by NUM_GPU)
# For 8x H100 with 80GB each, you can use a larger batch size
TOTAL_BATCH_SIZE=128  # 16 per GPU

# Number of training steps
NUM_TRAIN_STEPS=100000  # Adjust based on your needs

# Number of data loader workers per GPU
NUM_WORKERS=8

# Save checkpoint every N steps
SAVE_INTERVAL=10000

# Learning rate (default: 3e-5)
LEARNING_RATE=3e-5

# ======================
# Cluster-Specific Paths
# ======================
# IMPORTANT: Update these paths to match where you uploaded the files on your cluster

# 1. PyTorch Weights Path
#    Upload: /home/jerry/.cache/openpi/pi0_base_pytorch/ to cluster
#    Should contain: model.safetensors, config.json
PYTORCH_WEIGHT_PATH="/data/cache/openpi/pi0_base_pytorch"
export TORCHINDUCTOR_CACHE_DIR=$HOME/.cache/torchinductor

# 2. Assets Directory (for normalization stats)
#    Upload: assets/pi0_bridge_finetune/ to cluster
#    Should contain: bridge_lerobot_224/norm_stats.json
#    The code will look for: ${ASSETS_DIR}/bridge_lerobot_224/norm_stats.json
# ASSETS_DIR="/data/cache/openpi/assets/pi0_bridge_finetune"

# Note: The asset_id "bridge_lerobot_224" is set in config.py and will be appended
# to ASSETS_DIR to find the norm stats

# ======================
# Launch Training with torchrun
# ======================

# Enter the open_pi_zero directory
cd code/openpi

# Use existing UV_CACHE_DIR from env.sh for Python installations
export UV_PYTHON_INSTALL_DIR=${UV_CACHE_DIR}/python

# Increase UV HTTP timeout for large packages (default is 30s)
export UV_HTTP_TIMEOUT=300  # 5 minutes

echo "Installing dependencies with uv..."
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
echo "Dependencies installed successfully!"

echo "Starting OpenPI training..."
echo "Experiment: $EXP_NAME"
echo "Total Batch Size: $TOTAL_BATCH_SIZE"
echo "Batch Size per GPU: $((TOTAL_BATCH_SIZE / NUM_GPU))"
echo "Training Steps: $NUM_TRAIN_STEPS"
echo ""
echo "Paths:"
echo "  PyTorch Weights: $PYTORCH_WEIGHT_PATH"
# echo "  Assets Dir: $ASSETS_DIR"
echo "  LeRobot Home: $HF_LEROBOT_HOME"
echo "  Checkpoint Dir: $CHECKPOINT_DIR"
echo "=========================================="

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPU \
  scripts/train_pytorch.py pi0_bridge_finetune \
  --exp-name $EXP_NAME \
  --num-train-steps $NUM_TRAIN_STEPS \
  --batch-size $TOTAL_BATCH_SIZE \
  --num-workers $NUM_WORKERS \
  --save-interval $SAVE_INTERVAL \
  --lr-schedule.peak-lr $LEARNING_RATE \
  --pytorch-weight-path $PYTORCH_WEIGHT_PATH \
  --checkpoint-base-dir $CHECKPOINT_DIR
  # --data.assets.assets-dir $ASSETS_DIR

# Capture the training exit code
TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "Training completed successfully!"
    echo "Check outputs at: $CHECKPOINT_DIR/pi0_bridge_finetune/$EXP_NAME/"
    echo "=========================================="
else
    echo "=========================================="
    echo "Training failed with exit code $TRAIN_EXIT_CODE"
    echo "Check logs for errors"
    echo "=========================================="
fi

# Output the exit code for record. Keep this line if you want to show your job's
# exit status correctly.
# The last line of the script's output will be copied to the timestamp file, with
# a timestamp appended by ``docker_run.sh``.
echo $TRAIN_EXIT_CODE:
