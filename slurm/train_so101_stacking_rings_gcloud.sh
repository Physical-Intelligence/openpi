#!/bin/bash
# GCloud Docker training script for SO101 stacking rings
# VM: openpi-so101-80g-2x (us-central1-c, 2x A100 80GB)
# Usage: run this ON the VM directly (or via gcloud ssh)

set -e

# --- Config ---
CONFIG_NAME="pi05_so101_stacking_rings"
EXP_NAME="so101_stacking_rings_run1"
REPO_DIR="/home/ps/openpi"
CHECKPOINT_DIR="${REPO_DIR}/checkpoints"
ASSETS_DIR="${REPO_DIR}/assets/pi05_so101_stacking_rings/assets"
WEIGHTS_DIR="${REPO_DIR}/weights"
LOG_DIR="${REPO_DIR}/logs"
IMAGE="openpi:latest"

mkdir -p "${LOG_DIR}" "${CHECKPOINT_DIR}"

LOG_FILE="${LOG_DIR}/${CONFIG_NAME}_$(date -u +%Y%m%d_%H%M%S).log"

# --- Header ---
{
echo "===================================="
echo "Config: ${CONFIG_NAME}"
echo "Experiment: ${EXP_NAME}"
echo "Image: ${IMAGE}"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)"
echo "Host: $(hostname)"
echo "Started (UTC): $(date -Is --utc)"
echo "===================================="
echo ""
} | tee "${LOG_FILE}"

start_time="$(date +%s)"

# --- Training ---
set +e
sudo docker run --gpus all --rm \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "${WEIGHTS_DIR}:/workspace/repo/weights" \
  -v "${REPO_DIR}/assets:/workspace/repo/assets" \
  -v "${REPO_DIR}/src:/workspace/repo/src" \
  -v "${CHECKPOINT_DIR}:/workspace/repo/checkpoints" \
  -e XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
  -e HF_HOME=/workspace/repo/weights/hf_cache \
  -e PYTHONUNBUFFERED=1 \
  -e WANDB_MODE=offline \
  "${IMAGE}" \
  uv run scripts/train.py "${CONFIG_NAME}" \
    --exp-name="${EXP_NAME}" \
    --assets-dir="/workspace/repo/assets/pi05_so101_stacking_rings/assets" \
    --overwrite 2>&1 | tee -a "${LOG_FILE}"
EXIT_CODE=${PIPESTATUS[0]}
set -e

end_time="$(date +%s)"
elapsed=$(( end_time - start_time ))
hours=$(( elapsed / 3600 ))
minutes=$(( (elapsed % 3600) / 60 ))
seconds=$(( elapsed % 60 ))

# --- Footer ---
{
echo ""
echo "===================================="
echo "Started (UTC):  $(date -Is --utc -d @${start_time} 2>/dev/null || date -u -r ${start_time} +%Y-%m-%dT%H:%M:%S%z)"
echo "Finished (UTC): $(date -Is --utc)"
echo "Runtime: ${hours}h ${minutes}m ${seconds}s"
echo "Exit Code: ${EXIT_CODE}"
echo "Log: ${LOG_FILE}"
echo "===================================="
} | tee -a "${LOG_FILE}"

exit ${EXIT_CODE}
