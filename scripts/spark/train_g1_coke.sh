#!/usr/bin/env bash
set -euo pipefail

: "${OPENPI_G1_COKE_DATASET_REPO_ID:?Set OPENPI_G1_COKE_DATASET_REPO_ID to owner/dataset}"

if [[ ! "${OPENPI_G1_COKE_DATASET_REPO_ID}" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]]; then
    echo "OPENPI_G1_COKE_DATASET_REPO_ID must be an owner/dataset identifier" >&2
    exit 2
fi

OPENPI_SPARK_IMAGE="${OPENPI_SPARK_IMAGE:-openpi-pi05-gb10:local}"
OPENPI_SPARK_DATA_DIR="${OPENPI_SPARK_DATA_DIR:-/var/lib/openpi-spark}"
OPENPI_G1_COKE_NUM_TRAIN_STEPS="${OPENPI_G1_COKE_NUM_TRAIN_STEPS:-10000000}"
OPENPI_G1_COKE_MAX_TRAIN_SECONDS="${OPENPI_G1_COKE_MAX_TRAIN_SECONDS:-172800}"
OPENPI_G1_COKE_EXPERIMENT="${OPENPI_G1_COKE_EXPERIMENT:-g1_coke_pickup_pi05_$(date -u +%Y%m%dT%H%M%SZ)}"
OPENPI_G1_COKE_STATS_PATH="${OPENPI_SPARK_DATA_DIR}/assets/pi05_spark_g1_coke_pickup/${OPENPI_G1_COKE_DATASET_REPO_ID}/norm_stats.json"
OPENPI_G1_COKE_REPORT_DIR="${OPENPI_SPARK_DATA_DIR}/reports"
OPENPI_G1_COKE_TRAINING_DIR="${OPENPI_SPARK_DATA_DIR}/training/pi05_spark_g1_coke_pickup/${OPENPI_G1_COKE_EXPERIMENT}"
OPENPI_LEROBOT_HOME="${OPENPI_SPARK_DATA_DIR}/lerobot"

mkdir -p "${OPENPI_G1_COKE_REPORT_DIR}"

if [[ ! -s "${OPENPI_G1_COKE_STATS_PATH}" ]]; then
    docker run --rm \
        --ipc=host \
        --volume "${OPENPI_SPARK_DATA_DIR}:/openpi_assets" \
        --env "HF_LEROBOT_HOME=/openpi_assets/lerobot" \
        --env "OPENPI_G1_COKE_DATASET_REPO_ID=${OPENPI_G1_COKE_DATASET_REPO_ID}" \
        "${OPENPI_SPARK_IMAGE}" \
        python3 scripts/compute_norm_stats.py pi05_spark_g1_coke_pickup
fi

test -s "${OPENPI_G1_COKE_STATS_PATH}"

resume_args=()
if find "${OPENPI_G1_COKE_TRAINING_DIR}" -mindepth 1 -maxdepth 1 -type d -name '[0-9]*' -print -quit 2>/dev/null | grep -q .; then
    resume_args+=(--resume)
fi

docker run --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --volume "${OPENPI_SPARK_DATA_DIR}:/openpi_assets" \
    --env "HF_LEROBOT_HOME=/openpi_assets/lerobot" \
    --env "OPENPI_G1_COKE_DATASET_REPO_ID=${OPENPI_G1_COKE_DATASET_REPO_ID}" \
    "${OPENPI_SPARK_IMAGE}" \
    python3 scripts/train_pytorch.py pi05_spark_g1_coke_pickup \
        --exp-name "${OPENPI_G1_COKE_EXPERIMENT}" \
        --num-train-steps "${OPENPI_G1_COKE_NUM_TRAIN_STEPS}" \
        --max-train-seconds "${OPENPI_G1_COKE_MAX_TRAIN_SECONDS}" \
        "${resume_args[@]}" \
    2>&1 | tee "${OPENPI_G1_COKE_REPORT_DIR}/${OPENPI_G1_COKE_EXPERIMENT}.log"

echo "Training directory: ${OPENPI_SPARK_DATA_DIR}/training/pi05_spark_g1_coke_pickup/${OPENPI_G1_COKE_EXPERIMENT}"
echo "Training log: ${OPENPI_G1_COKE_REPORT_DIR}/${OPENPI_G1_COKE_EXPERIMENT}.log"
echo "TensorBoard log: ${OPENPI_G1_COKE_TRAINING_DIR}/tensorboard"
