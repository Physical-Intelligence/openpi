#!/usr/bin/env bash
set -euo pipefail

OPENPI_SPARK_IMAGE="${OPENPI_SPARK_IMAGE:-openpi-pi05-gb10:local}"
OPENPI_SPARK_DATA_DIR="${OPENPI_SPARK_DATA_DIR:-/var/lib/openpi-spark}"
OPENPI_SPARK_REPORT_DIR="${OPENPI_SPARK_DATA_DIR}/reports"

mkdir -p "${OPENPI_SPARK_REPORT_DIR}"

docker run --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --volume "${OPENPI_SPARK_DATA_DIR}:/openpi_assets" \
    "${OPENPI_SPARK_IMAGE}" \
    python3 scripts/spark/verify_pi05.py \
        --checkpoint-dir /openpi_assets/checkpoints/pi05_base_pytorch \
        --output /openpi_assets/reports/pi05_inference.json

docker run --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --volume "${OPENPI_SPARK_DATA_DIR}:/openpi_assets" \
    "${OPENPI_SPARK_IMAGE}" \
    python3 scripts/train_pytorch.py pi05_spark_smoke \
    2>&1 | tee "${OPENPI_SPARK_REPORT_DIR}/pi05_train_smoke.log"

test -s "${OPENPI_SPARK_REPORT_DIR}/pi05_inference.json"
test -s "${OPENPI_SPARK_REPORT_DIR}/pi05_train_smoke.log"
test -s "${OPENPI_SPARK_DATA_DIR}/training/pi05_spark_smoke/gb10_full_model_smoke/1/model.safetensors"

echo "Inference report: ${OPENPI_SPARK_REPORT_DIR}/pi05_inference.json"
echo "Training log: ${OPENPI_SPARK_REPORT_DIR}/pi05_train_smoke.log"
echo "Fine-tuned checkpoint: ${OPENPI_SPARK_DATA_DIR}/training/pi05_spark_smoke/gb10_full_model_smoke/1"
