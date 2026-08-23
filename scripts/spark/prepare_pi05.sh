#!/usr/bin/env bash
set -euo pipefail

OPENPI_SPARK_IMAGE="${OPENPI_SPARK_IMAGE:-openpi-pi05-gb10:local}"
OPENPI_SPARK_DATA_DIR="${OPENPI_SPARK_DATA_DIR:-/var/lib/openpi-spark}"
OPENPI_SPARK_SOURCE_CHECKPOINT="${OPENPI_SPARK_SOURCE_CHECKPOINT:-gs://openpi-assets/checkpoints/pi05_base}"
OPENPI_SPARK_OUTPUT_CHECKPOINT="${OPENPI_SPARK_OUTPUT_CHECKPOINT:-/openpi_assets/checkpoints/pi05_base_pytorch}"

mkdir -p "${OPENPI_SPARK_DATA_DIR}"

if [[ -s "${OPENPI_SPARK_DATA_DIR}/checkpoints/pi05_base_pytorch/model.safetensors" ]]; then
    echo "pi0.5 PyTorch checkpoint already exists; preserving it"
    exit 0
fi

docker run --rm \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --volume "${OPENPI_SPARK_DATA_DIR}:/openpi_assets" \
    --env JAX_PLATFORMS=cpu \
    --env OPENPI_DATA_HOME=/openpi_assets \
    --env "OPENPI_SPARK_SOURCE_CHECKPOINT=${OPENPI_SPARK_SOURCE_CHECKPOINT}" \
    --env "OPENPI_SPARK_OUTPUT_CHECKPOINT=${OPENPI_SPARK_OUTPUT_CHECKPOINT}" \
    "${OPENPI_SPARK_IMAGE}" \
    bash -lc '
        set -euo pipefail
        source_checkpoint="$(python3 -c '\''import os; from openpi.shared.download import maybe_download; print(maybe_download(os.environ["OPENPI_SPARK_SOURCE_CHECKPOINT"]))'\'')"
        python3 examples/convert_jax_model_to_pytorch.py \
            --checkpoint-dir "${source_checkpoint}" \
            --config-name pi05_libero \
            --output-path "${OPENPI_SPARK_OUTPUT_CHECKPOINT}" \
            --precision bfloat16
    '

test -s "${OPENPI_SPARK_DATA_DIR}/checkpoints/pi05_base_pytorch/model.safetensors"
sha256sum "${OPENPI_SPARK_DATA_DIR}/checkpoints/pi05_base_pytorch/model.safetensors" \
    | tee "${OPENPI_SPARK_DATA_DIR}/checkpoints/pi05_base_pytorch/model.safetensors.sha256"
