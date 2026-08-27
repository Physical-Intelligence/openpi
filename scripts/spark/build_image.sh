#!/usr/bin/env bash
set -euo pipefail

OPENPI_SPARK_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_SPARK_REPO_DIR="$(cd "${OPENPI_SPARK_SCRIPT_DIR}/../.." && pwd)"
OPENPI_SPARK_IMAGE="${OPENPI_SPARK_IMAGE:-openpi-pi05-gb10:local}"
OPENPI_SPARK_BASE_IMAGE="${OPENPI_SPARK_BASE_IMAGE:-codex/mjlab-bench:20260809-egl}"

docker image inspect "${OPENPI_SPARK_BASE_IMAGE}" >/dev/null

docker build \
    --progress=plain \
    --build-arg "BASE_IMAGE=${OPENPI_SPARK_BASE_IMAGE}" \
    --file "${OPENPI_SPARK_REPO_DIR}/scripts/docker/spark_gb10.Dockerfile" \
    --tag "${OPENPI_SPARK_IMAGE}" \
    "${OPENPI_SPARK_REPO_DIR}"

docker image inspect "${OPENPI_SPARK_IMAGE}" \
    --format 'built={{.Created}} image_id={{.Id}} size={{.Size}}'
