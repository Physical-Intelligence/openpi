#!/usr/bin/env bash
set -euo pipefail

image="${OPENPI_SPARK_IMAGE:-openpi-pi05-gb10:local}"
data_root="${OPENPI_SPARK_DATA_DIR:-/var/lib/openpi-spark}"
source_root="${OPENPI_SPARK_SOURCE_DIR:-${data_root}/openpi-source}"
raw_dataset_slug="${OPENPI_G1_COKE_RAW_DATASET_SLUG:-g1_coke_pickup_phase1_5cm_upright_v2}"
repo_id="${OPENPI_G1_COKE_DATASET_REPO_ID:-local/g1_coke_pickup_phase1_5cm_upright_v2}"
raw_dir="${data_root}/raw/${raw_dataset_slug}"

test -s "${raw_dir}/manifest.json"
test -f "${source_root}/scripts/spark/convert_g1_coke_sim_demos.py"
docker image inspect "${image}" >/dev/null

docker run --rm \
    --ipc=host \
    --volume "${data_root}:/openpi_assets" \
    --volume "${source_root}:/opt/openpi:ro" \
    --workdir /opt/openpi \
    --env "HF_LEROBOT_HOME=/openpi_assets/lerobot" \
    --env "PYTHONPATH=/opt/openpi/src" \
    "${image}" \
    python3 scripts/spark/convert_g1_coke_sim_demos.py \
        --raw-dir "/openpi_assets/raw/${raw_dataset_slug}" \
        --repo-id "${repo_id}" \
        --overwrite

test -s "${data_root}/lerobot/${repo_id}/meta/info.json"
