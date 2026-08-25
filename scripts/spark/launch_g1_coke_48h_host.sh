#!/usr/bin/env bash
set -euo pipefail

image="${OPENPI_SPARK_IMAGE:-openpi-pi05-gb10:local}"
data_root="${OPENPI_SPARK_DATA_DIR:-/var/lib/openpi-spark}"
source_root="${OPENPI_SPARK_SOURCE_DIR:-${data_root}/openpi-source}"
raw_dataset_slug="${OPENPI_G1_COKE_RAW_DATASET_SLUG:-g1_coke_pickup_phase1_5cm_upright_v2}"
repo_id="${OPENPI_G1_COKE_DATASET_REPO_ID:-local/g1_coke_pickup_phase1_5cm_upright_v2}"
experiment="${OPENPI_G1_COKE_EXPERIMENT:-g1_coke_pickup_pi05_upright_48h_20260824}"
trainer_name="g1-coke-pickup-pi05-48h"
tensorboard_name="g1-coke-pickup-pi05-tensorboard"

test -s "${data_root}/checkpoints/pi05_base_pytorch/model.safetensors"
test -s "${data_root}/lerobot/${repo_id}/meta/info.json"
test -s "${data_root}/raw/${raw_dataset_slug}/episode_000000.npz"
test -f "${source_root}/scripts/spark/train_g1_coke_48h.sh"
docker image inspect "${image}" >/dev/null
mkdir -p "${data_root}/reports"

if docker container inspect "${trainer_name}" >/dev/null 2>&1; then
    echo "${trainer_name} already exists; refusing to replace an existing training identity" >&2
    exit 3
fi

docker run --detach \
    --name "${trainer_name}" \
    --restart on-failure:20 \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --volume "${data_root}:/openpi_assets" \
    --volume "${source_root}:/opt/openpi:ro" \
    --workdir /opt/openpi \
    --env "OPENPI_G1_COKE_DATASET_REPO_ID=${repo_id}" \
    --env "OPENPI_G1_COKE_EXPERIMENT=${experiment}" \
    --env "OPENPI_G1_COKE_RAW_DATASET_SLUG=${raw_dataset_slug}" \
    --env "OPENPI_G1_COKE_SESSION_SECONDS=172800" \
    --env "OPENPI_G1_COKE_VIDEO_INTERVAL_SECONDS=21600" \
    --env "OPENPI_SPARK_DATA_DIR=/openpi_assets" \
    --env "HF_LEROBOT_HOME=/openpi_assets/lerobot" \
    --env "PYTHONPATH=/opt/openpi/src" \
    --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
    "${image}" \
    bash -lc "set -o pipefail; bash scripts/spark/train_g1_coke_48h.sh 2>&1 | tee -a /openpi_assets/reports/${experiment}.log"

if ! docker container inspect "${tensorboard_name}" >/dev/null 2>&1; then
    docker run --detach \
        --name "${tensorboard_name}" \
        --restart unless-stopped \
        --network host \
        --volume "${data_root}:/openpi_assets:ro" \
        "${image}" \
        tensorboard \
            --logdir "/openpi_assets/training/pi05_spark_g1_coke_pickup/${experiment}/tensorboard" \
            --bind_all \
            --port 6006
fi

docker inspect --format '{{.Name}} {{.State.Status}} {{.State.Pid}}' "${trainer_name}" "${tensorboard_name}"
