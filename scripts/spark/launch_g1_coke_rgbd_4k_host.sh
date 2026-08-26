#!/usr/bin/env bash
set -euo pipefail

image="${OPENPI_SPARK_IMAGE:-openpi-pi05-gb10:local}"
data_root="${OPENPI_SPARK_DATA_DIR:-/var/lib/openpi-spark}"
source_root="${OPENPI_SPARK_SOURCE_DIR:-${data_root}/openpi-source}"
train_repo_id="${OPENPI_G1_COKE_RGBD_DATASET_REPO_ID:-local/g1_coke_pickup_real_rgbd_left_rcoke_3_16_train_v1}"
experiment="${OPENPI_G1_COKE_RGBD_EXPERIMENT:-g1_coke_pickup_real_rgbd_left_arm14_pi05_4k_20260825}"
trainer_name="g1-coke-pickup-pi05-rgbd-arm14"
tensorboard_name="g1-coke-pickup-pi05-rgbd-tensorboard"

test -s "${data_root}/checkpoints/pi05_base_pytorch/model.safetensors"
test -s "${data_root}/lerobot/${train_repo_id}/meta/info.json"
test -f "${source_root}/scripts/spark/train_g1_coke_rgbd_4k.sh"
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
    --env "OPENPI_G1_COKE_RGBD_DATASET_REPO_ID=${train_repo_id}" \
    --env "OPENPI_G1_COKE_RGBD_EXPERIMENT=${experiment}" \
    --env "OPENPI_SPARK_DATA_DIR=/openpi_assets" \
    --env "HF_LEROBOT_HOME=/openpi_assets/lerobot" \
    --env "PYTHONPATH=/opt/openpi/src" \
    --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
    "${image}" \
    bash -lc "set -o pipefail; bash scripts/spark/train_g1_coke_rgbd_4k.sh 2>&1 | tee -a /openpi_assets/reports/${experiment}.log"

if ! docker container inspect "${tensorboard_name}" >/dev/null 2>&1; then
    docker run --detach \
        --name "${tensorboard_name}" \
        --restart unless-stopped \
        --network host \
        --volume "${data_root}:/openpi_assets:ro" \
        "${image}" \
        tensorboard \
            --logdir "/openpi_assets/training/pi05_spark_g1_coke_rgbd_arm14/${experiment}/tensorboard" \
            --bind_all \
            --port 6007
fi

docker inspect --format '{{.Name}} {{.State.Status}} {{.State.Pid}}' "${trainer_name}" "${tensorboard_name}"
