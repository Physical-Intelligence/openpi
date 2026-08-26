#!/usr/bin/env bash
set -euo pipefail

: "${OPENPI_G1_COKE_LEFT_ONLY_DATASET_REPO_ID:?Set OPENPI_G1_COKE_LEFT_ONLY_DATASET_REPO_ID}"
: "${OPENPI_G1_COKE_LEFT_ONLY_EVAL_REPO_ID:=local/g1_coke_pickup_real_rgbd_left_only_rcoke_3_16_eval_v1}"
export OPENPI_G1_COKE_LEFT_ONLY_EVAL_REPO_ID

config="pi05_spark_g1_coke_rgbd_left_arm7"
experiment="${OPENPI_G1_COKE_LEFT_ONLY_EXPERIMENT:-g1_coke_pickup_real_rgbd_left_arm7_upright_qualification_20260826}"
data_root="${OPENPI_SPARK_DATA_DIR:-/openpi_assets}"
run_dir="${data_root}/training/${config}/${experiment}"
stats_path="${data_root}/assets/${config}/${OPENPI_G1_COKE_LEFT_ONLY_DATASET_REPO_ID}/norm_stats.json"
train_info_path="${data_root}/lerobot/${OPENPI_G1_COKE_LEFT_ONLY_DATASET_REPO_ID}/meta/info.json"
train_contract_path="${data_root}/lerobot/${OPENPI_G1_COKE_LEFT_ONLY_DATASET_REPO_ID}/meta/wendy-rgbd-conversion.json"
eval_info_path="${data_root}/lerobot/${OPENPI_G1_COKE_LEFT_ONLY_EVAL_REPO_ID}/meta/info.json"
eval_contract_path="${data_root}/lerobot/${OPENPI_G1_COKE_LEFT_ONLY_EVAL_REPO_ID}/meta/wendy-rgbd-conversion.json"
run_state_path="${run_dir}/run_state.json"

mkdir -p "${run_dir}"
test -s "${train_info_path}"
test -s "${eval_info_path}"
python3 - "${train_contract_path}" "${eval_contract_path}" <<'PY'
import json
import sys

for path in sys.argv[1:]:
    payload = json.load(open(path, encoding="utf-8"))
    assert payload.get("contract") == "left-arm7", payload.get("contract")
    assert payload.get("state", {}).get("width") == 17
    assert payload.get("state", {}).get("right_arm_included") is False
    assert payload.get("action", {}).get("width") == 7
    assert payload.get("action", {}).get("right_arm_included") is False
PY

python3 - "${run_state_path}" "${OPENPI_G1_COKE_LEFT_ONLY_DATASET_REPO_ID}" "${OPENPI_G1_COKE_LEFT_ONLY_EVAL_REPO_ID}" <<'PY'
import json
import sys
import time

json.dump(
    {
        "state": "PREPARING",
        "started_epoch": int(time.time()),
        "config": "pi05_spark_g1_coke_rgbd_left_arm7",
        "contract": "left-arm7",
        "state_dim": 17,
        "action_dim": 7,
        "right_arm_inputs": False,
        "right_arm_actions": False,
        "dataset_repo_id": sys.argv[2],
        "checkpoint_eval_repo_id": sys.argv[3],
        "target_steps": 4000,
    },
    open(sys.argv[1], "w", encoding="utf-8"),
    sort_keys=True,
)
PY

if [[ ! -s "${stats_path}" ]]; then
    python3 scripts/compute_norm_stats.py --config-name "${config}"
fi
test -s "${stats_path}"

python3 scripts/train_pytorch.py "${config}" \
    --exp-name "${experiment}" \
    --num-train-steps 4000

python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p.get("state") == "TRAINING_STEPS_COMPLETE", p.get("state")' "${run_state_path}"
