#!/usr/bin/env bash
set -euo pipefail

: "${OPENPI_G1_COKE_RGBD_DATASET_REPO_ID:?Set OPENPI_G1_COKE_RGBD_DATASET_REPO_ID}"
: "${OPENPI_G1_COKE_RGBD_EVAL_REPO_ID:=local/g1_coke_pickup_real_rgbd_left_rcoke_3_16_eval_v1}"
export OPENPI_G1_COKE_RGBD_EVAL_REPO_ID

experiment="${OPENPI_G1_COKE_RGBD_EXPERIMENT:-g1_coke_pickup_real_rgbd_left_arm14_pi05_4k_20260825}"
data_root="${OPENPI_SPARK_DATA_DIR:-/openpi_assets}"
run_dir="${data_root}/training/pi05_spark_g1_coke_rgbd_arm14/${experiment}"
stats_path="${data_root}/assets/pi05_spark_g1_coke_rgbd_arm14/${OPENPI_G1_COKE_RGBD_DATASET_REPO_ID}/norm_stats.json"
eval_info_path="${data_root}/lerobot/${OPENPI_G1_COKE_RGBD_EVAL_REPO_ID}/meta/info.json"
run_state_path="${run_dir}/run_state.json"

mkdir -p "${run_dir}"
test -s "${eval_info_path}"
python3 -c 'import json,sys,time; json.dump({"state":"PREPARING","started_epoch":int(time.time()),"config":"pi05_spark_g1_coke_rgbd_arm14","dataset_repo_id":sys.argv[2],"checkpoint_eval_repo_id":sys.argv[3],"target_steps":4000}, open(sys.argv[1],"w"), sort_keys=True)' "${run_state_path}" "${OPENPI_G1_COKE_RGBD_DATASET_REPO_ID}" "${OPENPI_G1_COKE_RGBD_EVAL_REPO_ID}"

if [[ ! -s "${stats_path}" ]]; then
    python3 scripts/compute_norm_stats.py --config-name pi05_spark_g1_coke_rgbd_arm14
fi
test -s "${stats_path}"

python3 -c 'import json,sys,time; p=json.load(open(sys.argv[1])); p.update(state="TRAINING_RUNNING",training_started_epoch=int(time.time())); json.dump(p, open(sys.argv[1],"w"), sort_keys=True)' "${run_state_path}"

python3 scripts/train_pytorch.py pi05_spark_g1_coke_rgbd_arm14 \
    --exp-name "${experiment}" \
    --num-train-steps 4000

python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p.get("state") == "TRAINING_STEPS_COMPLETE", p.get("state")' "${run_state_path}"
