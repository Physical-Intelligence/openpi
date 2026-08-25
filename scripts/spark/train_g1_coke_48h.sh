#!/usr/bin/env bash
set -euo pipefail

: "${OPENPI_G1_COKE_DATASET_REPO_ID:?Set OPENPI_G1_COKE_DATASET_REPO_ID}"

experiment="${OPENPI_G1_COKE_EXPERIMENT:-g1_coke_pickup_pi05_48h_20260824}"
raw_dataset_slug="${OPENPI_G1_COKE_RAW_DATASET_SLUG:-g1_coke_pickup_phase1_5cm_upright_v2}"
session_seconds="${OPENPI_G1_COKE_SESSION_SECONDS:-172800}"
video_interval_seconds="${OPENPI_G1_COKE_VIDEO_INTERVAL_SECONDS:-21600}"
data_root="${OPENPI_SPARK_DATA_DIR:-/openpi_assets}"
run_dir="${data_root}/training/pi05_spark_g1_coke_pickup/${experiment}"
stats_path="${data_root}/assets/pi05_spark_g1_coke_pickup/${OPENPI_G1_COKE_DATASET_REPO_ID}/norm_stats.json"
episode="${OPENPI_G1_COKE_EVAL_EPISODE:-${data_root}/raw/${raw_dataset_slug}/episode_000000.npz}"
video_dir="${data_root}/videos/${experiment}"
session_path="${run_dir}/session_state.json"

mkdir -p "${run_dir}" "${video_dir}"
if [[ ! -s "${stats_path}" ]]; then
    python3 scripts/compute_norm_stats.py --config-name pi05_spark_g1_coke_pickup
fi
test -s "${stats_path}"
test -s "${episode}"

if [[ -s "${session_path}" ]]; then
    deadline_epoch="$(python3 -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["deadline_epoch"]))' "${session_path}")"
else
    started_epoch="$(date +%s)"
    deadline_epoch="$((started_epoch + session_seconds))"
    python3 -c 'import json,sys; json.dump({"state":"SESSION_RUNNING","started_epoch":int(sys.argv[2]),"deadline_epoch":int(sys.argv[3]),"video_interval_seconds":int(sys.argv[4])}, open(sys.argv[1],"w"), sort_keys=True)' "${session_path}" "${started_epoch}" "${deadline_epoch}" "${video_interval_seconds}"
fi

while [[ "$(date +%s)" -lt "${deadline_epoch}" ]]; do
    remaining="$((deadline_epoch - $(date +%s)))"
    chunk_seconds="${video_interval_seconds}"
    if [[ "${remaining}" -lt "${chunk_seconds}" ]]; then
        chunk_seconds="${remaining}"
    fi
    resume_args=()
    if find "${run_dir}" -mindepth 1 -maxdepth 1 -type d -name '[0-9]*' -print -quit | grep -q .; then
        resume_args+=(--resume)
    fi
    python3 scripts/train_pytorch.py pi05_spark_g1_coke_pickup \
        --exp-name "${experiment}" \
        --num-train-steps 10000000 \
        --max-train-seconds "${chunk_seconds}" \
        "${resume_args[@]}"

    checkpoint="$(find "${run_dir}" -mindepth 1 -maxdepth 1 -type d -name '[0-9]*' -print | sort -V | tail -1)"
    test -n "${checkpoint}"
    step="$(basename "${checkpoint}")"
    python3 scripts/spark/evaluate_g1_coke_checkpoint_video.py \
        --checkpoint-dir "${checkpoint}" \
        --episode "${episode}" \
        --output "${video_dir}/checkpoint_${step}.mp4" \
        --samples 24 \
        --denoise-steps 5
done

python3 -c 'import json,sys,time; p=json.load(open(sys.argv[1])); p.update(state="SESSION_DEADLINE_COMPLETE", completed_epoch=int(time.time())); json.dump(p, open(sys.argv[1],"w"), sort_keys=True)' "${session_path}"
