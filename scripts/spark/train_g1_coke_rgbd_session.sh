#!/usr/bin/env bash
set -euo pipefail

: "${OPENPI_G1_COKE_RGBD_DATASET_REPO_ID:?Set OPENPI_G1_COKE_RGBD_DATASET_REPO_ID}"
: "${OPENPI_G1_COKE_RGBD_EVAL_REPO_ID:?Set OPENPI_G1_COKE_RGBD_EVAL_REPO_ID}"

experiment="${OPENPI_G1_COKE_RGBD_EXPERIMENT:-g1_coke_rgbd_arm14_augmented_7h_20260825}"
session_seconds="${OPENPI_G1_COKE_RGBD_SESSION_SECONDS:-25200}"
evaluation_interval_seconds="${OPENPI_G1_COKE_RGBD_EVAL_INTERVAL_SECONDS:-7200}"
data_root="${OPENPI_SPARK_DATA_DIR:-/openpi_assets}"
run_dir="${data_root}/training/pi05_spark_g1_coke_rgbd_arm14/${experiment}"
report_dir="${data_root}/reports/${experiment}"
video_dir="${data_root}/videos/${experiment}"
session_path="${run_dir}/session_state.json"
qualification_path="${report_dir}/qualification.json"

if [[ "${session_seconds}" -lt 10800 ]]; then
    echo "RGB-D session must allow at least three hours for qualification" >&2
    exit 2
fi
if [[ "${evaluation_interval_seconds}" -lt 900 ]]; then
    echo "RGB-D evaluation interval must be at least 15 minutes" >&2
    exit 2
fi
mkdir -p "${run_dir}" "${report_dir}" "${video_dir}"

recovery_mode=false
if python3 scripts/spark/recover_g1_coke_rgbd_qualification.py \
    --run-dir "${run_dir}" \
    --session-state "${session_path}" \
    --qualification-report "${qualification_path}" \
    --check >/dev/null 2>&1; then
    recovery_mode=true
    read -r started_epoch deadline_epoch < <(
        python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); print(p["started_epoch"], p["deadline_epoch"])' "${session_path}"
    )
    python3 scripts/spark/recover_g1_coke_rgbd_qualification.py \
        --run-dir "${run_dir}" \
        --session-state "${session_path}" \
        --qualification-report "${qualification_path}" \
        --apply
else
    started_epoch="$(date +%s)"
    deadline_epoch="$((started_epoch + session_seconds))"
    python3 -c '
import json,sys
json.dump({"state":"QUALIFICATION_RUNNING","started_epoch":int(sys.argv[2]),"deadline_epoch":int(sys.argv[3]),"dataset_repo_id":sys.argv[4],"checkpoint_eval_repo_id":sys.argv[5]},open(sys.argv[1],"w"),sort_keys=True)
' "${session_path}" "${started_epoch}" "${deadline_epoch}" "${OPENPI_G1_COKE_RGBD_DATASET_REPO_ID}" "${OPENPI_G1_COKE_RGBD_EVAL_REPO_ID}"
fi

on_error() {
    local exit_code="$?"
    trap - ERR
    python3 -c '
import json,sys,time
path=sys.argv[1]; payload=json.load(open(path)); payload.update(state="SESSION_FAILED",failed_epoch=int(time.time()),exit_code=int(sys.argv[2])); json.dump(payload,open(path,"w"),sort_keys=True)
' "${session_path}" "${exit_code}" || true
    exec tail -f /dev/null
}
trap on_error ERR

best_checkpoint() {
    python3 -c '
import json,pathlib,sys
root=pathlib.Path(sys.argv[1])
manifest=json.loads((root/"reward_checkpoints.json").read_text())
records=[record for record in manifest.get("records",[]) if (root/str(record["step"])/"model.safetensors").is_file()]
if not records: raise SystemExit("no complete reward checkpoint")
print(root/str(max(records,key=lambda record:float(record["reward"]))["step"]))
' "${run_dir}"
}

evaluate_checkpoint() {
    local checkpoint="$1"
    local label="$2"
    python3 scripts/spark/evaluate_g1_coke_rgbd_checkpoint.py \
        --checkpoint-dir "${checkpoint}" \
        --repo-id "${OPENPI_G1_COKE_RGBD_EVAL_REPO_ID}" \
        --output-json "${report_dir}/${label}.json" \
        --output-video "${video_dir}/${label}.mp4" \
        --samples 32 \
        --denoise-steps 5
}

if [[ "${recovery_mode}" == false ]]; then
    bash scripts/spark/train_g1_coke_rgbd_4k.sh
fi
checkpoint="$(best_checkpoint)"
step="$(basename "${checkpoint}")"
evaluate_checkpoint "${checkpoint}" "qualification_checkpoint_${step}"
python3 scripts/spark/qualify_g1_coke_rgbd_run.py \
    --run-dir "${run_dir}" \
    --evaluation-report "${report_dir}/qualification_checkpoint_${step}.json" \
    --output "${qualification_path}"

python3 -c '
import json,sys,time
path=sys.argv[1]; payload=json.load(open(path)); payload.update(state="QUALIFIED_CONTINUATION_RUNNING",qualified_epoch=int(time.time()),qualification_report=sys.argv[2]); json.dump(payload,open(path,"w"),sort_keys=True)
' "${session_path}" "${qualification_path}"

while [[ "$(date +%s)" -lt "${deadline_epoch}" ]]; do
    remaining="$((deadline_epoch - $(date +%s)))"
    chunk_seconds="${evaluation_interval_seconds}"
    if [[ "${remaining}" -lt "${chunk_seconds}" ]]; then
        chunk_seconds="${remaining}"
    fi
    python3 scripts/train_pytorch.py pi05_spark_g1_coke_rgbd_arm14 \
        --exp-name "${experiment}" \
        --num-train-steps 10000000 \
        --max-train-seconds "${chunk_seconds}" \
        --resume
    checkpoint="$(best_checkpoint)"
    step="$(basename "${checkpoint}")"
    label="checkpoint_${step}_$(date +%s)"
    evaluate_checkpoint "${checkpoint}" "${label}"
    python3 -c '
import json,sys,time
path=sys.argv[1]; payload=json.load(open(path)); payload.update(state="QUALIFIED_CONTINUATION_RUNNING",last_evaluation_epoch=int(time.time()),last_evaluation_report=sys.argv[2],best_checkpoint=sys.argv[3]); json.dump(payload,open(path,"w"),sort_keys=True)
' "${session_path}" "${report_dir}/${label}.json" "${checkpoint}"
done

checkpoint="$(best_checkpoint)"
python3 -c '
import json,sys,time
path=sys.argv[1]; payload=json.load(open(path)); payload.update(state="SESSION_COMPLETE",completed_epoch=int(time.time()),best_checkpoint=sys.argv[2]); json.dump(payload,open(path,"w"),sort_keys=True)
' "${session_path}" "${checkpoint}"

# Wendy treats an exited one-shot service as unhealthy. Keep the completed
# session inspectable without restarting training or changing the checkpoint.
exec tail -f /dev/null
