#!/usr/bin/env bash
# Drive convert_to_lerobot.py across every dataset in foldclo_datasets.txt.
#
# For each line "allenai/<basename>" in foldclo_datasets.txt:
#   - dst = "leokswang/<basename>_lerobot_format"
#   - Skips if a state file says it already completed (convert + push).
#   - Runs the converter (which auto-resumes within a single dataset's episodes).
#   - On success, marks done and moves on.
#
# Concurrency: this script runs datasets serially. If you launch two copies in
# parallel, the .state file acts as a coarse lock per dataset (skip if marked).
#
# Logs: each dataset's log goes to logs/<basename>.log
# State: state/<basename>.state contains "converted" or "pushed"
#
# Re-running is safe: completed datasets are skipped; partially-converted
# datasets pick up at the next episode.

set -u  # don't set -e: a single failure shouldn't kill the loop

cd "$(dirname "$0")"
LIST="${LIST:-foldclo_datasets.txt}"
LOG_DIR="${LOG_DIR:-logs}"
STATE_DIR="${STATE_DIR:-state}"
PUSH="${PUSH:-1}"  # set PUSH=0 to convert without pushing
MAX_DATASETS="${MAX_DATASETS:-1}"  # set to 0 for no limit
mkdir -p "$LOG_DIR" "$STATE_DIR"

push_flag=()
[[ "$PUSH" == "1" ]] && push_flag=(--push_to_hub)

_count=0
while IFS= read -r src; do
    [[ -z "$src" ]] && continue
    [[ "$MAX_DATASETS" -gt 0 && "$_count" -ge "$MAX_DATASETS" ]] && break
    ((_count++))
    base="${src##*/}"
    dst="leokswang/${base}_lerobot_format"
    state_file="$STATE_DIR/$base.state"
    log_file="$LOG_DIR/$base.log"

    if [[ -f "$state_file" ]] && grep -q "^done$" "$state_file"; then
        echo "[skip] $src already done"
        continue
    fi
    echo "[run ] $src -> $dst (log: $log_file)"
    if uv run convert_to_lerobot.py \
        --src_repo_id "$src" \
        --dst_repo_id "$dst" \
        "${push_flag[@]}" >>"$log_file" 2>&1
    then
        echo "done" >"$state_file"
        echo "[ok  ] $src"
    else
        rc=$?
        echo "fail rc=$rc $(date -Iseconds)" >>"$state_file"
        echo "[FAIL] $src (exit $rc) — see $log_file"
    fi
done <"$LIST"
