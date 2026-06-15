#!/usr/bin/env bash
# Drive convert_to_lerobot.py across the datasets in examples/babel/raw_yam_data.txt.
#
# Mirrors convert_all_foldclo.sh. For each line "allenai/<basename>":
#   - dst = "leokswang/<basename>_lerobot_format"
#   - Skips if a state file says it already completed (convert + push).
#   - Runs the converter (which auto-resumes within a single dataset's episodes).
#   - On success, marks done and moves on.
#
# Processes datasets serially, top-to-bottom. MAX_DATASETS caps how many lines
# are *considered* this run (already-done lines still count toward the cap, so
# re-running after a partial run continues past the first 100 correctly only if
# you bump MAX_DATASETS; for "first 100" semantics leave it at 100).
#
# Logs: logs/<basename>.log   State: state/<basename>.state ("done" or "fail ...")
# Re-running is safe: completed datasets are skipped; partially-converted
# datasets pick up at the next episode.

set -u  # don't set -e: a single failure shouldn't kill the loop

cd "$(dirname "$0")"
LIST="${LIST:-examples/babel/raw_yam_data.txt}"
LOG_DIR="${LOG_DIR:-logs}"
STATE_DIR="${STATE_DIR:-state}"
PUSH="${PUSH:-1}"                  # set PUSH=0 to convert without pushing
MAX_DATASETS="${MAX_DATASETS:-100}"  # set to 0 for no limit
# After a successful push, delete the local converted copy to keep disk flat
# (the Hub copy is the source of truth). Only ever cleans when PUSH=1, so we
# never delete data that wasn't uploaded. Set CLEANUP=0 to keep local copies.
CLEANUP="${CLEANUP:-1}"
LEROBOT_HOME="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}"
mkdir -p "$LOG_DIR" "$STATE_DIR"

push_flag=()
[[ "$PUSH" == "1" ]] && push_flag=(--push_to_hub)

_count=0
_ok=0
_fail=0
while IFS= read -r src; do
    src="${src%%[[:space:]]}"
    [[ -z "$src" ]] && continue
    [[ "$MAX_DATASETS" -gt 0 && "$_count" -ge "$MAX_DATASETS" ]] && break
    ((_count++))
    base="${src##*/}"
    dst="leokswang/${base}_lerobot_format"
    state_file="$STATE_DIR/$base.state"
    log_file="$LOG_DIR/$base.log"

    if [[ -f "$state_file" ]] && grep -q "^done$" "$state_file"; then
        echo "[skip] ($_count/$MAX_DATASETS) $src already done"
        ((_ok++))
        continue
    fi
    echo "[run ] ($_count/$MAX_DATASETS) $src -> $dst (log: $log_file)"
    if uv run convert_to_lerobot.py \
        --src_repo_id "$src" \
        --dst_repo_id "$dst" \
        "${push_flag[@]}" >>"$log_file" 2>&1
    then
        echo "done" >"$state_file"
        echo "[ok  ] ($_count/$MAX_DATASETS) $src"
        ((_ok++))
        if [[ "$CLEANUP" == "1" && "$PUSH" == "1" ]]; then
            out_dir="$LEROBOT_HOME/$dst"
            if [[ -d "$out_dir" ]]; then
                echo "[clean] rm $out_dir" | tee -a "$log_file"
                rm -rf "$out_dir"
            fi
        fi
    else
        rc=$?
        echo "fail rc=$rc $(date -Iseconds)" >>"$state_file"
        echo "[FAIL] ($_count/$MAX_DATASETS) $src (exit $rc) — see $log_file"
        ((_fail++))
    fi
done <"$LIST"

echo "[summary] considered=$_count ok=$_ok fail=$_fail"
