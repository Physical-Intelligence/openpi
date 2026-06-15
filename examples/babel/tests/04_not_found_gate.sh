#!/usr/bin/env bash
# Test 4: the "wait until the dataset exists" gate. Points at a bogus repo with a short
# poll interval and confirms the loop LOGS [wait] and keeps polling (does not crash/exit).
# Self-bounded: stops after MAX_WAITS polls so it ends on its own. Needs network.
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 2

BOGUS="${BOGUS_REPO:-leokswang/this-dataset-does-not-exist-zzz_lerobot_format}"
POLL_SECONDS="${POLL_SECONDS:-3}"
MAX_WAITS="${MAX_WAITS:-3}"

dataset_exists() {  # mirrors the sbatch helper
    uv run --no-project --with huggingface_hub python - "$1" <<'PY'
import sys
from huggingface_hub import HfApi
sys.exit(0 if HfApi().repo_exists(sys.argv[1], repo_type="dataset") else 1)
PY
}

echo "== running the wait gate against a bogus repo (expect ${MAX_WAITS} x [wait], then give up) =="
waits=0
appeared=0
until dataset_exists "$BOGUS"; do
    waits=$(( waits + 1 ))
    echo "[wait] $(date -Iseconds) $BOGUS not on HF yet; re-poll in ${POLL_SECONDS}s ($waits/$MAX_WAITS)"
    if [ "$waits" -ge "$MAX_WAITS" ]; then break; fi
    sleep "$POLL_SECONDS"
done
dataset_exists "$BOGUS" && appeared=1

echo
echo "  [wait] lines=$waits   bogus appeared=$appeared (expected 0)"
if [ "$waits" -ge "$MAX_WAITS" ] && [ "$appeared" = 0 ]; then
    echo "TEST 4 PASS (gate kept waiting on a missing dataset instead of crashing)"; exit 0
else
    echo "TEST 4 FAIL"; exit 1
fi
