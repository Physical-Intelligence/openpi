#!/usr/bin/env bash
# Test 2: the dataset_exists gate used by the sbatch. Needs network (queries HF).
# Confirms a real repo returns exit 0 and a bogus one returns exit 1.
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 2
fail=0

REAL="${REAL_REPO:-leokswang/18012026-block-13_lerobot_format}"
BOGUS="${BOGUS_REPO:-leokswang/this-dataset-does-not-exist-zzz_lerobot_format}"

dataset_exists() {  # $1 = repo_id ; mirrors the function in the sbatch
    uv run --no-project --with huggingface_hub python - "$1" <<'PY'
import sys
from huggingface_hub import HfApi
sys.exit(0 if HfApi().repo_exists(sys.argv[1], repo_type="dataset") else 1)
PY
}

echo "== real repo should exist: $REAL =="
if dataset_exists "$REAL"; then echo "  OK (exit 0)"; else echo "  FAIL: real repo reported missing"; fail=1; fi

echo "== bogus repo should NOT exist: $BOGUS =="
if dataset_exists "$BOGUS"; then echo "  FAIL: bogus repo reported present"; fail=1; else echo "  OK (exit 1)"; fi

echo
[ "$fail" = 0 ] && echo "TEST 2 PASS" || echo "TEST 2 FAIL"
exit "$fail"
