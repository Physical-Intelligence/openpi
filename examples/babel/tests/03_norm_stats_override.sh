#!/usr/bin/env bash
# Test 3: compute_norm_stats.py --repo-id override. Downloads one dataset to /scratch and
# writes per-dataset norm stats. Needs network + CPU (no GPU). Downloads a full dataset, so
# run it where /scratch is available; --max-frames keeps the compute short.
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 2
fail=0

CONFIG_NAME="${CONFIG_NAME:-pi05_BiYAMMolmoAct2_loravlm}"
REPO="${REPO:-leokswang/18012026-block-13_lerobot_format}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/scratch/${USER}/lerobot}"
export HF_HOME="${HF_HOME:-/scratch/${USER}/hf}"
export HF_HUB_DISABLE_XET=1
mkdir -p "$HF_LEROBOT_HOME" "$HF_HOME"

out_dir="assets/${CONFIG_NAME}/${REPO}"
echo "== computing norm stats for $REPO (max 200 frames) =="
echo "   download -> $HF_LEROBOT_HOME/$REPO"
echo "   stats    -> $out_dir"
rm -f "${out_dir}/norm_stats.json"

if uv run scripts/compute_norm_stats.py --config-name "$CONFIG_NAME" --repo-id "$REPO" --max-frames 200; then
    echo "  norm stats run finished"
else
    echo "  FAIL: compute_norm_stats.py errored"; fail=1
fi

echo "== norm_stats.json written under assets/<config>/<repo>/ =="
if [ -f "${out_dir}/norm_stats.json" ]; then echo "  OK  ${out_dir}/norm_stats.json"; else echo "  FAIL: norm_stats.json not found"; fail=1; fi

echo "== dataset landed on /scratch (HF_LEROBOT_HOME) =="
if [ -d "${HF_LEROBOT_HOME}/${REPO}" ]; then echo "  OK  ${HF_LEROBOT_HOME}/${REPO}"; else echo "  FAIL: dataset not under HF_LEROBOT_HOME"; fail=1; fi

echo
[ "$fail" = 0 ] && echo "TEST 3 PASS" || echo "TEST 3 FAIL"
echo "(tip: 'rm -rf ${HF_LEROBOT_HOME:?}/${REPO}' to reclaim scratch after this test)"
exit "$fail"
