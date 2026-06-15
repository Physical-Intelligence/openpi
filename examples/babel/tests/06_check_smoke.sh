#!/usr/bin/env bash
# Test 6 (smoke result check): run AFTER the smoke job (test 5) finishes.
# Usage: bash 06_check_smoke.sh <jobid>
# Validates: both datasets trained, the 2nd resumed from the 1st, checkpoints exist and are
# capped at 3, and /scratch was cleaned. Run on a COMPUTE node or login node that can see
# /data/user_data/$USER (stat the full path first to trigger the AutoFS mount).
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 2
fail=0

JOBID="${1:-}"
EXP_NAME="${EXP_NAME:-yam_smoke}"
CONFIG_NAME="${CONFIG_NAME:-pi05_BiYAMMolmoAct2_loravlm}"
ckpt="/data/user_data/${USER}/openpi_checkpoints/${CONFIG_NAME}/${EXP_NAME}"
scratch="/scratch/${USER}/lerobot"
out=""
[ -n "$JOBID" ] && out="yam_smoke-${JOBID}.out"

ls -d "$ckpt" >/dev/null 2>&1 || true   # trigger AutoFS mount

echo "== checkpoint dir exists: $ckpt =="
if [ -d "$ckpt" ]; then echo "  OK"; else echo "  FAIL: no checkpoint dir"; fail=1; fi

echo "== checkpoint step dirs (numeric) =="
steps=$(find "$ckpt" -maxdepth 1 -mindepth 1 -type d -regextype posix-extended -regex '.*/[0-9]+' -printf '%f\n' 2>/dev/null | sort -n | tr '\n' ' ')
echo "  steps: ${steps:-<none>}"
nsteps=$(echo $steps | wc -w)
# Expect a checkpoint near the end of dataset 1 (~19) AND dataset 2 (~39); <=3 kept.
if [ "$nsteps" -ge 2 ] && [ "$nsteps" -le 3 ]; then
    echo "  OK ($nsteps checkpoints kept, within max_to_keep=3)"
else
    echo "  FAIL: expected 2-3 checkpoints, found $nsteps"; fail=1
fi
last=$(echo $steps | awk '{print $NF}')
if [ -n "$last" ] && [ "$last" -ge 39 ] 2>/dev/null; then
    echo "  OK final step $last reached (both datasets trained)"
else
    echo "  FAIL: final step '$last' < 39 (2nd dataset may not have trained)"; fail=1
fi

if [ -n "$out" ] && [ -f "$out" ]; then
    echo "== log shows --overwrite then --resume (continual) =="
    grep -q 'overwrite, num_train_steps=20' "$out"  && echo "  OK dataset1 --overwrite @20" || { echo "  FAIL: no overwrite@20 line"; fail=1; }
    grep -q 'resume, num_train_steps=40' "$out"     && echo "  OK dataset2 --resume @40"    || { echo "  FAIL: no resume@40 line"; fail=1; }
    echo "== log shows scratch cleanup =="
    grep -q '\[clean\] removed' "$out" && echo "  OK clean line present" || { echo "  FAIL: no [clean] line"; fail=1; }
else
    echo "== (skip log checks: pass <jobid> to inspect yam_smoke-<jobid>.out) =="
fi

echo "== /scratch was cleaned (no smoke datasets left) =="
left=$(find "$scratch/leokswang" -maxdepth 1 -mindepth 1 -type d -name '18012026-block-1*_lerobot_format' 2>/dev/null | wc -l)
if [ "$left" = 0 ]; then echo "  OK scratch clean"; else echo "  WARN: $left smoke dataset(s) still on scratch"; fi

echo
[ "$fail" = 0 ] && echo "SMOKE CHECK PASS" || echo "SMOKE CHECK FAIL"
exit "$fail"
