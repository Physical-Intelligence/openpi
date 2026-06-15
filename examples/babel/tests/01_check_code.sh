#!/usr/bin/env bash
# Test 1: code + config wiring. Login-node safe (no GPU, no network, no downloads).
# Verifies the openpi edits load and expose the new CLI flags.
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 2
CONFIG_NAME="${CONFIG_NAME:-pi05_BiYAMMolmoAct2_loravlm}"
fail=0

echo "== py_compile edited files =="
uv run python -m py_compile \
    scripts/compute_norm_stats.py scripts/train.py \
    src/openpi/training/checkpoints.py src/openpi/training/config.py \
    && echo "  OK" || { echo "  FAIL: syntax error"; fail=1; }

echo "== config exposes max_to_keep=3 / keep_period=None / num_train_steps=2000 =="
uv run python -c "
import openpi.training.config as c
cfg = c.get_config('${CONFIG_NAME}')
assert cfg.max_to_keep == 3, f'max_to_keep={cfg.max_to_keep}'
assert cfg.keep_period is None, f'keep_period={cfg.keep_period}'
assert cfg.num_train_steps == 2000, f'num_train_steps={cfg.num_train_steps}'
print('  OK  max_to_keep=3 keep_period=None num_train_steps=2000')
" || { echo "  FAIL: config fields wrong"; fail=1; }

echo "== compute_norm_stats.py exposes --repo-id =="
uv run scripts/compute_norm_stats.py --help 2>&1 | grep -q -- '--repo-id' \
    && echo "  OK" || { echo "  FAIL: --repo-id missing"; fail=1; }

echo "== train.py exposes --data.repo-id / --num-train-steps / --resume / --overwrite =="
helptext="$(uv run scripts/train.py "${CONFIG_NAME}" --help 2>&1)"
for flag in -- '--data.repo-id' '--num-train-steps' '--resume' '--overwrite'; do
    [ "$flag" = -- ] && continue
    echo "$helptext" | grep -q -- "$flag" \
        && echo "  OK  $flag" || { echo "  FAIL: $flag missing"; fail=1; }
done

echo
[ "$fail" = 0 ] && echo "TEST 1 PASS" || echo "TEST 1 FAIL"
exit "$fail"
