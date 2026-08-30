#!/usr/bin/env bash
set -euo pipefail
: "${H100_HOST:?}" "${H100_USER:?}" "${H100_CHECKPOINT_ROOT:?}" "${LOCAL_CHECKPOINT_ROOT:?}"
args=(-av --partial --append-verify)
[[ "${1:-}" == "--dry-run" ]] && args+=(--dry-run)
mkdir -p "$LOCAL_CHECKPOINT_ROOT"
rsync "${args[@]}" "$H100_USER@$H100_HOST:$H100_CHECKPOINT_ROOT/" "$LOCAL_CHECKPOINT_ROOT/"
