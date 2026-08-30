#!/usr/bin/env bash
set -euo pipefail
: "${H100_HOST:?}" "${H100_USER:?}" "${H100_DATA_ROOT:?}" "${LOCAL_DATA_ROOT:?}"
args=(-av --partial --append-verify)
[[ "${1:-}" == "--dry-run" ]] && args+=(--dry-run)
rsync "${args[@]}" "$LOCAL_DATA_ROOT/" "$H100_USER@$H100_HOST:$H100_DATA_ROOT/"
