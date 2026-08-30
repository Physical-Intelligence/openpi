#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
: "${CHECKPOINT:?set CHECKPOINT to an Orbax checkpoint directory}"
: "${PROMPT:=perform the task}" "${PORT:=8000}"
uv run scripts/serve_policy.py --port "$PORT" policy:checkpoint --policy.config=pi05_ur5_twinwrist --policy.dir="$CHECKPOINT" &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT INT TERM
for _ in $(seq 1 60); do (echo >/dev/tcp/127.0.0.1/"$PORT") 2>/dev/null && break; sleep 1; done
kill -0 "$server_pid"
uv run examples/ur5_twinwrist/robot_runtime.py --mode infer --shadow --fake --checkpoint "$CHECKPOINT" --prompt "$PROMPT"
