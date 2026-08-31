#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
: "${CHECKPOINT:?set CHECKPOINT to an Orbax checkpoint directory}"
: "${PROMPT:=perform the task}" "${PORT:=8000}" "${FAKE:=1}" "${POLICY_TIMEOUT_S:=1.0}"
uv run scripts/serve_policy.py --port "$PORT" policy:checkpoint --policy.config=pi05_ur5_twinwrist --policy.dir="$CHECKPOINT" &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT INT TERM
for _ in $(seq 1 60); do
  if curl --fail --silent --max-time 1 "http://127.0.0.1:${PORT}/healthz" >/dev/null; then
    break
  fi
  sleep 1
done
kill -0 "$server_pid"
curl --fail --silent --max-time 1 "http://127.0.0.1:${PORT}/healthz" >/dev/null
runtime_args=(
  examples/ur5_twinwrist/robot_runtime.py
  --mode infer
  --shadow
  --checkpoint "$CHECKPOINT"
  --prompt "$PROMPT"
  --host localhost
  --port "$PORT"
  --policy-timeout-s "$POLICY_TIMEOUT_S"
  --chunk-steps 1
)
if [[ "$FAKE" == "1" ]]; then
  runtime_args+=(--fake)
fi
uv run "${runtime_args[@]}"
