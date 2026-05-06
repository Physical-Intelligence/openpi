#!/usr/bin/env bash
# eval.sh: Sync checkpoint and code to a remote server and launch pi0.5 inference/eval.
# Usage: ./experiments/eval.sh <server_name> [config_file] [checkpoint_step]
# Example: ./experiments/eval.sh aws-L40S-48gb experiments/configs/lipbalm.yaml 20000

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Parse arguments ---
SERVER="${1:?Usage: $0 <server_name> [config_file] [checkpoint_step]}"
CONFIG="${2:-$SCRIPT_DIR/configs/lipbalm.yaml}"
CHECKPOINT_STEP="${3:-}"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config file not found: $CONFIG"
    exit 1
fi

# --- Parse YAML config ---
parse_yaml() {
    local key="$1"
    local default="$2"
    if command -v yq &>/dev/null; then
        val=$(yq -r "$key // \"$default\"" "$CONFIG")
    else
        val=$(grep -oP "(?<=  ${key##*.}: ).*" "$CONFIG" 2>/dev/null | tr -d ' "' || echo "$default")
    fi
    echo "${val:-$default}"
}

REMOTE_DIR=$(parse_yaml '.server.remote_dir' '~/openpi')
REMOTE_DATA_DIR=$(parse_yaml '.server.remote_data_dir' '~/data/lipbalm')
EXP_NAME=$(parse_yaml '.experiment.name' 'lipbalm_pi05')
EVAL_PORT=$(parse_yaml '.eval.port' '8000')

# Use CLI arg, then YAML config, then default to "latest"
if [[ -z "$CHECKPOINT_STEP" ]]; then
    CHECKPOINT_STEP=$(parse_yaml '.eval.checkpoint_step' '')
fi

echo "=== Eval Job ==="
echo "Server:      $SERVER"
echo "Config:      $CONFIG"
echo "Experiment:  $EXP_NAME"
echo "Checkpoint:  ${CHECKPOINT_STEP:-latest}"
echo "Port:        $EVAL_PORT"
echo ""

# --- Verify SSH connectivity ---
echo "[1/4] Checking SSH connection to $SERVER..."
if ! ssh -o ConnectTimeout=10 "$SERVER" "echo ok" &>/dev/null; then
    echo "ERROR: Cannot connect to $SERVER. Check ~/.ssh/config"
    exit 1
fi
echo "  Connected."

# --- Sync code to remote ---
echo "[2/4] Syncing code to $SERVER:$REMOTE_DIR..."
rsync -az --delete \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'experiments/data/raw' \
    --exclude 'experiments/data/processed' \
    --exclude 'experiments/train/' \
    --exclude 'experiments/eval/' \
    --exclude '.venv' \
    --exclude 'wandb/' \
    "$PROJECT_DIR/" "$SERVER:$REMOTE_DIR/"
echo "  Code synced."

# --- Sync processed data to remote (needed for norm stats) ---
echo "[3/4] Syncing processed data to $SERVER:$REMOTE_DATA_DIR..."
ssh "$SERVER" "mkdir -p $REMOTE_DATA_DIR"
PROCESSED_DIR=$(parse_yaml '.data.processed_dir' 'experiments/data/processed')
rsync -az "$PROJECT_DIR/$PROCESSED_DIR/" "$SERVER:$REMOTE_DATA_DIR/"
echo "  Data synced."

# --- Launch eval on remote ---
echo "[4/4] Launching eval on $SERVER..."

EVAL_CMD="cd $REMOTE_DIR && HF_LEROBOT_HOME=$REMOTE_DATA_DIR python3 experiments/run_eval.py"
EVAL_CMD="$EVAL_CMD --config experiments/configs/lipbalm.yaml"
EVAL_CMD="$EVAL_CMD --port $EVAL_PORT"

if [[ -n "$CHECKPOINT_STEP" ]] && [[ "$CHECKPOINT_STEP" != "null" ]]; then
    EVAL_CMD="$EVAL_CMD --checkpoint_step $CHECKPOINT_STEP"
fi

echo "  Command: $EVAL_CMD"
echo ""

# Launch in a tmux session
ssh "$SERVER" "tmux new-session -d -s eval_${EXP_NAME} '$EVAL_CMD 2>&1 | tee $REMOTE_DIR/experiments/eval/eval.log'"

echo "=== Eval launched ==="
echo "Monitor: ssh $SERVER 'tmux attach -t eval_${EXP_NAME}'"
echo "Logs:    ssh $SERVER 'tail -f $REMOTE_DIR/experiments/eval/eval.log'"
echo "Server:  http://$SERVER:$EVAL_PORT"
