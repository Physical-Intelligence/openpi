#!/usr/bin/env bash
# train.sh: Sync data and code to a remote server and launch pi0.5 training.
# Usage: ./experiments/train.sh <server_name> [config_file]
# Example: ./experiments/train.sh aws-L40S-48gb experiments/configs/lipbalm.yaml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Parse arguments ---
SERVER="${1:?Usage: $0 <server_name> [config_file]}"
CONFIG="${2:-$SCRIPT_DIR/configs/lipbalm.yaml}"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config file not found: $CONFIG"
    exit 1
fi

# --- Parse YAML config (requires yq or falls back to grep) ---
parse_yaml() {
    local key="$1"
    local default="$2"
    if command -v yq &>/dev/null; then
        val=$(yq -r "$key // \"$default\"" "$CONFIG")
    else
        # Fallback: simple grep-based parsing for flat keys
        val=$(grep -oP "(?<=  ${key##*.}: ).*" "$CONFIG" 2>/dev/null | tr -d ' "' || echo "$default")
    fi
    echo "${val:-$default}"
}

REMOTE_DIR=$(parse_yaml '.server.remote_dir' '~/openpi')
REMOTE_DATA_DIR=$(parse_yaml '.server.remote_data_dir' '~/data/lipbalm')
GPU_COUNT=$(parse_yaml '.server.gpu_count' '1')
EXP_NAME=$(parse_yaml '.experiment.name' 'lipbalm_pi05')
PROCESSED_DIR=$(parse_yaml '.data.processed_dir' 'experiments/data/processed')

echo "=== Training Job ==="
echo "Server:      $SERVER"
echo "Config:      $CONFIG"
echo "Remote dir:  $REMOTE_DIR"
echo "Data dir:    $REMOTE_DATA_DIR"
echo "Experiment:  $EXP_NAME"
echo "GPUs:        $GPU_COUNT"
echo ""

# --- Verify SSH connectivity ---
echo "[1/5] Checking SSH connection to $SERVER..."
if ! ssh -o ConnectTimeout=10 "$SERVER" "echo ok" &>/dev/null; then
    echo "ERROR: Cannot connect to $SERVER. Check ~/.ssh/config"
    exit 1
fi
echo "  Connected."

# --- Verify processed data exists locally ---
echo "[2/5] Checking processed data..."
if [[ ! -d "$PROJECT_DIR/$PROCESSED_DIR" ]] || [[ -z "$(ls -A "$PROJECT_DIR/$PROCESSED_DIR" 2>/dev/null)" ]]; then
    echo "  No processed data found. Running processing script..."
    python3 "$SCRIPT_DIR/data/process_single_arm.py" \
        --raw-dir "$SCRIPT_DIR/data/raw" \
        --out-dir "$SCRIPT_DIR/data/processed" \
        --merge
    echo "  Processing complete."
else
    echo "  Processed data found in $PROCESSED_DIR"
fi

# --- Sync code to remote ---
echo "[3/5] Syncing code to $SERVER:$REMOTE_DIR..."
rsync -az --delete \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'checkpoints/' \
    --exclude 'experiments/data/raw' \
    --exclude 'experiments/data/processed' \
    --exclude 'experiments/train/' \
    --exclude 'experiments/eval/' \
    --exclude '.venv' \
    --exclude 'wandb/' \
    "$PROJECT_DIR/" "$SERVER:$REMOTE_DIR/"
echo "  Code synced."

# --- Sync processed data to remote ---
echo "[4/5] Syncing processed data to $SERVER:$REMOTE_DATA_DIR..."
ssh "$SERVER" "mkdir -p $REMOTE_DATA_DIR"
rsync -az "$PROJECT_DIR/$PROCESSED_DIR/" "$SERVER:$REMOTE_DATA_DIR/"
echo "  Data synced."

# --- Launch training on remote ---
echo "[5/5] Launching training on $SERVER..."

TRAIN_CMD="cd $REMOTE_DIR && HF_LEROBOT_HOME=$REMOTE_DATA_DIR"

if [[ "$GPU_COUNT" -gt 1 ]]; then
    TRAIN_CMD="$TRAIN_CMD torchrun --standalone --nnodes=1 --nproc_per_node=$GPU_COUNT"
else
    TRAIN_CMD="$TRAIN_CMD python3"
fi

TRAIN_CMD="$TRAIN_CMD experiments/run_train.py --config experiments/configs/lipbalm.yaml --exp_name $EXP_NAME"

echo "  Command: $TRAIN_CMD"
echo ""

# Launch in a tmux session so it survives SSH disconnect
ssh "$SERVER" "tmux new-session -d -s train_${EXP_NAME} '$TRAIN_CMD 2>&1 | tee $REMOTE_DIR/experiments/train/train.log'"

echo "=== Training launched ==="
echo "Monitor: ssh $SERVER 'tmux attach -t train_${EXP_NAME}'"
echo "Logs:    ssh $SERVER 'tail -f $REMOTE_DIR/experiments/train/train.log'"
