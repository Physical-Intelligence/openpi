SESSION=openpi-libero

# Check if session exists
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session $SESSION already exists. Attaching..."
    tmux attach-session -t "$SESSION"
    exit 0
fi

# Create new session
tmux new-session -d -s "$SESSION"

# Pane 0: Terminal 1
tmux send-keys -t $SESSION "
cd \"$(pwd)\"

conda activate libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
python qd_spatial.py
" C-m

# Pane 1: Terminal 2
tmux split-window -h -t "$SESSION"
tmux send-keys -t "$SESSION:0.1" "
cd \"$(pwd)\"

conda activate openpi

uv run scripts/serve_policy.py --env LIBERO
" C-m

# Attach
tmux attach-session -t "$SESSION"
