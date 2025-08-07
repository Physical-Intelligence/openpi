## Installation
```bash
git clone --recurse-submodules https://github.com/szhaovas/openpi.git
cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
```

## Experiment
```bash
./run_experiment.sh
```
Archive heatmaps, checkpoints, and other metrics are logged to `test_logs`.

## Visualization
Needs to be run within the LIBERO venv. Make sure [dash](https://pypi.org/project/dash/) is installed.
```bash
source examples/libero/.venv/bin/activate
python -m pip install dash
```
Enter the scheduler `.pkl` checkpoint you wish to visualize near the end of `viz_spatial_attack.py`, and run 
```python
python viz_spatial_attack.py
```
This will display an interactive archive heatmap at `localhost:8050`. You can view it in the browser and click on a cell to save rollouts of that cell's solution to `interactive_vids`. If you are on ssh, you can also configure port forwarding to view and interact with heatmap on your own computer.