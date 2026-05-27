cd /home/jiajun/SA-WM/vlas/openpi
source .venv/bin/activate

export UV_CACHE_DIR=/tmp/uv-cache
export OPENPI_DATA_HOME=/data/shared/models/openpi
export HF_HOME=/data/yuhang/huggingface
export HF_DATASETS_CACHE=/tmp/hf-datasets-cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false

CUDA_VISIBLE_DEVICES=4,5 uv run scripts/serve_policy.py \
    --port 8000   policy:checkpoint \
    --policy.config pi05_libero  \
    --policy.dir /data/shared/models/openpi/openpi-assets/checkpoints/pi05_libero