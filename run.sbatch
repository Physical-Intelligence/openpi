#!/bin/bash
#SBATCH --job-name=Openpi_Finetune
#SBATCH --output=results/run-%J.out
#SBATCH --error=results/run-%J.err
#SBATCH --cpus-per-task=16
#SBATCH --time=26:00:00
#SBATCH --account=bfbo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=220G
#SBATCH --nodes=1

set -euo pipefail

# module load cuda/12.6.1
source ~/.bashrc
cd /work/nvme/bfbo/xzhang42/openpi
PY=/work/nvme/bfbo/xzhang42/openpi/.venv/bin/python
# export CUDA_VISIBLE_DEVICES=0
export XDG_CACHE_HOME=/work/nvme/bfbo/xzhang42/.cache
export OPENPI_DATA_HOME=/work/nvme/bfbo/xzhang42/openpi/.cache

# EXTREMELY IMPORTANT: completely remove 12.3 SDK paths
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v '/opt/nvidia/hpc_sdk/Linux_aarch64/24.3' | paste -sd:)

export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0"

unset JAX_PLUGINS JAX_PLATFORM_NAME JAX_PLATFORMS
unset NCCL_NET NCCL_NET_PLUGIN NCCL_NET_OFI_PROVIDER NCCL_SOCKET_IFNAME NCCL_IB_DISABLE NCCL_CROSS_NIC NCCL_PLUGIN_P2P

"$PY" - <<'PY'
import jax, jaxlib, jax.numpy as jnp
print("jax", jax.__version__)
print("jaxlib", jaxlib.__version__)
print("jaxlib file", jaxlib.__file__)
print("devices", jax.devices())
x = jnp.ones((2048, 2048), dtype=jnp.float16)
y = x @ x
print(y.block_until_ready())
PY


# python scripts/compute_norm_stats.py \
#   --config-name pi05_tea_use_steel_spoon \
#   --repo-root /work/nvme/bfbo/xzhang42/datasets/qrafty-ai/tea_use_steel_spoon

# python scripts/train.py \
# pi05_tea_use_steel_spoon \
# --data.repo_id /work/nvme/bfbo/xzhang42/datasets/qrafty-ai/tea_use_steel_spoon \
# --exp-name action50 \
# --num-train-steps 70001\

# python scripts/compute_norm_stats.py \
#   --config-name pi05_tea_use_spoon \
#   --repo-root /work/nvme/bfbo/xzhang42/datasets/qrafty-ai/tea_use_spoon_openpi

# python scripts/train.py \
# pi05_tea_use_spoon \
# --data.repo_id /work/nvme/bfbo/xzhang42/datasets/qrafty-ai/tea_use_spoon_openpi \
# --exp-name action50 \

# python scripts/train.py \
#   pi05_tea_use_spoon \
#   --data.repo_id /work/nvme/bfbo/xzhang42/datasets/qrafty-ai/tea_use_spoon_openpi \
#   --exp-name action50 \
#   --resume \
#   --num-train-steps 70001

# python scripts/compute_norm_stats.py \
#   --config-name pi05_tea_pick_cup\
#   --repo-root /work/nvme/bfbo/xzhang42/datasets/qrafty-ai/tea_pick_cup

# python scripts/train.py \
# pi05_tea_pick_cup \
# --data.repo_id /work/nvme/bfbo/xzhang42/datasets/qrafty-ai/tea_pick_cup \
# --exp-name action50 \

# python scripts/compute_norm_stats.py \
#   --config-name pi05_franka_object\
#   --repo-root /work/nvme/bfbo/xzhang42/datasets/pink_blue_blocks_base_task

"$PY" scripts/train.py \
  pi05_franka_object \
  --data.repo_id /work/nvme/bfbo/xzhang42/datasets/pink_blue_blocks_base_task \
  --exp-name delta-full0 \
  --num-train-steps 30001
