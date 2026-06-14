#!/bin/bash
#SBATCH --job-name=openpi-train-droid100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --time=24:00:00
#SBATCH --account=tsilver
#SBATCH --output=/scratch/gpfs/TSILVER/ss1824/openpi/slurm/logs/train_droid100_%j.out
#SBATCH --error=/scratch/gpfs/TSILVER/ss1824/openpi/slurm/logs/train_droid100_%j.err

export UV_CACHE_DIR=/scratch/gpfs/TSILVER/ss1824/.cache/uv
export HF_HOME=/scratch/gpfs/TSILVER/ss1824/.cache/huggingface
export HF_LEROBOT_HOME=/scratch/gpfs/TSILVER/ss1824/.cache/huggingface/lerobot
export OPENPI_DATA_HOME=/scratch/gpfs/TSILVER/ss1824/.cache/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
# # Compute nodes have no general internet -> keep wandb fully offline (logs sync later with `wandb sync`).
# export WANDB_MODE=offline

cd /scratch/gpfs/TSILVER/ss1824/openpi

module load proxy/default
echo "=== Fine-tuning pi05_droid100_lerobot (DROID-100 joint, fresh norm stats) on $(hostname) at $(date) ==="
uv run --frozen scripts/train.py pi05_droid100_lerobot --exp-name=droid100_joint_ft --overwrite
echo "=== Done at $(date) ==="
