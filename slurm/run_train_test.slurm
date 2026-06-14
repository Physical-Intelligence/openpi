#!/bin/bash
#SBATCH --job-name=openpi-train-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --constraint=a100
#SBATCH --time=00:30:00
#SBATCH --account=tsilver
#SBATCH --output=/scratch/gpfs/TSILVER/ss1824/openpi/slurm/logs/train_test_%j.out
#SBATCH --error=/scratch/gpfs/TSILVER/ss1824/openpi/slurm/logs/train_test_%j.err

export UV_CACHE_DIR=/scratch/gpfs/TSILVER/ss1824/.cache/uv
export HF_HOME=/scratch/gpfs/TSILVER/ss1824/.cache/huggingface
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

cd /scratch/gpfs/TSILVER/ss1824/openpi

echo "=== Running train unit test on $(hostname) at $(date) ==="
uv run pytest scripts/train_test.py -v
echo "=== Done at $(date) ==="
