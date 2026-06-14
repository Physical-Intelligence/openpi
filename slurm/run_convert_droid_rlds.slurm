#!/bin/bash
#SBATCH --job-name=droid-rlds2lerobot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --account=tsilver
#SBATCH --output=/scratch/gpfs/TSILVER/ss1824/openpi/slurm/logs/convert_droid_%j.out
#SBATCH --error=/scratch/gpfs/TSILVER/ss1824/openpi/slurm/logs/convert_droid_%j.err

export UV_CACHE_DIR=/scratch/gpfs/TSILVER/ss1824/.cache/uv
export HF_HOME=/scratch/gpfs/TSILVER/ss1824/.cache/huggingface
export HF_LEROBOT_HOME=/scratch/gpfs/TSILVER/ss1824/.cache/huggingface/lerobot
export OPENPI_DATA_HOME=/scratch/gpfs/TSILVER/ss1824/.cache/openpi

cd /scratch/gpfs/TSILVER/ss1824/openpi

module load proxy/default
echo "=== Converting r2d2_faceblur RLDS -> LeRobot (joint space) on $(hostname) at $(date) ==="
uv run --group rlds --frozen examples/droid/convert_droid_rlds_to_lerobot.py \
    --rlds-data-dir /scratch/gpfs/TSILVER/ss1824/.cache/openpi/droid_rlds \
    --builder-name r2d2_faceblur --version 1.0.0 \
    --repo-id samratsahoo/droid_100_joint
echo "=== Done at $(date) ==="
