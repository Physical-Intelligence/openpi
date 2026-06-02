#!/bin/bash
#SBATCH --job-name=pi05_so101_rings
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=0G
#SBATCH --exclusive
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --requeue

# pi05_so101_stacking_rings — SO101 stacking rings (6D joint-space, delta actions).
#
# Cluster: Isambard u6kr. Submit from the worktree so slurm logs stay isolated:
#   cd /home/u6kr/pravsels.u6kr/openpi_so101_stacking_rings
#   sbatch slurm/train_so101_stacking_rings_slurm.sh
#
# Branch: task/stack_rings
#
# Before first sbatch:
#   - openpi_arm64.sif at ${data_dir}/container/
#   - pi05_base weights at ${data_dir}/weights/pi05_base/params
#   - HF / W&B tokens in ${scratch_dir}/.secrets/

set -e

module purge
module load brics/apptainer-multi-node

# --- Infrastructure (u6kr) ---
home_dir="/home/u6kr/pravsels.u6kr"
scratch_dir="/scratch/u6kr/pravsels.u6kr"
repo_dir="${home_dir}/openpi_so101_stacking_rings"
data_dir="${scratch_dir}/openpi"
container="${data_dir}/container/openpi_arm64.sif"
HF_CACHE="${scratch_dir}/huggingface_cache"
HF_TOKEN_FILE="${scratch_dir}/.secrets/.hf_token"
WANDB_DIR="${data_dir}"
WANDB_CACHE_DIR="${scratch_dir}/.cache/wandb"
WANDB_CONFIG_DIR="${scratch_dir}/.config/wandb"
XDG_CACHE_HOME="${scratch_dir}/.cache"
XDG_CONFIG_HOME="${scratch_dir}/.config"

# --- Experiment (matches GCloud replication run) ---
CONFIG_NAME="pi05_so101_stacking_rings"
EXP_NAME="so101_stacking_rings"

CHECKPOINT_DIR="${data_dir}/checkpoints/${CONFIG_NAME}/${EXP_NAME}"
ASSETS_DIR="${data_dir}/assets/${CONFIG_NAME}/${EXP_NAME}"

if [ -z "${ASSETS_DIR}" ]; then
    echo "ERROR: ASSETS_DIR is empty; refusing to run."
    exit 1
fi
if [ ! -f "${container}" ]; then
    echo "ERROR: container not found: ${container}"
    exit 1
fi
if [ ! -f "${HF_TOKEN_FILE}" ]; then
    echo "ERROR: HF token not found: ${HF_TOKEN_FILE}"
    exit 1
fi

mkdir -p "${HF_CACHE}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" \
    "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${data_dir}/checkpoints" \
    "${data_dir}/assets" "${data_dir}/weights" "${data_dir}/.venv" "${ASSETS_DIR}"

start_time="$(date -Is --utc)"
echo "===================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Config: ${CONFIG_NAME}"
echo "Experiment: ${EXP_NAME}"
echo "Repo: ${repo_dir}"
echo "Started (UTC): ${start_time}"
echo "===================================="

COMPUTE_NORM_STATS_CMD="uv run scripts/compute_norm_stats_per_timestep.py --config-name=${CONFIG_NAME} --assets-dir=${ASSETS_DIR}"
NORM_STATS_PATH="${ASSETS_DIR}/norm_stats.json"
PER_TIMESTEP_STATS_PATH="${ASSETS_DIR}/norm_stats_actions_per_timestep.json"

TRAIN_FLAGS="--exp-name=${EXP_NAME} --assets-dir=${ASSETS_DIR}"
if [ -d "${CHECKPOINT_DIR}" ] && [ -n "$(find "${CHECKPOINT_DIR}" -mindepth 1 -maxdepth 1 -type d ! -name assets 2>/dev/null | head -1)" ]; then
    TRAIN_FLAGS="${TRAIN_FLAGS} --resume"
else
    TRAIN_FLAGS="${TRAIN_FLAGS} --overwrite"
fi
TRAIN_CMD="uv run scripts/train.py ${CONFIG_NAME} ${TRAIN_FLAGS}"

EXPORT_VARS="export PYTHONUNBUFFERED=1"
EXPORT_VARS="${EXPORT_VARS} && export WANDB_MODE=offline"
EXPORT_VARS="${EXPORT_VARS} && export WANDB_DIR=${WANDB_DIR}"
EXPORT_VARS="${EXPORT_VARS} && export WANDB_CACHE_DIR=${WANDB_CACHE_DIR}"
EXPORT_VARS="${EXPORT_VARS} && export WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR}"
EXPORT_VARS="${EXPORT_VARS} && export XDG_CACHE_HOME=${XDG_CACHE_HOME}"
EXPORT_VARS="${EXPORT_VARS} && export XDG_CONFIG_HOME=${XDG_CONFIG_HOME}"
EXPORT_VARS="${EXPORT_VARS} && export WANDB_ENTITY=pravsels"
EXPORT_VARS="${EXPORT_VARS} && export OPENPI_DATA_HOME=${data_dir}"
EXPORT_VARS="${EXPORT_VARS} && export UV_PROJECT_ENVIRONMENT=${data_dir}/.venv"
EXPORT_VARS="${EXPORT_VARS} && export HF_TOKEN=\$(tr -d '\n' < ${HF_TOKEN_FILE})"
EXPORT_VARS="${EXPORT_VARS} && export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95"

VALID_INDICES_PATH="${ASSETS_DIR}/valid_indices.txt"

PRECOMPUTE_CMD=""
if [ -f "${NORM_STATS_PATH}" ] && [ -f "${PER_TIMESTEP_STATS_PATH}" ]; then
    echo "Skipping normalization precompute (found stats files)."
else
    echo "Running normalization precompute..."
    echo "Command: ${COMPUTE_NORM_STATS_CMD}"
    echo ""
    PRECOMPUTE_CMD="${PRECOMPUTE_CMD}${COMPUTE_NORM_STATS_CMD} && "
fi

if [ ! -f "${VALID_INDICES_PATH}" ]; then
    echo "Generating valid_indices.txt (all frames valid for this dataset)..."
    PRECOMPUTE_CMD="${PRECOMPUTE_CMD}python -c \"
from openpi.training import config as _config
from openpi.training.data_loader import create_torch_dataset
cfg = _config.get_config('${CONFIG_NAME}')
data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
ds = create_torch_dataset(data_config, cfg.model.action_horizon, cfg.model)
n = len(ds)
with open('${VALID_INDICES_PATH}', 'w') as f:
    f.write(','.join(str(i) for i in range(n)))
print(f'Wrote {n} valid indices to ${VALID_INDICES_PATH}')
\" && "
fi

echo "Running training command..."
echo "Command: ${TRAIN_CMD}"
echo ""

set +e
srun --ntasks=1 --gpus-per-task=4 --cpu-bind=cores \
apptainer exec --nv \
    --pwd "${repo_dir}" \
    --bind "${scratch_dir}:${scratch_dir}" \
    --bind "${data_dir}/assets:${repo_dir}/assets" \
    --bind "${data_dir}/weights:${repo_dir}/weights" \
    --bind "${data_dir}/checkpoints:${repo_dir}/checkpoints" \
    --bind "${HF_CACHE}:/root/.cache/huggingface" \
    --env "HF_HOME=/root/.cache/huggingface" \
    "${container}" \
    bash -c "${EXPORT_VARS} && ${PRECOMPUTE_CMD}${TRAIN_CMD}"
EXIT_CODE=$?
set -e

end_time="$(date -Is --utc)"

echo ""
echo "===================================="
echo "Started (UTC):  ${start_time}"
echo "Finished (UTC): ${end_time}"
echo "Exit Code: ${EXIT_CODE}"
echo "Checkpoint location: ${CHECKPOINT_DIR}"
echo "===================================="

if [ ${EXIT_CODE} -ne 0 ]; then
    echo ""
    echo "ERROR: Training failed with exit code ${EXIT_CODE}"
    echo "Check slurm-${SLURM_JOB_ID}.err for detailed error messages"
    exit ${EXIT_CODE}
fi

exit 0
