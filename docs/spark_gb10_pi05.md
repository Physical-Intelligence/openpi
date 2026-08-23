# pi0.5 on NVIDIA GB10 / DGX Spark

This branch provides a working ARM64 CUDA 13 path for pi0.5 inference and full
PyTorch fine-tuning. It keeps NVIDIA PyTorch 26.01 and TorchVision from the GB10
base image, uses CPU JAX only to read the official Orbax checkpoint, and converts
that checkpoint to SafeTensors before any GPU work.

The upstream CUDA-12 JAX and PyTorch 2.7 wheels are intentionally not installed:
they do not carry the GB10 `sm_121` kernel/runtime contract. MuJoCo-only ALOHA
dependencies are also pruned because upstream MuJoCo 2.3.7 has no compatible
Python 3.12 ARM64 wheel and is not needed to train or serve the G1 policy.

## Reproduce the verified runtime

Run these commands on the Spark host from this repository:

```bash
./scripts/spark/build_image.sh
./scripts/spark/prepare_pi05.sh
./scripts/spark/run_smoke.sh
```

The scripts preserve all model and report data under `/var/lib/openpi-spark`.
The smoke test is deliberately a full pi0.5 model, not the small `debug_pi05`
network. It runs denoising on the GPU, executes a backward and AdamW update, saves
the new model and optimizer state, then leaves evidence at:

- `reports/pi05_inference.json`
- `reports/pi05_train_smoke.log`
- `training/pi05_spark_smoke/gb10_full_model_smoke/1`

## G1 Fruit Ninja fine-tuning contract

pi0.5 does not learn from the PPO checkpoint or the AMP motion file directly. It
needs time-aligned behavior demonstrations in LeRobot format. Each frame must
contain these features:

| LeRobot feature | Type and shape | Meaning |
| --- | --- | --- |
| `observation.images.head` | RGB image, `3xHxW` or `HxWx3` | G1 head RealSense color frame |
| `observation.state` | `float32`, at most 32 values | fixed-order G1 proprioceptive state; 29 joint positions are the recommended starting contract |
| `action` | `float32[21]` | existing Fruit Ninja task command: 3 walking, 1 chop-phase speed, 17 upper-body residuals |
| task | string | language instruction such as `slice the fruit` |

Frames, state, and the action actually applied to the controller must share the
same timestamp. At 50 Hz, the configured 10-action horizon represents 0.2 s. Bad
or aborted demonstrations should be excluded rather than labeled as successes.

The G1 transform uses the head image as the base camera and supplies two masked
zero wrist-camera slots. OpenPI pads the 29-D state and 21-D action to pi0.5's
32-D internal width; policy output is cropped back to exactly 21 task actions.
It does not bypass the existing frozen Unitree stabilizer, action scaling, safety
supervisor, or operator arm/start gate.

Once the demonstrations are uploaded as a LeRobot dataset, start a persistent
full-model run with:

```bash
export OPENPI_G1_DATASET_REPO_ID=owner/dataset
export OPENPI_G1_EXPERIMENT=fruit_ninja_pi05_v1
export OPENPI_G1_NUM_TRAIN_STEPS=20000
./scripts/spark/train_g1.sh
```

Simulation rollouts produced by `g1-fruit-ninja-mjwarp` can be validated and
converted into this exact LeRobot contract before upload:

```bash
python scripts/spark/convert_g1_sim_demos.py \
  --raw-dir /openpi_assets/demonstrations/g1-fruit-ninja \
  --repo-id owner/g1-fruit-ninja-sim
```

The converter checks every episode's SHA-256, RGB, 29-joint state, 21-action
command, frame count, contiguous frame index, finite values, and exact 50 Hz
timestamps. It refuses failed episodes by default and creates private Hugging
Face datasets when `--push-to-hub` is explicitly supplied.

The launcher computes and persists normalization statistics before training. A
new experiment name is required for each run so an older checkpoint is not
silently replaced.

## Hardware boundary

A successful smoke test proves model execution and gradient updates on Spark
48fd. It does not prove the policy can safely control the physical G1. Before any
robot trial, validate the 21-D joint/order contract offline, replay output in
simulation, enforce finite/range/rate checks, and require a supervised operator
start with a zero-command fallback.
