# UR5 twin-wrist π0.5 runbook

All machines must use OpenPI commit `215abfb217dbac7d5f1273282331b9b1866c0479` plus the same `twinpath-pi05`
custom commit, official `pyproject.toml`/`uv.lock`, Python 3.11, schema and `ACTION_SPACE.md` hashes. Create `.venv`
independently with `uv venv --python 3.11 && uv sync --frozen`.

## 4090 collection and conversion

```bash
# No hardware open / no motion. Archive this report with the session.
uv run examples/ur5_twinwrist/teleop_preflight.py \
  > data/teleop_preflight_$(date +%Y%m%dT%H%M%S).json
uv run examples/ur5_twinwrist/teleop_collect.py \
  --episodes 1 --record-hz 10 --no-dashboard

# Software-only writer acceptance.
uv run examples/ur5_twinwrist/record_hdf5.py --fake --output /tmp/ur5_raw
uv run examples/ur5_twinwrist/validate_dataset.py /tmp/ur5_raw --max-camera-skew-ms 50
uv run examples/ur5_twinwrist/convert_to_lerobot.py --source /tmp/ur5_raw --repo-id local/ur5_twinwrist

# Real collection only after preflight says ready=true and the physical checklist is complete.
uv run examples/ur5_twinwrist/teleop_collect.py \
  --continuous --record-hz 10 --gripper-backend hiwonder \
  --output data/raw/task2_$(date +%Y%m%dT%H%M%S) \
  --enable-motion --confirm I_UNDERSTAND_REAL_ROBOT_MOTION

H100_HOST=... H100_USER=... H100_DATA_ROOT=... LOCAL_DATA_ROOT=... \
  examples/ur5_twinwrist/sync_data_to_h100.sh --dry-run
```

The current read-only preflight sees only one configured D435 and none of the three configured serial by-id nodes, so
the real-motion command is intentionally blocked at present. Failed, discarded, stale, serial-error and legacy
emergency-save episodes go to `rejected/`; Fit seals the task segment before HOME frames.

## H100 training

First run the exact script help and a 100-step smoke with the frozen code/dataset. This local runbook does not submit
CCI or ACP: the SenseCore gate requires a fresh confirmation, then one GPU/GPU0/BS1/one step CCI before four-H100 ACP.

```bash
uv run scripts/train.py --help
uv run scripts/compute_norm_stats.py --help
CUDA_VISIBLE_DEVICES=0,1,2,3 TRAIN_STEPS=100 BATCH_SIZE=16 FSDP_DEVICES=4 \
  examples/ur5_twinwrist/train_h100.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 TRAIN_STEPS=10000 BATCH_SIZE=16 FSDP_DEVICES=4 \
  examples/ur5_twinwrist/train_h100.sh
```

Accept only finite loss/gradient, current run evidence, a complete Orbax checkpoint and completion marker—never merely
`RUNNING`. No JAX/PyTorch conversion is involved.

## 4090 shadow inference

```bash
H100_HOST=... H100_USER=... H100_CHECKPOINT_ROOT=... LOCAL_CHECKPOINT_ROOT=... \
  examples/ur5_twinwrist/sync_checkpoint_to_4090.sh --dry-run
CHECKPOINT=/absolute/orbax/checkpoint PROMPT='place the block' \
  examples/ur5_twinwrist/run_infer_4090.sh
```

The wrapper starts localhost policy server, waits for the port, runs a shadow client and cleans up. First real rollout
must execute one chunk step at low speed. Increase to 2–4 only after stable logs. NaN/Inf, stale observation, camera or
policy timeout, serial error, limit violation or e-stop must stop/hold.

## Rollback

The integration is isolated on `twinpath-pi05`. Stop runtime/server, return to the recorded official commit in a new
worktree or branch, and retain raw/rejected episodes and checkpoints. Never reset the dirty legacy tree. rsync scripts
do not use `--delete`; a dry run is available and partial transfers resume.
