# UR5 twin-wrist π0.5 pipeline

This directory is the thin integration layer for the official OpenPI baseline. Hardware reuse is lazy-imported from
the legacy repository; no legacy environment or OpenPI core is copied. The locked action is 6D accepted `speedL`
twist, two absolute wrist targets, and one absolute gripper command. Read [TELEOPERATION.md](TELEOPERATION.md) first,
then [ACTION_SPACE.md](ACTION_SPACE.md).

Teleoperation collection preflight and dry-run (neither moves hardware):

```bash
uv run examples/ur5_twinwrist/teleop_preflight.py
uv run examples/ur5_twinwrist/teleop_collect.py --episodes 1 --record-hz 10 --no-dashboard
```

Software-only acceptance:

```bash
uv run examples/ur5_twinwrist/record_hdf5.py --fake --output /tmp/ur5_raw
uv run examples/ur5_twinwrist/validate_dataset.py /tmp/ur5_raw
uv run pytest -q tests/test_ur5_twinwrist_transforms.py tests/test_ur5_twinwrist_dataset.py \
  tests/test_ur5_twinwrist_legacy_collection.py tests/test_ur5_twinwrist_gripper_worker.py
uv run examples/ur5_twinwrist/robot_runtime.py --mode infer --shadow --fake
```

Copy `config.example.toml` to a git-ignored station config and fill UR IP, `/dev/serial/by-id/` paths and D435 serials.
Real movement always requires `--enable-motion`; inference defaults to shadow. Any stale/invalid observation,
non-finite action, hardware fault, serial failure or workspace violation is fail-closed.

The current legacy configuration is dirty and station-specific. `teleop_preflight.py` reports its commit, relevant
file status and hashes; archive that JSON beside every collection session. Do not use the stale legacy
`sc9/sc9x/scwx` shortcuts.
