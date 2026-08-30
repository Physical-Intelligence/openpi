# π0.5 pipeline audit

Audit date: 2026-08-31 CST. Backend is explicitly **JAX**.

## Baseline evidence

- New official recursive clone: `/home/user/haitao_files/robowrist/pi0.5/openpi`
- Initial OpenPI commit: `215abfb217dbac7d5f1273282331b9b1866c0479`; branch: `twinpath-pi05`.
- Submodules: Aloha `d1dc83a`, LIBERO `f78abd6`.
- Official `uv.lock` SHA-256: `793488b5a55bb87200db90a61fd0af51922b686d94e1da4f4c587ab119b37d74`.
- Locked LeRobot source: Hugging Face Git commit `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`.
- Host Python was 3.13.13; the new `.venv` uses CPython 3.11.15. uv is 0.12.3.
- GPU: RTX 4090; driver 580.95.05. No remote H100 inspection was performed.
- Legacy reference commit: `3f94334a29b4f94270f284c29dba01713dcc63a5`, with extensive user changes; untouched.
- The OpenPI worktree already contained unrelated deletions under `examples/droid/` and
  `src/openpi/policies/aloha_policy.py`. They were not created, restored or modified by this phase; the missing Aloha
  policy can break broad training-config imports even though the focused collection tests do not use it.

## Existing module topology

```text
collect_real / real_workflows
  -> site_adapter.StationSession + safety supervisor
     -> SpaceMouse client/device/evdev/mapping
     -> UR5 worker/runtime -> RTDE speedL (and speedJ only for home/wrist-3 special mode)
     -> OpenRB two-axis wrist teleop -> calibrated absolute target_q
     -> gripper factory -> Hiwonder or Feetech serial driver
  -> RealSenseCapture -> per-serial capture threads -> FrameSynchronizer
  -> StationSynchronizer -> VLA recorder / LeRobot writers
deploy_real_jax -> OpenPI/JAX policy -> RealPolicyBridge -> the same safety supervisor
```

## Reusable evidence

- UR5: `src/slai_mi/devices/ur5/{worker,runtime,process}.py` and guarded calls in `site_adapter.py`.
- Wrist: `src/slai_mi/devices/wrist_sensor/{openrb_v2,teleop}.py`.
- Grippers: `src/slai_mi/devices/gripper/hiwonder.py`, `feetech_sts3215.py`, and `factory.py`.
- SpaceMouse: `src/slai_mi/devices/spacemouse/`; mappings and button gates already tested.
- Three-camera sync: `realsense_capture.py` binds serials, imports RealSense only in `start()`, and minimizes host-time skew.
- Recording/inference: `collection/vla_recorder.py`, `datasets/pi05_writer.py`, `apps/deploy_real_jax.py` are reference implementations.
- Safety: workspace guard, speed limits, supervisor heartbeats, stale-state handling and flight recorder are reusable.

## Side effects and concurrency risks

Core modules generally defer hardware open until construction/start, but CLI entry modules must not be used as library
factories until their `main` guards are verified. RealSense correctly defers `pyrealsense2` import to `start()`.
The legacy gripper control path calls `read_state()` from control, HOME and recorder paths; individual driver locks
prevent byte interleaving but still let several business threads own the serial link. The phase-two adapter now wraps
the selected backend with `SingleOwnerGripper`: only one worker may open/read/write/close, all consumers read a cache,
and serial exceptions permanently fail closed. Fake tests prove all delegate calls use one non-caller thread; real USB
stress testing remains pending.

## Actual state/action contract

The legacy 9-DoF schema records state `[UR actual_q(6 rad), wrist actual_q(2 rad), gripper actual(1 normalized)]`.
Normal SpaceMouse arm commands go through mapping, speed selection and workspace guard, then use RTDE `speedL`.
However the unmodified legacy data is not always the final command: HOME and keys 1/2 use `speedJ` while recorded TCP
action is zero, the UR worker may floor-clamp Z after the parent snapshot, and the old synchronizer read adjacent
control attributes in two publication steps. The phase-two adapter therefore publishes a receipt only after one full
command cycle has completed, rejects
nonzero `target_qd`, and rejects any command that could activate the worker floor clamp. Accepted action is
`[TCP velocity(6), wrist absolute target(2), gripper absolute target(1)]`, never raw SpaceMouse axes. Capture stays
30 Hz and the new raw HDF5 sink samples at 10 Hz by default.

## Phase-two collection audit findings

- The production camera pairing logic is `devices/cameras/realsense_capture.py::FrameSynchronizer`, not the richer
  unused `collection/synchronization.py::RealFrameSynchronizer`.
- `StationSynchronizer` takes `now` before its blocking camera read. UR and SpaceMouse receive that fabricated same
  timestamp/sequence; only cameras, wrist and gripper carry device/cache timestamps.
- Existing legacy collection writes about 30 Hz directly to LeRobot. It does not implement requested 30 Hz capture /
  10 Hz recording, per-episode HDF5 temp files, atomic rename or rejected episodes.
- A read-only sample of the latest legacy dataset contained 12,217 frames. Wrist state age exceeded the configured
  100 ms limit on 300 frames (maximum 334.8 ms) while validity remained 1; 1,761 frames had nonzero UR `target_qd`
  but zero TCP action. Thus telemetry limits were descriptive, not recording gates.
- Legacy Fit records the entire coordinated HOME and abnormal exceptions call an emergency success save. The new sink
  seals on Fit, writes no HOME frames, and recognizes/rejects the legacy emergency call path.
- Camera role and state channel both use the name `wrist` in the 7-source telemetry vector. The adapter preserves the
  ordered source indices instead of collapsing them into a dict.
- Cross-camera skew is evaluated from aligned host monotonic timestamps. Raw RealSense timestamps remain stored per
  camera for duplicate/diagnostic checks and are never assumed to share a clock.

The full code tree, hard-coded key map, exact Home/zero distinctions and parameter locations are documented in
`examples/ur5_twinwrist/TELEOPERATION.md`.

## Minimal changes

1. Keep the official model and train entry intact; register one data factory and `TrainConfig` only in `config.py`.
2. Add three-camera/9D policy transforms and crop model output from 32 to 9.
3. Lazy-import legacy hardware through an adapter; add HOME, completed-command receipt and one-owner-serial wrappers without
   changing the legacy tree.
4. Record crash-safe 10 Hz raw HDF5, validate it, then convert offline with locked LeRobot.
5. Gate all real output behind `--enable-motion`; inference is shadow by default and all faults stop motion.
6. Verify station-specific serial stability, wrist sign/limits, camera roles, UR PolyScope limits/TCP/payload and
   task-home on hardware. Current preflight detects only one of three configured cameras and no configured serial node,
   so the motion launcher correctly refuses to start.
