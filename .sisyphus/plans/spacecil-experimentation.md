# SpaceCIL Experimentation Phase — From Infrastructure to Real-Robot Experiments

## TL;DR

> **Quick Summary**: Complete all remaining code tasks to bridge SpaceCIL's finished infrastructure (285 tests, 15 modules) to runnable real-robot experiments. This covers implementing the training loop body, data conversion pipeline, scorer/evaluation wiring, baseline configs, and CLI integration — all testable on CPU before hardware arrives.
> 
> **Deliverables**:
> - Fully functional `train_spacecil.py` with working `make_train_fn`, distillation integration, and baseline support
> - Data conversion script (`scripts/convert_rm75_data_to_lerobot.py`) for HDF5 → LeRobot format
> - Scorer validation utility script
> - 5 baseline config variants for ablation experiments
> - All evaluation/metrics/checkpoint paths wired and working
> - ~30+ new tests covering all new code paths
> 
> **Estimated Effort**: Medium-Large (12-16 tasks across 4 waves)
> **Parallel Execution**: YES — 4 waves
> **Critical Path**: Task 1 (imports/config fix) → Task 2 (make_train_fn) → Task 5 (uncomment main wiring) → Task 8 (distillation integration) → Task 12 (integration test)

---

## Context

### Original Request
Complete all remaining code tasks to take SpaceCIL (Paper A) from "infrastructure complete" to "ready for real-robot experiments." The robot is available soon; a separate GPU machine with working JAX is available.

### Interview Summary
**Key Discussions**:
- **Scope**: Paper A (SpaceCIL) only — Paper B (LunarCompose) deferred until Paper A experiments are stable
- **Teleop format**: Not decided — plan recommends HDF5 per-episode files (matches ALOHA/DROID patterns)
- **Robot access**: Available soon — plan front-loads CPU-testable code tasks
- **GPU access**: Separate machine with working JAX GPU — data may need transfer

**Research Findings**:
- `make_train_fn()` in `train_spacecil.py` raises `NotImplementedError` — the single biggest blocker
- `main()` execution flow is entirely commented out — script does nothing when run
- Scorers and eval_episodes are empty `{}` dicts — evaluation produces no results
- TaskRouter is complete but isolated (not wired to anything)
- BehaviorDistillation loss/memory/teacher is complete but never called
- Baseline config variants (fulltune, shared_lora, oracle, etc.) don't exist
- `--oracle-routing` and `--random-routing` CLI flags referenced in experiment guide but not implemented
- `repo_id` fields in all 4 task configs are placeholders (`placeholder/spacecil_{task}`)
- Upstream provides: `init_train_state()`, `train_step()`, `create_data_loader()` — all with well-documented signatures
- `LeRobotRM75DataConfig` is fully implemented and ready to use
- `compute_norm_stats.py` must run on real data before training

### Metis Review
**Identified Gaps** (addressed):
- **Model state carry between tasks**: `init_train_state` must be called ONCE; LoRA swapped per task outside JIT
- **Optimizer state reset**: Adam moments must be re-initialized per task (stale momentum is meaningless for new LoRA)
- **Router integration scope**: Deferred from this plan's core — oracle routing (task ID as CLI arg) is sufficient for initial experiments
- **Distillation as optional path**: `--enable-distillation` flag already exists; distillation is disabled by default
- **Per-task config resolution**: Use `get_config(f"spacecil_rm75_{task_id}")` per task (configs already exist)
- **Norm stats pre-computation**: All 4 task datasets must exist before starting continual sequence
- **Cross-machine data transfer**: Conversion runs on robot machine; training on GPU machine; LeRobot dataset is portable via `HF_LEROBOT_HOME`
- **Adapter bank checkpointing**: Must save after each task for crash recovery

---

## Work Objectives

### Core Objective
Make `train_spacecil.py` a fully functional training script that can run the complete SpaceCIL continual learning protocol: sequential task training with per-task LoRA adapters, optional behavior distillation, offline evaluation via scorers, and metric reporting.

### Concrete Deliverables
- `scripts/train_spacecil.py` — complete, runnable training script
- `scripts/convert_rm75_data_to_lerobot.py` — HDF5 → LeRobot conversion
- `scripts/validate_scorers.py` — scorer precision/recall validation utility
- `src/openpi/research/spacecil/config.py` — updated with baseline config variants + configurable `repo_id`
- Updated tests for all modified modules

### Definition of Done
- [ ] `JAX_PLATFORMS=cpu uv run pytest src/openpi/research/ -x -q` passes ≥300 tests (currently 285)
- [ ] `JAX_PLATFORMS=cpu uv run python scripts/train_spacecil.py --config spacecil_debug --task-sequence debug_task --num-steps-per-task 2` completes without error
- [ ] All `NotImplementedError` markers removed from `train_spacecil.py`
- [ ] All `# TODO:` comments in `train_spacecil.py` `main()` resolved
- [ ] All commented-out code blocks in `main()` either uncommented or replaced with working code

### Must Have
- Working `make_train_fn` that creates data loader, runs train_step loop, returns model
- Model state carried across tasks (init once, swap LoRA per task)
- Optimizer state re-initialized per task (fresh Adam moments for new LoRA)
- Scorer wiring with real scorer instances per task
- Eval episode loading from disk
- Adapter bank save/load after each task
- Metrics computation and logging after continual sequence completes
- At least 3 baseline config variants (no-distillation, oracle-routing, random-routing)
- Data conversion script following libero template pattern
- CLI flags: `--oracle-routing`, `--random-routing`
- All new code has co-located `*_test.py` tests

### Must NOT Have (Guardrails)
- Do NOT modify `scripts/train.py` or any openpi core training files
- Do NOT call `init_train_state` per task (destroys continual learning — must be called ONCE)
- Do NOT swap adapters inside JIT-compiled functions (causes recompilation)
- Do NOT add router training to `make_train_fn` — router training is a separate concern, deferred
- Do NOT require live robot rollouts for evaluation — offline scoring of pre-collected episodes only
- Do NOT create baseline configs as full standalone definitions — use `dataclasses.replace` on existing configs
- Do NOT assume any specific teleop framework in the conversion script — accept generic HDF5 input
- Do NOT hardcode `repo_id` paths — must be configurable via CLI or environment variable
- Do NOT store images as float32 in parquet — use `dtype: "image"` and pass uint8 arrays
- Do NOT suppress type errors with `# type: ignore` — fix the types

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: YES (pytest + conftest.py with JAX CPU fallback)
- **Automated tests**: YES (tests-after) — 285 tests already exist, targeting ≥300
- **Framework**: pytest via `JAX_PLATFORMS=cpu uv run pytest src/openpi/research/ -x -q`
- **Pattern**: Co-located `*_test.py` files following existing convention

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Training script**: Use Bash — run with debug config, assert exit code 0, check output for expected log lines
- **Config resolution**: Use Bash (Python one-liner) — `get_config(name)` must not raise
- **Data conversion**: Use Bash (Python one-liner) — verify dataset schema after conversion
- **Tests**: Use Bash — `pytest` must pass with ≥N tests

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately — foundation fixes, independent of each other):
├── Task 1: Fix imports and repo_id configurability in config.py [quick]
├── Task 2: Implement make_train_fn body [deep]
├── Task 3: Write data conversion script (HDF5 → LeRobot) [unspecified-high]
└── Task 4: Write scorer validation utility script [quick]

Wave 2 (After Wave 1 — wiring that depends on make_train_fn):
├── Task 5: Uncomment and wire main() execution flow [deep]
├── Task 6: Wire scorers and eval_episodes into ContinualHarness [quick]
├── Task 7: Add baseline config variants via dataclasses.replace [quick]
└── Task 8: Integrate distillation loss into training loop [deep]

Wave 3 (After Wave 2 — CLI, metrics, and integration):
├── Task 9: Add --oracle-routing and --random-routing CLI flags [quick]
├── Task 10: Uncomment metrics computation and adapter bank save [quick]
├── Task 11: Write compute_all_norm_stats.sh helper script [quick]
└── Task 12: End-to-end integration test with spacecil_debug [deep]

Wave FINAL (After ALL tasks — independent review):
├── Task F1: Plan compliance audit [oracle]
├── Task F2: Code quality review [unspecified-high]
├── Task F3: Full QA — run all scenarios [unspecified-high]
└── Task F4: Scope fidelity check [deep]

Critical Path: Task 1 → Task 2 → Task 5 → Task 8 → Task 12 → F1-F4
Parallel Speedup: ~60% faster than sequential
Max Concurrent: 4 (Waves 1 & 2)
```

### Dependency Matrix

| Task | Depends On | Blocks |
|------|-----------|--------|
| 1 | — | 2, 5, 7 |
| 2 | 1 | 5, 8, 12 |
| 3 | — | 12 |
| 4 | — | 12 |
| 5 | 1, 2 | 8, 9, 10, 12 |
| 6 | — | 5, 12 |
| 7 | 1 | 9, 12 |
| 8 | 2, 5 | 12 |
| 9 | 5, 7 | 12 |
| 10 | 5 | 12 |
| 11 | — | 12 |
| 12 | ALL 1-11 | F1-F4 |

### Agent Dispatch Summary

- **Wave 1** (4 tasks): T1 → `quick`, T2 → `deep`, T3 → `unspecified-high`, T4 → `quick`
- **Wave 2** (4 tasks): T5 → `deep`, T6 → `quick`, T7 → `quick`, T8 → `deep`
- **Wave 3** (4 tasks): T9 → `quick`, T10 → `quick`, T11 → `quick`, T12 → `deep`
- **FINAL** (4 tasks): F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs


- [ ] 1. Fix Missing Imports and Make repo_id Configurable in SpaceCIL Config

  **What to do**:
  - Add missing import `from scripts.train import init_train_state` to `scripts/train_spacecil.py` (currently only `train_step` is imported)
  - Make `repo_id` in `spacecil/config.py` configurable via environment variable or CLI parameter instead of hardcoded `placeholder/spacecil_{task}`
  - Pattern: `repo_id=os.environ.get('SPACECIL_DATA_PREFIX', 'placeholder') + f'/spacecil_{task}'` — allows override without code changes
  - Verify all 4 task configs + debug config still resolve after changes
  - Add `import os` if not already present in config.py

  **Must NOT do**:
  - Do NOT change the config names (must remain `spacecil_rm75_payload`, etc.)
  - Do NOT modify any other file in `src/openpi/training/`
  - Do NOT remove the `placeholder` default — it must work without env var set (for tests)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3, 4)
  - **Blocks**: Tasks 2, 5, 7
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References**:
  - `src/openpi/research/spacecil/config.py:55-70` — Current config definitions with `placeholder/spacecil_{task}` repo_id
  - `src/openpi/training/misc/polaris_config.py` — Example of how external configs handle environment-specific paths

  **API/Type References**:
  - `src/openpi/training/config.py:DataConfigFactory` (line ~166) — `repo_id: str` field definition
  - `scripts/train.py:init_train_state` (line 85) — The function that needs to be imported

  **Test References**:
  - `src/openpi/research/spacecil/config_test.py` — Existing config resolution tests (7 tests)

  **WHY Each Reference Matters**:
  - `config.py:55-70`: Shows exactly where `placeholder/spacecil_{task}` is used and the structure to modify
  - `polaris_config.py`: Shows the established pattern for environment-configurable paths in openpi configs
  - `init_train_state` signature: Needed to add the correct import statement

  **Acceptance Criteria**:
  - [ ] `from scripts.train import init_train_state` import added to `train_spacecil.py`
  - [ ] `repo_id` in config.py reads from `SPACECIL_DATA_PREFIX` env var with `placeholder` fallback
  - [ ] All existing 7 config tests still pass

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Config resolution with default placeholder
    Tool: Bash
    Preconditions: No SPACECIL_DATA_PREFIX env var set
    Steps:
      1. JAX_PLATFORMS=cpu uv run python -c "from openpi.training import config as _config; c = _config.get_config('spacecil_rm75_payload'); print(c.data.repo_id)"
      2. Assert output contains 'placeholder/spacecil_payload'
    Expected Result: Config resolves with placeholder repo_id
    Evidence: .sisyphus/evidence/task-1-config-default.txt

  Scenario: Config resolution with custom prefix
    Tool: Bash
    Preconditions: SPACECIL_DATA_PREFIX=myorg set
    Steps:
      1. SPACECIL_DATA_PREFIX=myorg JAX_PLATFORMS=cpu uv run python -c "from openpi.training import config as _config; c = _config.get_config('spacecil_rm75_payload'); print(c.data.repo_id)"
      2. Assert output contains 'myorg/spacecil_payload'
    Expected Result: Config resolves with custom repo_id
    Evidence: .sisyphus/evidence/task-1-config-custom.txt

  Scenario: Import verification
    Tool: Bash
    Preconditions: None
    Steps:
      1. JAX_PLATFORMS=cpu uv run python -c "from scripts.train_spacecil import main; print('import OK')"
      2. Assert output is 'import OK' (no ImportError)
    Expected Result: All imports resolve without error
    Evidence: .sisyphus/evidence/task-1-import-check.txt
  ```

  **Commit**: YES
  - Message: `fix: add missing imports and configurable repo_id to spacecil config`
  - Files: `scripts/train_spacecil.py`, `src/openpi/research/spacecil/config.py`
  - Pre-commit: `JAX_PLATFORMS=cpu uv run pytest src/openpi/research/spacecil/config_test.py -x -q`

- [ ] 2. Implement make_train_fn Body — The Core Training Loop

  **What to do**:
  - Replace the `NotImplementedError` in `make_train_fn()` (lines 76-91 of `train_spacecil.py`) with a working training loop
  - The function receives `config, train_state, state_sharding, mesh, rng` from the outer scope
  - The returned closure `train_fn(task_id: str)` must:
    1. Resolve the per-task config: `task_config = _config.get_config(f'spacecil_rm75_{task_id}')` (or `config` if task_id matches debug)
    2. Create data config: `data_config = task_config.data.create(task_config.assets_dirs, task_config.model)`
    3. Create data loader: `from openpi.training.data_loader import create_data_loader; loader = create_data_loader(task_config, sharding=data_sharding, shuffle=True)`
    4. JIT-compile train_step: `ptrain_step = jax.jit(functools.partial(train_step, task_config), in_shardings=(...), out_shardings=(...), donate_argnums=(1,))` — follow exact pattern from `scripts/train.py:243-248`
    5. Run training loop for `args.num_steps_per_task` steps: `for i in range(args.num_steps_per_task): batch = next(data_iter); train_state, info = ptrain_step(rng, train_state, batch)`
    6. Extract model: `model = nnx.merge(train_state.model_def, train_state.params)`
    7. Return `(model, collected_infos)`
  - **CRITICAL**: Do NOT call `init_train_state` inside the closure — the `train_state` is captured from outer scope and carried across tasks
  - **CRITICAL**: The LoRA parameters in `train_state.params` must be swapped before each task via `adapter_bank.merge_into_model()` — but this happens in `ContinualHarness.run_sequence()`, not in `make_train_fn`
  - **CRITICAL**: Reset optimizer state for LoRA params when switching tasks — call `train_state.tx.init(new_lora_params)` and merge into `train_state.opt_state`
  - Handle the `data_sharding` computation: `data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())` for replicated data
  - Add progress logging: `logging.info(f'Task {task_id} step {i}/{args.num_steps_per_task} loss={info["loss"]:.4f}')`

  **Must NOT do**:
  - Do NOT call `init_train_state` per task (destroys continual learning)
  - Do NOT swap adapters inside the JIT-compiled `ptrain_step`
  - Do NOT modify `scripts/train.py` or any openpi core files
  - Do NOT add router training logic here — router is a separate concern
  - Do NOT hardcode batch size or learning rate — these come from `task_config`

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 3, 4 in Wave 1; depends on Task 1)
  - **Parallel Group**: Wave 1
  - **Blocks**: Tasks 5, 8, 12
  - **Blocked By**: Task 1 (needs import fix)

  **References**:

  **Pattern References**:
  - `scripts/train.py:205-270` — The upstream training loop: mesh setup, JIT compilation, data loading, train_step invocation. **Copy the JIT sharding pattern exactly.**
  - `scripts/train.py:243-248` — Exact `jax.jit(functools.partial(train_step, config), in_shardings=..., out_shardings=..., donate_argnums=(1,))` pattern
  - `scripts/train.py:253-258` — The main training loop: `batch = next(data_iter); train_state, info = ptrain_step(train_rng, train_state, batch)`

  **API/Type References**:
  - `scripts/train.py:init_train_state` (line 85) — Signature: `(config, init_rng, mesh, *, resume) → tuple[TrainState, state_sharding]`
  - `scripts/train.py:train_step` (line 137) — Signature: `(config, rng, state, batch) → tuple[new_state, info_dict]` where info has `loss`, `grad_norm`, `param_norm`
  - `src/openpi/training/data_loader.py:create_data_loader` (line ~223) — Signature: `(config, *, sharding, shuffle, num_batches, skip_norm_stats) → DataLoader`
  - `src/openpi/training/utils.py:TrainState` — Fields: `step, params, model_def, opt_state, tx, ema_decay, ema_params`
  - `flax.nnx.merge(graph_def, state) → Module` — How to reconstruct a module from TrainState

  **Test References**:
  - `src/openpi/research/spacecil/continual_harness_test.py` — Shows how `train_fn` is mocked: `lambda task_id: (mock_model, [{'loss': 0.1}])`

  **WHY Each Reference Matters**:
  - `train.py:205-270`: The canonical JAX training loop — must replicate the mesh/sharding/JIT setup exactly or training will fail silently
  - `train.py:243-248`: The `donate_argnums=(1,)` is critical — it tells JAX to reuse the `train_state` buffer, halving memory usage
  - `continual_harness_test.py`: Shows the exact return type expected from `train_fn` — `(model, info_list)` — must match this contract

  **Acceptance Criteria**:
  - [ ] `NotImplementedError` removed from `make_train_fn`
  - [ ] Function signature unchanged: `make_train_fn(config, train_state, state_sharding, mesh, rng) → callable`
  - [ ] Returned callable has signature: `train_fn(task_id: str) → tuple[Module, list[dict]]`
  - [ ] Uses `get_config(f'spacecil_rm75_{task_id}')` for per-task data loading
  - [ ] Does NOT call `init_train_state` inside the closure

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: make_train_fn returns callable
    Tool: Bash
    Preconditions: JAX_PLATFORMS=cpu
    Steps:
      1. JAX_PLATFORMS=cpu uv run python -c "
         from scripts.train_spacecil import make_train_fn
         print('make_train_fn imported successfully')
         import inspect
         sig = inspect.signature(make_train_fn)
         print(f'Parameters: {list(sig.parameters.keys())}')
         assert list(sig.parameters.keys()) == ['config', 'train_state', 'state_sharding', 'mesh', 'rng']
         print('PASS: signature correct')
         "
    Expected Result: Function importable with correct 5-parameter signature
    Evidence: .sisyphus/evidence/task-2-signature-check.txt

  Scenario: No NotImplementedError in file
    Tool: Bash
    Preconditions: None
    Steps:
      1. grep -c 'NotImplementedError' scripts/train_spacecil.py
      2. Assert count is 0
    Expected Result: Zero occurrences of NotImplementedError
    Evidence: .sisyphus/evidence/task-2-no-notimpl.txt
  ```

  **Commit**: YES
  - Message: `feat: implement make_train_fn body for spacecil training loop`
  - Files: `scripts/train_spacecil.py`
  - Pre-commit: `JAX_PLATFORMS=cpu uv run pytest src/openpi/research/spacecil/ -x -q`


- [ ] 3. Write Data Conversion Script (HDF5 → LeRobot Format)

  **What to do**:
  - Create `scripts/convert_rm75_data_to_lerobot.py` following the `examples/libero/convert_libero_data_to_lerobot.py` template
  - Accept CLI arguments: `--raw-dir` (path to directory of HDF5 episode files), `--repo-id` (LeRobot dataset name, e.g., `myorg/spacecil_payload`), `--fps` (recording frequency, default 30), `--task-label` (language instruction string)
  - HDF5 input schema assumption (document clearly, make configurable):
    - Each file = one episode
    - Keys: `joint_position (N, 7)`, `joint_velocity (N, 7)`, `gripper_position (N, 1)`, `wrist_image (N, H, W, 3)`, `action_joint (N, 7)`, `action_gripper (N, 1)`
    - Allow key name overrides via `--key-map` JSON argument
  - Use `LeRobotDataset.create()` API with features matching `LeRobotRM75DataConfig` expectations:
    - `observation/wrist_image` (dtype: image, shape: camera resolution)
    - `observation/joint_position` (dtype: float32, shape: (7,))
    - `observation/joint_velocity` (dtype: float32, shape: (7,))
    - `observation/gripper_position` (dtype: float32, shape: (1,))
    - `actions` (dtype: float32, shape: (8,)) — concatenate joint + gripper
  - Call `dataset.add_frame()` per timestep, `dataset.save_episode()` per episode, `dataset.finalize()` at end
  - Set `use_videos=False` (PNG bytes in parquet — simpler, lossless)
  - Pass `task=task_label` to populate the language instruction
  - Add `--dry-run` flag that reads first episode and prints schema without writing
  - Add verification step at end: load dataset back and print summary stats

  **Must NOT do**:
  - Do NOT assume any specific teleop framework (ROS, vendor SDK)
  - Do NOT store images as float32 — use uint8 arrays with `dtype: 'image'`
  - Do NOT call `push_to_hub()` — local only
  - Do NOT hardcode image resolution — read from first frame

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 4)
  - **Blocks**: Task 12 (integration test)
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References**:
  - `examples/libero/convert_libero_data_to_lerobot.py` — The canonical conversion template (~104 lines). **Copy its structure exactly**: argparse → `LeRobotDataset.create()` → loop episodes → `add_frame()` → `save_episode()` → `finalize()`
  - `examples/aloha_real/convert_aloha_data_to_lerobot.py` — Shows HDF5 reading patterns with `h5py`, multi-camera handling, and episode-per-file structure. **Use this for HDF5 reading patterns.**
  - `examples/droid/convert_droid_data_to_lerobot.py` — Shows HDF5 + MP4 handling for DROID format

  **API/Type References**:
  - `lerobot.common.datasets.lerobot_dataset.LeRobotDataset.create()` — Factory method: `(repo_id, fps, robot_type, features, use_videos, image_writer_threads)`
  - `lerobot.common.datasets.lerobot_dataset.LeRobotDataset.add_frame(frame_dict)` — Adds one timestep
  - `lerobot.common.datasets.lerobot_dataset.LeRobotDataset.save_episode(task=str)` — Commits episode to parquet
  - `lerobot.common.datasets.lerobot_dataset.LeRobotDataset.finalize()` — Flushes metadata, MANDATORY before training

  **External References**:
  - LeRobot v2.1 dataset format: `meta/info.json`, `meta/tasks.jsonl`, `meta/episodes.jsonl`, `data/chunk-NNN/episode_NNNNNN.parquet`

  **WHY Each Reference Matters**:
  - `convert_libero_data_to_lerobot.py`: Exact API usage pattern — copy, don't reinvent
  - `convert_aloha_data_to_lerobot.py`: HDF5 reading with `h5py.File` — our raw format is HDF5
  - `LeRobotRM75DataConfig`: The feature key names MUST match exactly or the repack transform will fail silently

  **Acceptance Criteria**:
  - [ ] Script exists at `scripts/convert_rm75_data_to_lerobot.py`
  - [ ] Accepts `--raw-dir`, `--repo-id`, `--fps`, `--task-label`, `--dry-run` arguments
  - [ ] Produces valid LeRobot dataset with correct feature keys
  - [ ] `--dry-run` prints schema without writing

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Dry run with synthetic HDF5 episode
    Tool: Bash
    Preconditions: Create a temporary HDF5 file with synthetic data
    Steps:
      1. JAX_PLATFORMS=cpu uv run python -c "
         import h5py, numpy as np, tempfile, os
         d = tempfile.mkdtemp()
         with h5py.File(os.path.join(d, 'episode_000.hdf5'), 'w') as f:
             f.create_dataset('joint_position', data=np.random.randn(50, 7).astype(np.float32))
             f.create_dataset('joint_velocity', data=np.random.randn(50, 7).astype(np.float32))
             f.create_dataset('gripper_position', data=np.random.randn(50, 1).astype(np.float32))
             f.create_dataset('wrist_image', data=np.random.randint(0, 255, (50, 224, 224, 3), dtype=np.uint8))
             f.create_dataset('action_joint', data=np.random.randn(50, 7).astype(np.float32))
             f.create_dataset('action_gripper', data=np.random.randn(50, 1).astype(np.float32))
         print(d)
         "
      2. Run: uv run python scripts/convert_rm75_data_to_lerobot.py --raw-dir $TMPDIR --repo-id test/rm75_test --fps 30 --task-label 'test task' --dry-run
      3. Assert output shows feature schema with correct key names and shapes
    Expected Result: Dry run prints schema without creating dataset
    Evidence: .sisyphus/evidence/task-3-dry-run.txt

  Scenario: Full conversion with synthetic data
    Tool: Bash
    Preconditions: Same synthetic HDF5 as above
    Steps:
      1. Create synthetic HDF5 episode (same as above)
      2. Run conversion WITHOUT --dry-run
      3. Verify: uv run python -c "
         from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
         ds = LeRobotDataset('test/rm75_test')
         assert ds.meta.total_episodes == 1
         assert 'observation/wrist_image' in ds.meta.features
         assert 'observation/joint_position' in ds.meta.features
         assert ds.meta.features['actions']['shape'] == [8]
         print('PASS: dataset schema valid')
         "
    Expected Result: Dataset created with 1 episode, correct schema
    Evidence: .sisyphus/evidence/task-3-full-conversion.txt
  ```

  **Commit**: YES
  - Message: `feat: add RM75 HDF5 to LeRobot data conversion script`
  - Files: `scripts/convert_rm75_data_to_lerobot.py`
  - Pre-commit: `uv run python scripts/convert_rm75_data_to_lerobot.py --help` (verify CLI works)

- [ ] 4. Write Scorer Validation Utility Script

  **What to do**:
  - Create `scripts/validate_scorers.py` — a utility that computes precision/recall of scorers against manually-labeled episodes
  - Accept CLI args: `--episodes-dir` (path to directory of serialized Episode JSON files), `--labels-csv` (path to CSV with columns `episode_id,success`), `--task` (one of: payload, latch, clean, connector)
  - Load episodes using `Episode.from_dict()` from `episode_schema.py`
  - Instantiate the appropriate scorer (`PayloadTransferScorer`, `LatchActuationScorer`, etc.)
  - Run scorer on each episode, compare `scorer.score(episode).success` to manual label
  - Print confusion matrix, precision, recall, F1 (use sklearn.metrics if available, else compute manually)
  - Print threshold tuning suggestions if precision or recall < 0.80
  - Exit with code 1 if either precision or recall < 0.80

  **Must NOT do**:
  - Do NOT import sklearn as a hard dependency — gracefully fall back to manual computation
  - Do NOT modify existing scorer classes
  - Do NOT require live robot data — must work with serialized Episode objects

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 3)
  - **Blocks**: Task 12 (integration test)
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References**:
  - `src/openpi/research/shared/scorer_base.py` — All 4 scorer classes: `PayloadTransferScorer`, `LatchActuationScorer`, `SurfaceCleaningScorer`, `ConnectorMatingScorer`. Each takes an `Episode` and returns `ScorerResult(success, confidence, fail_type, details)`

  **API/Type References**:
  - `src/openpi/research/shared/episode_schema.py:Episode` — Has `from_dict()` classmethod and `steps: list[EpisodeStep]`
  - `src/openpi/research/shared/scorer_base.py:ScorerResult` — `success: bool, confidence: float, fail_type: str | None, details: dict`

  **WHY Each Reference Matters**:
  - `scorer_base.py`: Need the exact scorer interface to call `.score(episode)` and check `.success`
  - `episode_schema.py`: Need `Episode.from_dict()` to deserialize saved episodes

  **Acceptance Criteria**:
  - [ ] Script exists at `scripts/validate_scorers.py`
  - [ ] Accepts `--episodes-dir`, `--labels-csv`, `--task` arguments
  - [ ] Prints precision, recall, F1 scores
  - [ ] Exits with code 1 if precision or recall < 0.80

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Validation with synthetic perfect labels
    Tool: Bash
    Preconditions: Create synthetic episodes + CSV with matching labels
    Steps:
      1. Create test fixtures: 5 episodes where scorer should return success, 5 where it should return failure
      2. Create labels CSV matching scorer output exactly
      3. Run: uv run python scripts/validate_scorers.py --episodes-dir $TMPDIR/eps --labels-csv $TMPDIR/labels.csv --task payload
      4. Assert precision == 1.0 and recall == 1.0
    Expected Result: Perfect precision/recall when labels match scorer output
    Evidence: .sisyphus/evidence/task-4-perfect-labels.txt

  Scenario: Validation detects low recall
    Tool: Bash
    Preconditions: Create episodes where scorer misses some successes
    Steps:
      1. Create 10 episodes, label all as success, but make 5 have low joint displacement (scorer returns False)
      2. Run validation
      3. Assert exit code is 1 (recall < 0.80)
    Expected Result: Script exits with error when recall is below threshold
    Evidence: .sisyphus/evidence/task-4-low-recall.txt
  ```

  **Commit**: YES
  - Message: `feat: add scorer validation utility script`
  - Files: `scripts/validate_scorers.py`
  - Pre-commit: `uv run python scripts/validate_scorers.py --help`

- [ ] 5. Uncomment and Wire main() Execution Flow

  **What to do**:
  - Uncomment the blocked-out code in `main()` of `train_spacecil.py` (lines ~138-145):
    ```python
    mesh = jax.sharding.Mesh(jax.devices(), ('fsdp',))
    rng = jax.random.PRNGKey(args.seed)
    train_state, state_sharding = init_train_state(config, rng, mesh, resume=False)
    train_fn = make_train_fn(config, train_state, state_sharding, mesh, rng)
    result = harness.run_sequence(train_fn)
    ```
  - Ensure `init_train_state` is called ONCE here (not per task)
  - Wire `result` to the metrics computation and logging blocks (Tasks 9-10 will uncomment those)
  - Add `logging.info(f'Continual sequence complete. Result matrix shape: {result.result_matrix.shape}')` after `run_sequence`
  - Handle the `ContinualResult` return value: store it for metrics computation
  - Add error handling: `try/except` around `run_sequence` with adapter bank save on failure (crash recovery)

  **Must NOT do**:
  - Do NOT call `init_train_state` more than once
  - Do NOT modify `ContinualHarness` to hold model references — keep model in `make_train_fn` closure
  - Do NOT add GPU-specific code without `jax.devices()` fallback (must work on CPU for testing)

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 6, 7 in Wave 2)
  - **Parallel Group**: Wave 2
  - **Blocks**: Tasks 8, 9, 10, 12
  - **Blocked By**: Tasks 1, 2

  **References**:

  **Pattern References**:
  - `scripts/train.py:196-210` — Mesh creation and init_train_state call pattern in upstream
  - `scripts/train_spacecil.py:138-161` — The current commented-out blocks to uncomment and wire

  **API/Type References**:
  - `src/openpi/research/spacecil/continual_harness.py:ContinualResult` — Return type of `run_sequence()`: has `result_matrix`, `task_sequence`, `training_infos`
  - `scripts/train.py:init_train_state` — `(config, rng, mesh, *, resume) → (TrainState, state_sharding)`

  **WHY Each Reference Matters**:
  - `train.py:196-210`: The exact mesh/init pattern — must match or sharding will fail
  - `ContinualResult`: Need to know what fields are available for metrics computation

  **Acceptance Criteria**:
  - [ ] No commented-out `mesh =`, `rng =`, `init_train_state(`, `run_sequence(` lines remaining
  - [ ] `harness.run_sequence(train_fn)` is called and result is captured
  - [ ] Error handling with adapter bank save on failure

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: main() calls run_sequence (verify via code inspection)
    Tool: Bash
    Preconditions: None
    Steps:
      1. grep -n 'run_sequence' scripts/train_spacecil.py
      2. Assert at least one uncommented call to harness.run_sequence
      3. grep -n '# mesh =' scripts/train_spacecil.py (check no commented-out mesh)
      4. Assert zero results (all uncommented)
    Expected Result: run_sequence is called, no commented-out execution blocks
    Evidence: .sisyphus/evidence/task-5-wiring-check.txt

  Scenario: Error handling saves adapter bank on crash
    Tool: Bash
    Preconditions: None
    Steps:
      1. grep -A5 'except' scripts/train_spacecil.py
      2. Assert adapter_bank.save is called in except/finally block
    Expected Result: Crash recovery saves adapter state
    Evidence: .sisyphus/evidence/task-5-error-handling.txt
  ```

  **Commit**: YES (groups with Task 6)
  - Message: `feat: wire main() execution flow, scorers, and eval episodes`
  - Files: `scripts/train_spacecil.py`
  - Pre-commit: `JAX_PLATFORMS=cpu uv run pytest src/openpi/research/spacecil/ -x -q`

- [ ] 6. Wire Scorers and Eval Episodes into ContinualHarness

  **What to do**:
  - Replace `scorers={}` in `main()` with real scorer instances:
    ```python
    from openpi.research.shared.scorer_base import (
        PayloadTransferScorer, LatchActuationScorer,
        SurfaceCleaningScorer, ConnectorMatingScorer,
    )
    scorers = {
        'payload': PayloadTransferScorer(),
        'latch': LatchActuationScorer(),
        'clean': SurfaceCleaningScorer(),
        'connector': ConnectorMatingScorer(),
    }
    ```
  - Replace `eval_episodes={}` with a loading function:
    ```python
    def load_eval_episodes(eval_dir: str, task_ids: list[str]) -> dict[str, list[Episode]]:
        episodes = {}
        for task_id in task_ids:
            task_dir = os.path.join(eval_dir, task_id)
            if os.path.isdir(task_dir):
                episodes[task_id] = [Episode.from_dict(json.load(open(f))) for f in sorted(glob.glob(f'{task_dir}/*.json'))]
            else:
                episodes[task_id] = []
        return episodes
    ```
  - Add `--eval-dir` CLI argument (default: `data/eval_episodes/`)
  - If eval_dir doesn't exist or is empty, log warning and use empty dict (graceful degradation)
  - For debug config, allow empty eval_episodes (harness handles empty dicts)

  **Must NOT do**:
  - Do NOT require eval episodes to exist for the script to run — must degrade gracefully
  - Do NOT modify scorer classes
  - Do NOT require live robot rollouts

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 5, 7, 8 in Wave 2)
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 12
  - **Blocked By**: None (can start immediately, but logically part of Wave 2)

  **References**:

  **Pattern References**:
  - `src/openpi/research/shared/scorer_base.py:PayloadTransferScorer` — Constructor: `(goal_region_threshold=0.1)` — all scorers have sensible defaults
  - `src/openpi/research/shared/episode_schema.py:Episode.from_dict()` — Deserializes JSON to Episode

  **API/Type References**:
  - `src/openpi/research/spacecil/continual_harness.py:ContinualHarness.__init__` — `scorers: dict[str, Scorer]`, `eval_episodes: dict[str, list[Episode]]`

  **WHY Each Reference Matters**:
  - `scorer_base.py`: Need exact class names and constructor signatures
  - `ContinualHarness.__init__`: Need exact type expectations for the dict parameters

  **Acceptance Criteria**:
  - [ ] `scorers={}` replaced with real scorer dict for all 4 tasks
  - [ ] `eval_episodes={}` replaced with loader function that reads from `--eval-dir`
  - [ ] Graceful degradation when eval_dir is empty/missing
  - [ ] New `--eval-dir` CLI argument added

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Scorers are wired (no empty dict)
    Tool: Bash
    Preconditions: None
    Steps:
      1. grep -n 'scorers={}' scripts/train_spacecil.py
      2. Assert zero results (no empty scorers dict)
      3. grep -n 'PayloadTransferScorer' scripts/train_spacecil.py
      4. Assert at least one result (scorer is imported and used)
    Expected Result: Real scorers wired, no empty placeholder
    Evidence: .sisyphus/evidence/task-6-scorers-wired.txt

  Scenario: Graceful degradation with no eval dir
    Tool: Bash
    Preconditions: No eval_episodes directory exists
    Steps:
      1. JAX_PLATFORMS=cpu uv run python -c "
         import sys; sys.argv = ['train', '--config', 'spacecil_debug', '--task-sequence', 'debug_task']
         # Verify the script doesn't crash on import/setup even without eval dir
         from scripts.train_spacecil import parse_args
         args = parse_args()
         print('PASS: args parsed without eval dir')"
    Expected Result: Script handles missing eval directory gracefully
    Evidence: .sisyphus/evidence/task-6-no-eval-dir.txt
  ```

  **Commit**: YES (groups with Task 5)
  - Message: `feat: wire scorers and eval episodes into ContinualHarness`
  - Files: `scripts/train_spacecil.py`
  - Pre-commit: `JAX_PLATFORMS=cpu uv run pytest src/openpi/research/spacecil/ -x -q`

- [ ] 7. Add Baseline Config Variants via dataclasses.replace

  **What to do**:
  - Add 5 baseline config variants to `spacecil/config.py` using `dataclasses.replace` on existing task configs:
    1. **`spacecil_rm75_{task}_fulltune`** — Full fine-tuning (no LoRA freeze): `freeze_filter=nnx.Nothing` (NOTE: requires >70GB GPU, flag in docstring)
    2. **`spacecil_rm75_{task}_nodistill`** — Same as base but explicitly marks no-distillation (identical config, different name for tracking)
    3. **`spacecil_rm75_shared_lora`** — Single shared LoRA across all tasks (no adapter bank): same LoRA params, no per-task swap
    4. **`spacecil_rm75_{task}_oracle`** — Per-task PEFT with oracle routing (same as base config, different name for tracking — routing is a CLI flag)
    5. **`spacecil_rm75_{task}_random`** — Per-task PEFT with random routing (same as base config, different name — routing is a CLI flag)
  - Use a helper function to generate variants from base configs:
    ```python
    def _make_baseline_variants(base_configs: list[TrainConfig]) -> list[TrainConfig]:
        variants = []
        for cfg in base_configs:
            # fulltune variant
            variants.append(dataclasses.replace(cfg, name=f'{cfg.name}_fulltune', freeze_filter=None))
            # ... etc
        return variants
    ```
  - Add all variants to the return value of `get_spacecil_configs()`
  - Add tests: each variant resolves without error

  **Must NOT do**:
  - Do NOT create full standalone config definitions — use `dataclasses.replace` on existing configs
  - Do NOT duplicate model/data definitions
  - Do NOT add configs that require different base model weights

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 5, 6, 8 in Wave 2)
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 9, 12
  - **Blocked By**: Task 1 (config.py changes)

  **References**:

  **Pattern References**:
  - `src/openpi/research/spacecil/config.py:get_spacecil_configs()` — Current function that returns 4 task configs + debug
  - `src/openpi/training/config.py:TrainConfig` — All fields available for `dataclasses.replace`: `name`, `freeze_filter`, `batch_size`, `num_train_steps`, etc.

  **API/Type References**:
  - `flax.nnx.Nothing` — The filter that matches nothing (used for no-freeze / full fine-tuning)
  - `src/openpi/training/config.py:TrainConfig.freeze_filter` — Controls which params are frozen

  **WHY Each Reference Matters**:
  - `get_spacecil_configs()`: Where variants must be added
  - `TrainConfig.freeze_filter`: The specific field to modify for fulltune variant

  **Acceptance Criteria**:
  - [ ] At least 5 new config names registered
  - [ ] All variants resolve via `_config.get_config(name)` without error
  - [ ] Fulltune variant has `freeze_filter=None` (or `nnx.Nothing`)
  - [ ] New test: `test_baseline_configs_resolve` in config_test.py

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: All baseline configs resolve
    Tool: Bash
    Preconditions: None
    Steps:
      1. JAX_PLATFORMS=cpu uv run python -c "
         from openpi.training import config as _config
         baselines = [
             'spacecil_rm75_payload_fulltune',
             'spacecil_rm75_payload_nodistill',
             'spacecil_rm75_shared_lora',
             'spacecil_rm75_payload_oracle',
             'spacecil_rm75_payload_random',
         ]
         for name in baselines:
             c = _config.get_config(name)
             print(f'OK: {name} -> batch_size={c.batch_size}')
         print('PASS: all baselines resolve')
         "
    Expected Result: All 5 baseline configs resolve without error
    Evidence: .sisyphus/evidence/task-7-baselines-resolve.txt

  Scenario: Fulltune variant has correct freeze filter
    Tool: Bash
    Preconditions: None
    Steps:
      1. JAX_PLATFORMS=cpu uv run python -c "
         from openpi.training import config as _config
         c = _config.get_config('spacecil_rm75_payload_fulltune')
         assert c.freeze_filter is None or str(c.freeze_filter) == 'Nothing', f'Expected no freeze, got {c.freeze_filter}'
         print('PASS: fulltune has no freeze filter')
         "
    Expected Result: Fulltune config has freeze_filter=None
    Evidence: .sisyphus/evidence/task-7-fulltune-freeze.txt
  ```

  **Commit**: YES
  - Message: `feat: add baseline config variants for SpaceCIL experiments`
  - Files: `src/openpi/research/spacecil/config.py`, `src/openpi/research/spacecil/config_test.py`
  - Pre-commit: `JAX_PLATFORMS=cpu uv run pytest src/openpi/research/spacecil/config_test.py -x -q`

- [ ] 8. Integrate Distillation Loss into Training Loop

  **What to do**:
  - Modify `make_train_fn` to optionally apply behavior distillation when `args.enable_distillation` is True
  - The distillation integration pattern (wraps AROUND train_step, does NOT modify it):
    1. Before each task (except first): snapshot teacher from previous task's model
    2. During training: after computing `train_step` loss, compute distillation loss separately
    3. Combine: `total_loss = task_loss + alpha * distillation_loss`
  - Use `BehaviorDistillation.compute_total_loss(task_loss, student_actions, teacher_actions)`
  - The teacher forward pass: reconstruct model from `TeacherSnapshot`, run on calibration batch, get action predictions
  - Add calibration episodes to `CalibrationMemory` after each task: `distillation.memory.add_calibration_episodes(task_id, calibration_episodes)`
  - **IMPORTANT**: Since `train_step` is JIT-compiled and we can't modify it, the distillation approach should be:
    - Option A: Write a custom `train_step_with_distillation` that calls `model.compute_loss()` directly (like `train_step` does internally) and adds the distillation term before computing gradients
    - Option B: Run train_step for task loss, then separately compute and apply distillation gradients
    - **Recommend Option A** for correctness (single backward pass with combined loss)
  - This is an OPTIONAL code path: when `--enable-distillation false` (default), skip entirely
  - Add `--calibration-dir` CLI argument for loading calibration episodes

  **Must NOT do**:
  - Do NOT modify `scripts/train.py:train_step` — write a separate function
  - Do NOT make distillation the default — it must be opt-in via `--enable-distillation`
  - Do NOT require calibration episodes to exist — graceful degradation with warning
  - Do NOT add router training here

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Task 5 completing first)
  - **Parallel Group**: Wave 2 (sequential after Task 5)
  - **Blocks**: Task 12
  - **Blocked By**: Tasks 2, 5

  **References**:

  **Pattern References**:
  - `src/openpi/research/spacecil/behavior_distillation.py` — Full file: `BehaviorDistillation`, `TeacherSnapshot`, `CalibrationMemory`, `compute_total_loss(task_loss, student_actions, teacher_actions)`
  - `scripts/train.py:train_step` (lines 137-191) — The loss computation pattern: `loss = model.compute_loss(observations, actions, ...)` — replicate this for the combined loss

  **API/Type References**:
  - `BehaviorDistillation.compute_total_loss(task_loss, student_actions, teacher_actions) → total_loss` — Combines task + distillation losses with alpha weight
  - `TeacherSnapshot.snapshot(model)` — Snapshot current model as teacher (stores params copy)
  - `TeacherSnapshot.get_params() → nnx.State` — Retrieve teacher params for forward pass (reconstruct model with `nnx.merge(model_def, teacher_params)` to get teacher actions)
  - `CalibrationMemory.add_calibration_episodes(task_id, episodes)` — Store calibration data
  - `CalibrationMemory.sample_batch(batch_size) → batch` — Sample replay batch

  **WHY Each Reference Matters**:
  - `behavior_distillation.py`: The complete distillation API — must use these exact methods
  - `train.py:train_step:137-191`: Shows how `model.compute_loss()` is called inside JIT — replicate for custom step

  **Acceptance Criteria**:
  - [ ] `--enable-distillation true` activates distillation code path
  - [ ] `--enable-distillation false` (default) skips distillation entirely
  - [ ] `--calibration-dir` CLI argument added
  - [ ] `BehaviorDistillation.compute_total_loss` is called when distillation is enabled
  - [ ] Teacher snapshot is updated after each task
  - [ ] No modification to `scripts/train.py`

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Distillation disabled by default
    Tool: Bash
    Preconditions: None
    Steps:
      1. grep -n 'enable_distillation.*True\|enable_distillation.*=.*True' scripts/train_spacecil.py
      2. Verify default is False in SpaceCILArgs
      3. grep 'enable_distillation: bool = False' scripts/train_spacecil.py || grep 'enable_distillation: bool = True' scripts/train_spacecil.py
    Expected Result: Default is False (distillation opt-in)
    Evidence: .sisyphus/evidence/task-8-distillation-default.txt

  Scenario: Distillation code path exists
    Tool: Bash
    Preconditions: None
    Steps:
      1. grep -n 'compute_total_loss' scripts/train_spacecil.py
      2. Assert at least one result (distillation loss function is called)
      3. grep -n 'update_teacher\|TeacherSnapshot' scripts/train_spacecil.py
      4. Assert at least one result (teacher is updated)
    Expected Result: Distillation functions are called in the training loop
    Evidence: .sisyphus/evidence/task-8-distillation-wired.txt
  ```

  **Commit**: YES
  - Message: `feat: integrate distillation loss into spacecil training loop`
  - Files: `scripts/train_spacecil.py`
  - Pre-commit: `JAX_PLATFORMS=cpu uv run pytest src/openpi/research/spacecil/ -x -q`

- [ ] 9. Add --oracle-routing and --random-routing CLI Flags

  **What to do**:
  - Add two new CLI flags to `SpaceCILArgs` and `parse_args()`:
    - `--oracle-routing` (bool, default False): When True, the adapter bank uses the ground-truth task ID at eval time (no router needed)
    - `--random-routing` (bool, default False): When True, adapter selection is random at eval time
  - These flags affect ONLY the evaluation path in `ContinualHarness`, not the training loop
  - Pass flags through to harness or configure routing strategy:
    ```python
    routing_strategy = 'learned'  # default
    if args.oracle_routing:
        routing_strategy = 'oracle'
    elif args.random_routing:
        routing_strategy = 'random'
    ```
  - Add mutual exclusivity check: `assert not (args.oracle_routing and args.random_routing)`
  - Log the routing strategy at startup

  **Must NOT do**:
  - Do NOT implement learned routing in this task — only oracle and random
  - Do NOT modify `ContinualHarness` class — routing logic stays in the script

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 10, 11 in Wave 3)
  - **Parallel Group**: Wave 3
  - **Blocks**: Task 12
  - **Blocked By**: Tasks 5, 7

  **References**:

  **Pattern References**:
  - `scripts/train_spacecil.py:SpaceCILArgs` — Current dataclass with existing flags
  - `scripts/train_spacecil.py:parse_args()` — Current argparse setup
  - `customized_docs/Experiment_Guide_SpaceCIL.md` (line ~748) — Documents the `--oracle-routing` and `--random-routing` flags

  **Acceptance Criteria**:
  - [ ] `--oracle-routing` flag added and parsed
  - [ ] `--random-routing` flag added and parsed
  - [ ] Mutual exclusivity enforced
  - [ ] Routing strategy logged at startup

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Flags parsed correctly
    Tool: Bash
    Preconditions: None
    Steps:
      1. uv run python scripts/train_spacecil.py --help 2>&1 | grep -E 'oracle-routing|random-routing'
      2. Assert both flags appear in help output
    Expected Result: Both routing flags in CLI help
    Evidence: .sisyphus/evidence/task-9-cli-flags.txt

  Scenario: Mutual exclusivity
    Tool: Bash
    Preconditions: None
    Steps:
      1. JAX_PLATFORMS=cpu uv run python -c "
         import sys; sys.argv = ['train', '--config', 'spacecil_debug', '--task-sequence', 'debug_task', '--oracle-routing', '--random-routing']
         from scripts.train_spacecil import parse_args
         try:
             parse_args()
             print('FAIL: should have raised')
         except (AssertionError, SystemExit):
             print('PASS: mutual exclusivity enforced')
         "
    Expected Result: Error when both flags set
    Evidence: .sisyphus/evidence/task-9-mutual-exclusivity.txt
  ```

  **Commit**: YES (groups with Task 10)
  - Message: `feat: add oracle/random routing CLI flags and metrics logging`
  - Files: `scripts/train_spacecil.py`
  - Pre-commit: `uv run python scripts/train_spacecil.py --help`

- [ ] 10. Uncomment Metrics Computation and Adapter Bank Save

  **What to do**:
  - Uncomment the metrics computation block (lines ~151-157 of `train_spacecil.py`):
    ```python
    final_metrics = {
        'average_success': metrics.average_success(result.result_matrix),
        'backward_transfer': metrics.backward_transfer(result.result_matrix),
        'forgetting': metrics.forgetting(result.result_matrix),
        'operational_forgetting': metrics.operational_forgetting(
            result.result_matrix,
            weights=np.array([0.6, 0.8, 0.5, 1.0])  # payload, latch, clean, connector
        ),
    }
    ```
  - Make operational weights configurable via `--operational-weights` CLI arg (JSON string), with sensible defaults
  - Uncomment adapter bank save (line ~160):
    ```python
    adapter_bank.save(f'{args.checkpoint_dir}/{args.exp_name}/adapter_bank')
    ```
  - Add `logging.info(f'Final metrics: {final_metrics}')` for console output
  - Add `json.dump(final_metrics, open(f'{args.checkpoint_dir}/{args.exp_name}/metrics.json', 'w'))` for persistence
  - Ensure `os.makedirs(f'{args.checkpoint_dir}/{args.exp_name}', exist_ok=True)` is called before save

  **Must NOT do**:
  - Do NOT hardcode operational weights — make configurable with defaults
  - Do NOT skip adapter bank save — it's essential for crash recovery and inference

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 9, 11 in Wave 3)
  - **Parallel Group**: Wave 3
  - **Blocks**: Task 12
  - **Blocked By**: Task 5

  **References**:

  **Pattern References**:
  - `scripts/train_spacecil.py:151-161` — The commented-out metrics and save blocks
  - `src/openpi/research/spacecil/metrics.py` — All metric functions: `average_success`, `backward_transfer`, `forgetting`, `operational_forgetting`, `routing_entropy`, `routing_accuracy`
  - `src/openpi/research/spacecil/task_adapter_bank.py:save()` — Saves adapter bank to directory

  **Acceptance Criteria**:
  - [ ] Metrics computation uncommented and functional
  - [ ] Adapter bank save uncommented and functional
  - [ ] Metrics saved to JSON file
  - [ ] Operational weights configurable via CLI

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: No commented-out metrics blocks
    Tool: Bash
    Preconditions: None
    Steps:
      1. grep -n '# final_metrics\|# adapter_bank.save' scripts/train_spacecil.py
      2. Assert zero results (all uncommented)
    Expected Result: No commented-out metric/save code
    Evidence: .sisyphus/evidence/task-10-uncommented.txt

  Scenario: Metrics JSON output path exists
    Tool: Bash
    Preconditions: None
    Steps:
      1. grep -n 'metrics.json' scripts/train_spacecil.py
      2. Assert at least one result (metrics file is written)
    Expected Result: Metrics are persisted to JSON
    Evidence: .sisyphus/evidence/task-10-metrics-json.txt
  ```

  **Commit**: YES (groups with Task 9)
  - Message: `feat: uncomment metrics computation and adapter bank persistence`
  - Files: `scripts/train_spacecil.py`
  - Pre-commit: `JAX_PLATFORMS=cpu uv run pytest src/openpi/research/spacecil/ -x -q`

- [ ] 11. Write compute_all_norm_stats.sh Helper Script

  **What to do**:
  - Create `scripts/compute_all_norm_stats.sh` — a simple bash script that runs norm stats for all SpaceCIL task configs
    ```bash
    #!/bin/bash
    set -euo pipefail
    CONFIGS=(spacecil_rm75_payload spacecil_rm75_latch spacecil_rm75_clean spacecil_rm75_connector)
    for config in "${CONFIGS[@]}"; do
        echo "Computing norm stats for $config..."
        uv run scripts/compute_norm_stats.py --config-name "$config"
    done
    echo "All norm stats computed successfully."
    ```
  - Make executable: `chmod +x`
  - Add a check at the top: verify datasets exist before running
  - Add `--max-frames` pass-through for fast estimation during development

  **Must NOT do**:
  - Do NOT modify `compute_norm_stats.py` — it's upstream openpi code
  - Keep this simple — <30 lines

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 9, 10 in Wave 3)
  - **Parallel Group**: Wave 3
  - **Blocks**: Task 12
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `scripts/compute_norm_stats.py` — The upstream script: `--config-name` flag, `--max-frames` flag

  **Acceptance Criteria**:
  - [ ] Script exists at `scripts/compute_all_norm_stats.sh`
  - [ ] Executable flag set
  - [ ] Iterates over all 4 task configs
  - [ ] Passes through `--max-frames` if provided

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Script is executable and has correct configs
    Tool: Bash
    Preconditions: None
    Steps:
      1. test -x scripts/compute_all_norm_stats.sh && echo 'executable' || echo 'not executable'
      2. grep -c 'spacecil_rm75_' scripts/compute_all_norm_stats.sh
      3. Assert count is 4 (all task configs listed)
    Expected Result: Script is executable and references all 4 configs
    Evidence: .sisyphus/evidence/task-11-script-check.txt
  ```

  **Commit**: YES
  - Message: `feat: add norm stats computation helper for all SpaceCIL tasks`
  - Files: `scripts/compute_all_norm_stats.sh`
  - Pre-commit: `bash -n scripts/compute_all_norm_stats.sh` (syntax check)

- [ ] 12. End-to-End Integration Test with spacecil_debug Config

  **What to do**:
  - Create `src/openpi/research/spacecil/integration_test.py` with comprehensive end-to-end tests:
    1. **Test `make_train_fn` produces callable**: Import, check signature, verify no NotImplementedError
    2. **Test config resolution for all variants**: All base + baseline configs resolve
    3. **Test scorer wiring**: Verify scorers dict has all 4 task keys
    4. **Test eval episode loading**: Verify graceful degradation with empty dir
    5. **Test CLI flags parsing**: Verify all new flags (--oracle-routing, --random-routing, --eval-dir, --calibration-dir, --operational-weights)
    6. **Test ContinualHarness with mock train_fn**: Run `harness.run_sequence(mock_train_fn)` and verify result matrix shape
    7. **Test metrics computation on mock result**: Verify all metrics produce valid floats
  - Also create a **smoke test** that attempts to import and parse args for the full script
  - All tests must run with `JAX_PLATFORMS=cpu` and no real data
  - Target: 15+ new tests in this file

  **Must NOT do**:
  - Do NOT require GPU or real data for any test
  - Do NOT test actual JAX training (that's GPU-bound) — test wiring and contracts
  - Do NOT duplicate tests that already exist in other test files

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on ALL Tasks 1-11)
  - **Parallel Group**: Wave 3 (final, sequential after all others)
  - **Blocks**: F1-F4 (Final Verification)
  - **Blocked By**: All Tasks 1-11

  **References**:

  **Pattern References**:
  - `src/openpi/research/spacecil/continual_harness_test.py` — Existing test patterns: mock train_fn, fake scorers, result matrix assertions
  - `src/openpi/research/spacecil/config_test.py` — Config resolution test patterns

  **API/Type References**:
  - `src/openpi/research/spacecil/continual_harness.py:ContinualResult` — `result_matrix: np.ndarray`, `task_sequence: list[str]`
  - `src/openpi/research/spacecil/metrics.py` — All 6 metric functions

  **WHY Each Reference Matters**:
  - `continual_harness_test.py`: Shows existing mock patterns — reuse, don't reinvent
  - `config_test.py`: Shows how to test config resolution — extend for baseline variants

  **Acceptance Criteria**:
  - [ ] Test file at `src/openpi/research/spacecil/integration_test.py`
  - [ ] ≥15 tests covering all new functionality
  - [ ] All tests pass with `JAX_PLATFORMS=cpu`
  - [ ] Total test count for `src/openpi/research/` is ≥300

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Full test suite passes
    Tool: Bash
    Preconditions: All Tasks 1-11 complete
    Steps:
      1. JAX_PLATFORMS=cpu uv run pytest src/openpi/research/ -x -q 2>&1 | tail -5
      2. Assert output shows ≥300 passed, 0 failed
    Expected Result: All tests pass including new integration tests
    Evidence: .sisyphus/evidence/task-12-full-suite.txt

  Scenario: Integration tests specifically pass
    Tool: Bash
    Preconditions: All Tasks 1-11 complete
    Steps:
      1. JAX_PLATFORMS=cpu uv run pytest src/openpi/research/spacecil/integration_test.py -v 2>&1 | tail -20
      2. Assert ≥15 passed, 0 failed
    Expected Result: All integration tests pass individually
    Evidence: .sisyphus/evidence/task-12-integration-tests.txt
  ```

  **Commit**: YES
  - Message: `test: add end-to-end integration tests for spacecil training flow`
  - Files: `src/openpi/research/spacecil/integration_test.py`
  - Pre-commit: `JAX_PLATFORMS=cpu uv run pytest src/openpi/research/spacecil/integration_test.py -x -q`
---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Rejection → fix → re-run.

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, run command). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in `.sisyphus/evidence/`. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `JAX_PLATFORMS=cpu uv run pytest src/openpi/research/ -x -q`. Review all changed files for: `# type: ignore`, empty catches, `print()` in production code, commented-out code blocks, unused imports. Check AI slop: excessive comments, over-abstraction, generic variable names. Verify all new functions have docstrings.
  Output: `Tests [N pass/N fail] | Files [N clean/N issues] | VERDICT`

- [ ] F3. **Full QA — Run All Scenarios** — `unspecified-high`
  Execute EVERY QA scenario from EVERY task — follow exact steps, capture evidence. Test cross-task integration: run `train_spacecil.py` with `spacecil_debug` config end-to-end. Verify metrics are printed. Verify adapter bank directory is created. Save to `.sisyphus/evidence/final-qa/`.
  Output: `Scenarios [N/N pass] | Integration [N/N] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff (`git log`/`diff`). Verify 1:1 — everything in spec was built (no missing), nothing beyond spec was built (no creep). Check "Must NOT do" compliance. Flag unaccounted changes.
  Output: `Tasks [N/N compliant] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

| Wave | Commit Message | Files |
|------|---------------|-------|
| 1 | `fix: add missing imports and configurable repo_id to spacecil config` | `spacecil/config.py` |
| 1 | `feat: implement make_train_fn body for spacecil training loop` | `train_spacecil.py` |
| 1 | `feat: add RM75 HDF5 to LeRobot data conversion script` | `scripts/convert_rm75_data_to_lerobot.py` |
| 1 | `feat: add scorer validation utility script` | `scripts/validate_scorers.py` |
| 2 | `feat: wire main() execution flow, scorers, and eval episodes` | `train_spacecil.py` |
| 2 | `feat: add baseline config variants for SpaceCIL experiments` | `spacecil/config.py` |
| 2 | `feat: integrate distillation loss into spacecil training loop` | `train_spacecil.py` |
| 3 | `feat: add oracle/random routing CLI flags and metrics logging` | `train_spacecil.py` |
| 3 | `feat: add norm stats computation helper` | `scripts/compute_all_norm_stats.sh` |
| 3 | `test: add integration tests for spacecil end-to-end flow` | `spacecil/*_test.py` |

---

## Success Criteria

### Verification Commands
```bash
# All existing + new tests pass
JAX_PLATFORMS=cpu uv run pytest src/openpi/research/ -x -q
# Expected: ≥300 passed

# Debug config end-to-end (CPU, no real data)
JAX_PLATFORMS=cpu uv run python scripts/train_spacecil.py \
    --config spacecil_debug \
    --task-sequence debug_task \
    --num-steps-per-task 2 \
    --enable-distillation false \
    --exp-name test_e2e
# Expected: exit code 0, prints metrics summary

# All baseline configs resolve
JAX_PLATFORMS=cpu uv run python -c "
from openpi.training import config as _config
for name in ['spacecil_debug', 'spacecil_rm75_payload', 'spacecil_rm75_payload_fulltune', 'spacecil_rm75_payload_nodistill']:
    c = _config.get_config(name)
    print(f'OK: {name}')
"
# Expected: all print OK

# No NotImplementedError remaining
grep -r "NotImplementedError" scripts/train_spacecil.py
# Expected: no output

# No TODO placeholders remaining in main()
grep -n "TODO" scripts/train_spacecil.py
# Expected: no output
```

### Final Checklist
- [ ] All "Must Have" items present and verified
- [ ] All "Must NOT Have" patterns absent from codebase
- [ ] All tests pass (≥300)
- [ ] `train_spacecil.py` runs end-to-end on debug config
- [ ] All baseline configs resolve without error
- [ ] Data conversion script produces valid LeRobot dataset
- [ ] Scorer validation script runs on test episodes
