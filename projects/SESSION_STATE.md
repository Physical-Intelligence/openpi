# SESSION STATE — SpaceCIL + LunarCompose
<!-- 
  THIS FILE IS THE PROJECT'S PERSISTENT MEMORY.
  
  PROTOCOL:
  - Every new opencode session MUST read this file FIRST (AGENTS.md points here).
  - Before ending a session or when the user asks to "save state", UPDATE this file
    with what was accomplished and what comes next.
  - Keep it factual, concise, and actionable — this is NOT a diary, it's a state machine.
  - The "Last Updated" timestamp and session info below should be refreshed each update.
  
  FORMAT RULES:
  - Section 1 (Progress) is cumulative — only append, never delete completed items.
  - Section 2 (Next Steps) is replaced each update with the current frontier.
  - Section 3 (Active Context) captures decisions and gotchas from the latest session.
  - Section 4 (Key Facts) is stable reference data — rarely changes.
-->

**Last Updated:** 2026-03-01T16:00:00+08:00  
**Updated By:** Atlas Orchestrator (opencode session)  
**Git State:** branch `lunarbot-research` at commit `84f340f` + uncommitted quality fixes — untracked: `.sisyphus/`, `opencode.json`  
**Tests:** 308 passing (`JAX_PLATFORMS=cpu uv run pytest src/openpi/research/ -x -q`)

---

## 1. Progress Tracker

### Phase A — Shared Infrastructure ✅ COMPLETE
| Module | Status | Tests | Commit |
|--------|--------|-------|--------|
| `shared/episode_schema.py` | ✅ Done | 19 | `06e0e85` |
| `shared/action_transforms.py` | ✅ Done | 22 | `06e0e85` |
| `shared/rm75_policy.py` | ✅ Done | 10 | `e8c73d0` |
| `shared/scorer_base.py` | ✅ Done | 18 | `e8c73d0` |
| **Gate G1** | ✅ Passed (code-level) | | |

### Phase B — SpaceCIL Core ✅ COMPLETE
| Module | Status | Tests | Commit |
|--------|--------|-------|--------|
| `spacecil/metrics.py` | ✅ Done | 35 | `ed29e7e` |
| `spacecil/task_adapter_bank.py` | ✅ Done | 20 | `ed29e7e` |
| `spacecil/behavior_distillation.py` | ✅ Done | 20 | `ed29e7e` |
| `spacecil/router.py` | ✅ Done | 28 | `ed29e7e` |
| `spacecil/config.py` | ✅ Done (22 configs) | 8 | `dd391a6` |
| `spacecil/continual_harness.py` | ✅ Done | 24 | `ed29e7e` |
| `scripts/train_spacecil.py` | ✅ Done (fully functional) | CLI + 22 integration | `84f340f` |
| **Gate G2** | ✅ Passed (code-level) | | |

### Phase C — LunarCompose Extension ✅ COMPLETE
| Module | Status | Tests | Commit |
|--------|--------|-------|--------|
| `lunarcompose/env_adapter_bank.py` | ✅ Done | 12 | `cf3f93c` |
| `lunarcompose/dual_head_router.py` | ✅ Done | 9 | `cf3f93c` |
| `lunarcompose/missing_corner_harness.py` | ✅ Done | 41 | `cf3f93c` |
| `lunarcompose/factorization_diagnostics.py` | ✅ Done | 12 | `cf3f93c` |
| `lunarcompose/config.py` | ✅ Done | 9 | `cf3f93c` |
| `scripts/train_lunarcompose.py` | ✅ Done (skeleton) | CLI ✅ | `cf3f93c` |
| **Gate G3** | ✅ Passed (code-level) | | |

### Documentation ✅ COMPLETE
| Document | Lines | Commit |
|----------|-------|--------|
| `customized_docs/API_Reference.md` | 2,675 | `113a53b` |
| `customized_docs/Developer_Guide.md` | 1,213 | `113a53b` |
| `customized_docs/Experiment_Guide_SpaceCIL.md` | 1,230 | `113a53b` |
| `customized_docs/Experiment_Guide_LunarCompose.md` | 1,056 | `113a53b` |
| `customized_docs/Research_Architecture_SpaceCIL_LunarCompose.md` | 1,013 | `113a53b` |

### SpaceCIL Experimentation Phase ✅ COMPLETE
| Task | Description | Status | Commit |
|------|-------------|--------|--------|
| T1 | Fix imports + configurable repo_id | ✅ Done | `61cbb58` |
| T2 | Implement make_train_fn body | ✅ Done | `b5cdbf5` |
| T3 | HDF5→LeRobot conversion script | ✅ Done | `d0041df` |
| T4 | Scorer validation utility | ✅ Done | `5f4c4bd` |
| T5+T6 | Wire main() + scorers + eval episodes | ✅ Done | `63a752e` |
| T7 | Baseline config variants (22 total) | ✅ Done | `dd391a6` |
| T8 | Distillation integration | ✅ Done | `03de6ae` |
| T9 | CLI routing flags | ✅ Done | `e437b34` |
| T10 | Metrics + adapter bank save | ✅ Done | `e437b34` |
| T11 | Norm stats helper script | ✅ Done | `82013fe` |
| T12 | Integration tests (22 tests) | ✅ Done | `84f340f` |
| F1-F4 | Final verification wave | ✅ ALL APPROVE | — |

**New files created:**
- `scripts/convert_rm75_data_to_lerobot.py` (397 lines) — HDF5→LeRobot conversion
- `scripts/validate_scorers.py` (109 lines) — Scorer precision/recall validation
- `scripts/compute_all_norm_stats.sh` (19 lines) — Norm stats helper for all 4 tasks
- `src/openpi/research/spacecil/integration_test.py` (518 lines) — 22 integration tests

**Key technical discoveries:**
- LeRobot v0.1.0 forbids `/` in feature key names → dot-notation (`observation.wrist_image`)
- RepackTransform maps dots→slashes at pipeline boundary
- `make_train_fn` signature: 7 params (config, train_state, state_sharding, mesh, rng, num_steps_per_task, *, distillation)
- Distillation monitoring approach (not full gradient integration) — pragmatic deferral
---

## 2. Next Steps (Current Frontier)

### SpaceCIL Experimentation Plan ✅ FULLY COMPLETE
All 12 implementation tasks + 4 final verification tasks are done. The plan file (`.sisyphus/plans/spacecil-experimentation.md`) has all 28 checkboxes marked `[x]`.

### What's Next: Paper B (LunarCompose) Experimentation
The same treatment that was applied to SpaceCIL needs to be applied to LunarCompose:
- Generate a `spacecil-experimentation`-style plan for LunarCompose
- Complete `scripts/train_lunarcompose.py` (currently skeleton)
- Wire missing-corner harness, dual-head router, env adapter bank
- Add factorization diagnostics and evaluation

### Alternatively: Real Robot Data Collection
SpaceCIL is code-complete and ready for real experiments. The remaining dependency is hardware:
- Physical robot access for data collection
- GPU machine setup for training
- Teleop recording in HDF5 format → conversion via `scripts/convert_rm75_data_to_lerobot.py`
- Run `scripts/compute_all_norm_stats.sh` on real data before training
- Execute: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run python scripts/train_spacecil.py --config spacecil_rm75_payload --task-sequence payload latch clean connector --num-steps-per-task 10000`

### Blocked On (user action required)
- Physical robot access for data collection
- GPU machine setup with working JAX
- Decision: proceed with LunarCompose plan or wait for robot data?
---

### Experimentation Plan Details
- **Plan generated**: `.sisyphus/plans/spacecil-experimentation.md`
- **Metis review**: Identified 6 unasked questions, 7 guardrails, 4 scope creep risks — all addressed in plan
- **Momus review**: **OKAY** — all 12 tasks verified, file references confirmed, minor API naming fixes applied
- **Minor fixes applied post-Momus**: `TeacherSnapshot.update()` → `.snapshot()`, `TeacherSnapshot.predict()` → `.get_params()`, operational_forgetting weights type dict → ndarray

### Key Architectural Decisions Made (Cumulative)
- Action space is **Absolute Joint Position (7 DoF) + Gripper (1 DoF) = 8D** (corrected from an early Delta EE mistake)
- Adapter swapping happens **outside JIT** to avoid recompilation
- One `nnx.State` filtered to `.*lora.*` is the single source of truth for adapter weights
- Env adapters target `.*siglip.*` (vision encoder) specifically
- Training scripts import openpi's `init_train_state()` and `train_step()` directly — no forking
- Configs use re-entrancy guard pattern to handle circular imports during `_CONFIGS` construction
- Scorers are heuristic proxies (joint displacement, gripper state) — need validation against manual labels before trusting results
- **NEW**: `init_train_state` must be called ONCE at start; LoRA swapped per task; optimizer state re-initialized per task (fresh Adam moments)
- **NEW**: Router training is a separate concern — NOT in `make_train_fn`. Only oracle/random routing for initial experiments.
- **NEW**: Distillation wraps AROUND `train_step` — custom `train_step_with_distillation`, does NOT modify upstream `train_step`
- **NEW**: Baseline configs use `dataclasses.replace` on existing configs — never full standalone definitions
- **NEW**: Data format is HDF5 per-episode files → LeRobot via `LeRobotDataset.create()` API

### Environment Notes
- **JAX GPU problem on this machine**: pynvml sees 2 GPUs but JAX crashes on GPU init. All tests run with `JAX_PLATFORMS=cpu`. User confirmed separate GPU machine available.
- **Python env**: `.venv/` via uv, Python 3.11+. Use `source .venv/bin/activate` or `uv run`.

### Things NOT to Forget
- `compute_norm_stats.py` must run on real dataset before training (all 4 tasks pre-computed)
- Config `repo_id` fields use `SPACECIL_DATA_PREFIX` env var with `placeholder` fallback — set before training with real data
- The `openpi/training/config.py` has been modified (lines ~561-572) with splat entries for our configs
- Only one openpi core file was modified: `src/openpi/training/config.py`
- `.sisyphus/` directory is untracked (gitignore it or commit selectively)

---

## 4. Key Facts (Stable Reference)

### Git
- **Remote**: `lunarbot` → `git@github.com:DsslRobot/openpi-lunarbot.git`
- **Branch**: `lunarbot-research`
- **NEVER push to `origin`** (upstream Physical-Intelligence/openpi)

### Commit History
```
84f340f test: add end-to-end integration tests for spacecil training flow
97d48a0 fix: remove duplicate fields and closing paren in train_spacecil.py
e437b34 feat: add oracle/random routing CLI flags
03de6ae feat: integrate distillation loss into spacecil training loop
82013fe feat: add norm stats computation helper for all SpaceCIL tasks
63a752e feat: wire main() execution flow, scorers, and eval episodes
dd391a6 feat: add baseline config variants for SpaceCIL experiments
b5cdbf5 feat: implement make_train_fn body for spacecil training loop
86b70c8 fix: update RepackTransform to map LeRobot dot-keys to pipeline slash-keys
d0041df feat: add RM75 HDF5 to LeRobot data conversion script
5f4c4bd feat: add scorer validation utility script
61cbb58 fix: add missing imports and configurable repo_id to spacecil config
91919fa chore: update session state with experimentation plan (Momus-approved)
6b17de2 chore: update session state
10a0a5b feat: add persistent session memory system for cross-session continuity
113a53b docs: comprehensive documentation for SpaceCIL + LunarCompose
cf3f93c feat: implement Phase C LunarCompose extension modules (285 tests)
ed29e7e feat: implement Phase B SpaceCIL core modules (206 tests)
e8c73d0 feat: implement rm75_policy and scorer_base with full test suites
1955f87 fix: correct action space from Delta EE to Absolute Joint Position
06e0e85 feat: implement episode_schema and action_transforms with full test suites
```

### Test Command
```bash
JAX_PLATFORMS=cpu uv run pytest src/openpi/research/ -x -q
```

### Config Names
- SpaceCIL base: `spacecil_rm75_payload`, `spacecil_rm75_latch`, `spacecil_rm75_clean`, `spacecil_rm75_connector`, `spacecil_debug`
- SpaceCIL baselines (22 total): `spacecil_rm75_{task}_fulltune`, `spacecil_rm75_{task}_nodistill`, `spacecil_rm75_shared_lora`, `spacecil_rm75_{task}_oracle`, `spacecil_rm75_{task}_random`
- LunarCompose: `lunarcompose_{task}_{env}` (12 cells), `lunarcompose_factorized`, `lunarcompose_monolithic`, `lunarcompose_debug`

### Documentation Map
| Document | Purpose |
|----------|---------|
| `AGENTS.md` | Bootstrap: architecture, conventions, git rules (read by opencode at start) |
| `projects/SESSION_STATE.md` | **THIS FILE**: persistent memory across sessions |
| `projects/PLAN.md` | Master plan overview (217 lines) |
| `projects/shared/PLAN.md` | Phase A detailed plan |
| `projects/spacecil/PLAN.md` | Phase B detailed plan (640 lines) |
| `projects/lunarcompose/PLAN.md` | Phase C detailed plan (606 lines) |
| `customized_docs/Research_Idea_Blueprint_*.md` | Scientific problem, claims, evaluation logic |
| `customized_docs/Implementation_Masterplan_*.md` | Engineering strategy, build order, doctrines |
| `customized_docs/API_Reference.md` | Full API reference for all 15 research modules |
| `customized_docs/Developer_Guide.md` | Architecture, data pipeline, integration guide |
| `customized_docs/Experiment_Guide_SpaceCIL.md` | Paper A end-to-end experiment recipe |
| `customized_docs/Experiment_Guide_LunarCompose.md` | Paper B end-to-end experiment recipe |
| `customized_docs/Research_Architecture_*.md` | System design with architecture diagrams |
| `.sisyphus/plans/spacecil-experimentation.md` | **Momus-approved** experimentation plan (1,306 lines, 12 tasks + 4 verification) |

### Explicit Constraints (Verbatim from User)
- "Do not modify existing openpi transform base classes"
- "The correct action space is Absolute Joint Position (7 DoF) + Gripper (1 DoF) = Total 8D"
- "No coding agent should rewrite the full stack from scratch unless a minimal patch path has already failed"
- "Never ask an agent to 'implement SpaceCIL' as one task — always decompose into verifiable modules"
- "NEVER push to `origin`"
- "Do not rewrite openpi core training logic. Import init_train_state and train_step directly."
- "Do not swap adapters inside JIT-compiled functions"
- "Do not maintain two LoRA state representations"
- "Do not hardcode operational weights inside metrics.py"
