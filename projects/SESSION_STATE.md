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

**Last Updated:** 2026-03-01T01:45:00+08:00  
**Updated By:** Sisyphus (opencode session)  
**Git State:** branch `lunarbot-research` at commit `6b17de2` — untracked: `.sisyphus/`, `opencode.json`  
**Tests:** 285 passing (`JAX_PLATFORMS=cpu uv run pytest src/openpi/research/ -x -q`)

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
| `spacecil/config.py` | ✅ Done | 7 | `ed29e7e` |
| `spacecil/continual_harness.py` | ✅ Done | 24 | `ed29e7e` |
| `scripts/train_spacecil.py` | ✅ Done (skeleton) | CLI ✅ | `ed29e7e` |
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

---

## 2. Next Steps (Current Frontier)

The code infrastructure is fully built. A **detailed experimentation plan** has been generated, Metis-reviewed, and Momus-approved.

### Experimentation Plan
- **Plan file**: `.sisyphus/plans/spacecil-experimentation.md` (1,306 lines, Momus-approved OKAY)
- **Scope**: Paper A (SpaceCIL) only — 12 implementation tasks + 4 final verification tasks
- **Structure**: 4 parallel waves → critical path: T1 → T2 → T5 → T8 → T12
- **Key decisions made this session**:
  - Teleop format: HDF5 per-episode files (matches ALOHA/DROID patterns)
  - Router integration: DEFERRED — only oracle/random routing via CLI flags for initial experiments
  - Distillation: Optional code path (`--enable-distillation`, disabled by default)
  - Baselines: 5 variants via `dataclasses.replace` (fulltune, nodistill, shared_lora, oracle, random)
  - Model state: `init_train_state` called ONCE, LoRA swapped per task, optimizer reset per task

### To Start Execution
Run `/start-work` in the next session — the plan is ready for immediate execution.

### Wave Summary
| Wave | Tasks | Description |
|------|-------|-------------|
| 1 (parallel) | T1-T4 | Fix imports, implement `make_train_fn`, data conversion script, scorer validation utility |
| 2 (after W1) | T5-T8 | Wire main() flow, scorers/eval, baseline configs, distillation integration |
| 3 (after W2) | T9-T12 | CLI flags, metrics/save, norm stats helper, integration tests |
| Final | F1-F4 | Compliance audit, code quality, full QA, scope fidelity |

### 10 Blockers Identified and Addressed in Plan
1. `make_train_fn` NotImplementedError → Task 2
2. `repo_id` placeholders → Task 1 (env var configurable)
3. Scorers/eval_episodes empty → Task 6
4. Metrics/save commented out → Task 10
5. `run_sequence()` never called → Task 5
6. Baseline configs missing → Task 7
7. CLI routing flags missing → Task 9
8. Router not integrated → Deferred (oracle/random only via CLI)
9. Distillation loss not called → Task 8
10. Physical sensor augmentation → Out of scope (hardware)

### Blocked On (user action required)
- Physical robot access for data collection (robot "available soon" per user)
- GPU machine setup: User confirmed separate machine with working JAX GPU exists — need to set up data transfer workflow
- Teleop recording format: Decided HDF5 per-episode, but actual recorder software TBD

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
- Config `repo_id` fields are placeholders (`placeholder/spacecil_*`) — Task 1 in plan makes them configurable via `SPACECIL_DATA_PREFIX` env var
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
- SpaceCIL baselines (planned, not yet created): `spacecil_rm75_{task}_fulltune`, `spacecil_rm75_{task}_nodistill`, `spacecil_rm75_shared_lora`, `spacecil_rm75_{task}_oracle`, `spacecil_rm75_{task}_random`
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
