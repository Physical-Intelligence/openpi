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

**Last Updated:** 2026-03-01T01:55:00+08:00  
**Updated By:** Sisyphus (opencode session)  
**Git State:** branch `lunarbot-research` at commit `10a0a5b` — clean working tree  
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

The code infrastructure is fully built. The project now transitions from **infrastructure building** to **real-robot experimentation**. The user wants to start with Paper A (SpaceCIL).

### Immediate Next Actions (Paper A: SpaceCIL)

1. **Hardware setup** — Set up RM75 arm + gripper + wrist camera, calibrate hand-eye, document workspace for 4 tasks
2. **Data collection** — Teleoperate 50-100 demos per task (payload, latch, clean, connector) in nominal environment
3. **Data conversion script** — Write a script to convert raw teleop recordings into LeRobot format using our episode schema (this is the ONE piece of glue code not yet written — depends on the user's teleop recording format)
4. **Scorer validation** — Run scorers on 20+ manually-labeled pilot episodes, check precision/recall
5. **Single-task training (G1 real)** — `uv run scripts/train.py spacecil_rm75_payload` with real data
6. **Fill in `make_train_fn()`** — The `train_spacecil.py` skeleton has a `NotImplementedError` in the train_fn body that needs to be completed for real GPU training
7. **Continual training (G2 real)** — Run full task sequence with adapter bank, distillation, router

### Blocked On (user action required)
- Teleop recording format (ROS bags? HDF5? CSV + images?) — needed to write conversion script
- Physical robot access for data collection
- GPU access for training (JAX crashes on GPU init on current machine — may need different hardware)

### Future (after Paper A)
- Paper B (LunarCompose) experiments: set up 3 environment conditions, collect data per cell, run missing-corner protocol
- Paper writing for both papers

---

## 3. Active Context (Decisions & Gotchas)

### Environment Issues
- **JAX GPU problem on this machine**: pynvml sees 2 GPUs but JAX crashes on GPU init. All tests run with `JAX_PLATFORMS=cpu`. Training will need a machine where JAX GPU works.
- **Python env**: `.venv/` via uv, Python 3.11+. Use `source .venv/bin/activate` or `uv run`.

### Key Architectural Decisions Made
- Action space is **Absolute Joint Position (7 DoF) + Gripper (1 DoF) = 8D** (corrected from an early Delta EE mistake)
- Adapter swapping happens **outside JIT** to avoid recompilation
- One `nnx.State` filtered to `.*lora.*` is the single source of truth for adapter weights
- Env adapters target `.*siglip.*` (vision encoder) specifically
- Training scripts import openpi's `init_train_state()` and `train_step()` directly — no forking
- Configs use re-entrancy guard pattern to handle circular imports during `_CONFIGS` construction
- Scorers are heuristic proxies (joint displacement, gripper state) — need validation against manual labels before trusting results

### Things NOT to Forget
- `compute_norm_stats.py` must run on real dataset before training
- Config `repo_id` fields are placeholders (`placeholder/spacecil_*`) — must be updated to real LeRobot dataset paths
- The `openpi/training/config.py` has been modified (lines ~561-572) with splat entries for our configs
- Only one openpi core file was modified: `src/openpi/training/config.py`

---

## 4. Key Facts (Stable Reference)

### Git
- **Remote**: `lunarbot` → `git@github.com:DsslRobot/openpi-lunarbot.git`
- **Branch**: `lunarbot-research`
- **NEVER push to `origin`** (upstream Physical-Intelligence/openpi)

### Commit History
```
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
- SpaceCIL: `spacecil_rm75_payload`, `spacecil_rm75_latch`, `spacecil_rm75_clean`, `spacecil_rm75_connector`, `spacecil_debug`
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
