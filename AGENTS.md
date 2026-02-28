# AGENTS.md — SpaceCIL + LunarCompose on openpi

## Project Overview

This project builds research infrastructure for two academic papers on top of the **openpi** VLA (Vision-Language-Action) codebase:

- **Paper A: SpaceCIL** — Continual skill acquisition on a real mobile manipulation platform for expanding lunar-analog operations. Studies whether a released pi0.5 VLA backbone can be continually specialized as new operational tasks arrive sequentially.
- **Paper B: LunarCompose** — Factorized task × environment adaptation for lunar-analog compositional generalization. Studies whether factorized adaptation improves generalization to unseen task-environment combinations.

**Paper B depends on Paper A infrastructure.** Build order: Phase A (shared infra) → Phase B (SpaceCIL core) → Phase C (LunarCompose extension).

### Platform
- Wheeled base + RM75 7-DoF arm + two-finger gripper
- Primary policy camera: wrist RGB (hand-eye calibrated)
- Action space: Delta EE + gripper
- Backbone: released pi0.5 via openpi flow-matching policy path

## Environment

### Python Environment
```bash
# Activate the existing uv virtual environment
source .venv/bin/activate

# Or run scripts directly via uv
uv run <python_script>
uv run scripts/train.py <config_name>
```

- **Runtime**: Python 3.11+ via [uv](https://docs.astral.sh/uv/)
- **venv location**: `.venv/` (already created, configured, tested per openpi README)
- **Framework**: JAX (primary training), PyTorch (alternative path)
- **Package manager**: `uv sync` for dependency management
- **GPU memory**: Set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` before training

### Key Commands
```bash
# Training (JAX)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <config_name> --exp-name=<name>

# Compute norm stats (required before first training)
uv run scripts/compute_norm_stats.py --config-name <config_name>

# Run tests
uv run pytest src/ scripts/ -x -q

# Serve policy for inference
uv run scripts/serve_policy.py policy:checkpoint --policy.config=<config> --policy.dir=<checkpoint_dir>
```

## Repository Structure

### openpi Core (DO NOT modify unless necessary)
```
src/openpi/
├── models/          # JAX model definitions (pi0, pi0_fast, pi05, gemma, siglip, lora, vit)
├── models_pytorch/  # PyTorch implementations
├── policies/        # Policy wrappers (aloha, droid, libero, policy_config)
├── training/        # Training pipeline (config.py, data_loader, optimizer, checkpoints)
│   └── misc/        # External config modules (polaris_config, roboarena_config)
├── transforms.py    # Data transform system (Group, RepackTransform, DeltaActions, etc.)
├── shared/          # Utilities (download, normalize)
└── serving/         # Policy server
scripts/             # train.py, serve_policy.py, compute_norm_stats.py
examples/            # Per-robot examples (aloha_real, aloha_sim, droid, libero, ur5)
```

### Research Extension (OUR CODE)
```
src/openpi/research/                 # Research code — inside openpi package for clean imports
├── __init__.py
├── shared/                          # Shared infrastructure for both papers
│   ├── __init__.py
│   ├── episode_schema.py            # Unified episode schema (obs, action, meta fields)
│   ├── episode_schema_test.py       # Co-located test
│   ├── action_transforms.py         # Delta EE + gripper action transform layer
│   ├── action_transforms_test.py
│   ├── rm75_policy.py               # RM75 DataConfigFactory + Inputs/Outputs (like libero_policy.py)
│   ├── rm75_policy_test.py
│   ├── scorer_base.py               # Base scorer interface + per-task scorers
│   └── scorer_base_test.py
├── spacecil/                        # Paper A: SpaceCIL modules
│   ├── __init__.py
│   ├── config.py                    # get_spacecil_configs() → list[TrainConfig]
│   ├── task_adapter_bank.py         # Per-task PEFT module registry
│   ├── task_adapter_bank_test.py
│   ├── router.py                    # Language-visual router (no oracle task ID)
│   ├── router_test.py
│   ├── behavior_distillation.py     # Anti-forgetting via teacher snapshot
│   ├── behavior_distillation_test.py
│   ├── continual_harness.py         # Sequential task training + backward transfer eval
│   ├── continual_harness_test.py
│   ├── metrics.py                   # Mission-aware forgetting metrics
│   └── metrics_test.py
└── lunarcompose/                    # Paper B: LunarCompose modules
    ├── __init__.py
    ├── config.py                    # get_lunarcompose_configs() → list[TrainConfig]
    ├── env_adapter_bank.py          # Visual-path environment adapters
    ├── env_adapter_bank_test.py
    ├── dual_head_router.py          # Separate task + environment routing heads
    ├── dual_head_router_test.py
    ├── missing_corner_harness.py    # Train subset, eval unseen task-env combos
    ├── missing_corner_harness_test.py
    ├── factorization_diagnostics.py # Task-env entanglement analysis
    └── factorization_diagnostics_test.py

scripts/
├── train_spacecil.py                # Custom continual training script (imports openpi functions)
└── train_lunarcompose.py            # Custom factorized training script

projects/                            # Plans and documentation (NOT source code)
├── PLAN.md                          # Master plan overview
├── shared/
│   └── PLAN.md                      # Shared infrastructure plan
├── spacecil/
│   └── PLAN.md                      # SpaceCIL detailed plan
└── lunarcompose/
    └── PLAN.md                      # LunarCompose detailed plan
```

## Architecture Decisions

### 1. Code Placement: `src/openpi/research/`
Research code lives inside the openpi package (not a separate `projects/` source tree) because:
- Standard Python imports: `from openpi.research.spacecil import task_adapter_bank`
- Automatic inclusion in `uv sync` / editable install
- `conftest.py` coverage for JAX CPU backend fallback
- No `sys.path` hacks needed

### 2. Config Integration: Polaris Pattern
New configs are registered via the **polaris_config.py pattern** — one-line splat into `_CONFIGS`:
```python
# In src/openpi/training/config.py, add at end of _CONFIGS list:
*spacecil_config.get_spacecil_configs(),
*lunarcompose_config.get_lunarcompose_configs(),
```
Config names MUST be globally unique. Namespace: `spacecil_*`, `lunarcompose_*`.

### 3. PEFT/LoRA: Wrap openpi's LoRA
Task adapter bank wraps openpi's existing LoRA infrastructure (`models/lora.py`):
- Each task adapter = separate LoRA parameter set (`nnx.State` filtered to `.*lora.*`)
- Bank = dictionary `{task_id: lora_state_dict}`
- Swap via `nnx.update(model, lora_state)` before forward passes
- `freeze_filter` already supports frozen-old/train-new semantics

### 4. Training Loop: Custom Scripts (Not Forks)
Custom training scripts (`scripts/train_spacecil.py`, `scripts/train_lunarcompose.py`) that:
- Import openpi's `init_train_state()`, `train_step()` from `scripts/train.py`
- Add custom outer loops (task sequencing, distillation, router training)
- Keep JIT-compiled `train_step` for standard gradient computation
- Adapter swapping happens OUTSIDE JIT boundaries (avoid recompilation)

### 5. Data Pipeline: Work Within openpi Transforms
- "Episode schema" → implemented as `RepackTransform`
- "Action transforms" → implemented as `DataTransformFn` (like `DeltaActions`)
- "RM75 policy" → `RM75Inputs` + `RM75Outputs` + `LeRobotRM75DataConfig` (like `libero_policy.py`)

### 6. Tests: Co-located `*_test.py`
Following openpi convention:
- Tests live next to source as `*_test.py`
- `conftest.py` at `src/openpi/conftest.py` handles JAX CPU fallback
- Use `debug` config pattern (`paligemma_variant="dummy"`) for fast test models
- pytest test paths: `["src", "scripts", "packages"]`

## Conventions

### Naming
- Config names: `spacecil_<task>`, `lunarcompose_<task>_<env>`
- Module files: `snake_case.py`
- Test files: `<module>_test.py` (co-located)
- Classes: `PascalCase`
- Functions: `snake_case`

### Anti-Patterns (NEVER do these)
- Do NOT rewrite openpi core from scratch — reuse first, patch minimally
- Do NOT ask an agent to "implement SpaceCIL" as one task — always decompose into verifiable modules
- Do NOT start Paper B before Paper A infrastructure is stable
- Do NOT maintain duplicate schemas for different papers
- Do NOT maintain two inconsistent action converters
- Do NOT suppress type errors with `as any`, `@ts-ignore`, `@ts-expect-error`
- Do NOT modify `_CONFIGS` list beyond adding splat entries for new config modules

### Module Classification
For every module, classify as one of:
| Classification | Meaning |
|---|---|
| **reuse as-is** | Use openpi module directly without changes |
| **patch lightly** | Minimal modification to openpi core (e.g., adding config splat) |
| **new module** | New code in `src/openpi/research/` |

### Build Order Gates
| Gate | Requirement | Before |
|---|---|---|
| **G1** | One task trains and replays correctly; wrist-camera policy works; scorer matches manual labels | Any continual-learning experiments |
| **G2** | ≥2 tasks train sequentially; router beats random; distillation executes stably; adapter checkpoint restore works | Paper A mainline claims |
| **G3** | Env metadata enforced without leakage; missing-corner split verified; env adaptation path exists | Paper B mainline claims |

## Reference Documents
- `customized_docs/Research_Idea_Blueprint_SpaceCIL_LunarCompose.md` — Scientific problem, claims, evaluation logic
- `customized_docs/Implementation_Masterplan_SpaceCIL_LunarCompose.md` — Repo strategy, module list, build order, doctrines

## Version Control

### Git Remotes
| Remote | URL | Purpose |
|--------|-----|---------|
| `origin` | `git@github.com:Physical-Intelligence/openpi.git` | Upstream openpi (READ-ONLY — never push here) |
| `lunarbot` | `git@github.com:DsslRobot/openpi-lunarbot.git` | Our research fork (push all work here) |

### Branch Strategy
- **`lunarbot-research`** — Main working branch, tracked on `lunarbot` remote
- All research work happens on this branch (or feature branches off it)
- `main` tracks upstream openpi — do NOT commit research code to `main`

### Git Workflow (MANDATORY for all agents)

**Before starting any implementation work:**
```bash
git status                    # Check for uncommitted changes
git stash                     # Stash if needed before switching context
```

**After completing a logical unit of work:**
```bash
git add <changed files>       # Stage specific files (never blind `git add .`)
git commit -m "type: description"  # Commit with conventional message
git push lunarbot lunarbot-research # Push to our fork
```

**Commit Message Format:**
```
feat: add episode schema dataclass with obs/action/meta fields
fix: correct delta action normalization bounds
test: add scorer_base protocol compliance tests
refactor: extract common transform logic to shared/
docs: update PLAN.md with revised build order
chore: update .gitignore for checkpoint dirs
```

**Commit Granularity:**
- One commit per logical unit (one module, one bug fix, one test suite)
- NEVER commit broken code — all tests must pass before committing
- NEVER commit large unrelated changes in a single commit
- Commit early and often — small, traceable, reversible changes

### Safety Rules
- **NEVER push to `origin`** (upstream Physical-Intelligence/openpi)
- **ALWAYS push to `lunarbot`** remote only
- **NEVER force-push** unless explicitly requested by the user
- **NEVER commit secrets**, credentials, API keys, or `.env` files
- **NEVER commit large binary files** (checkpoints, datasets, model weights)
- Run `uv run pytest src/openpi/research/ -x -q` before every push
- If tests fail, fix before committing — do not push broken code

### Recovery
```bash
# View recent history
git log --oneline -10

# Undo last commit (keep changes staged)
git reset --soft HEAD~1

# Revert a specific commit (safe, creates new commit)
git revert <commit-hash>

# See what changed in a file
git log --oneline -5 -- <filepath>
git diff HEAD~1 -- <filepath>
```

### .gitignore Essentials (already configured)
```
assets/               # Downloaded model assets
checkpoints/          # Training checkpoints
data/                 # Local datasets
wandb/                # W&B logs
.opencode/            # OpenCode local config
.venv/                # Virtual environment
__pycache__/          # Python bytecode
```
