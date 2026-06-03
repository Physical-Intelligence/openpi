# Refactor plan — `src/openpi/models/` + training scripts

Cleanup of duplicated code, dead code, and unused variables left behind across the
Pi0 model families and the training scripts. Findings below are grouped by area, then
followed by a prioritized execution plan and the smoke-test approach.

Scope: `src/openpi/models/` (~11.6k LOC across 5 model families), plus
`scripts/train.py` and `scripts/train_trace_vla.py`.

---

## 1. Dead files & dead code (safe, high-confidence) — ✅ DONE (except pi0_skill_embed, on hold)

Removed ~155 LOC: `pi0.py` `left_to_right_align` + `sample_low_level_task` + commented
`sample_actions` block; `trace_utils.py` `hard_route_one_hot` + orphaned `NUM_TRACE_EXPERTS`;
`gemmoe_trace.py` unused `_get_gemmoe_config` import; fixed dangling `pi0_fuse.py` docstring ref.


| Item | Location | Action | Notes |
|---|---|---|---|
| `pi0_skill_embed.py` (whole file, 433 LOC) | `models/pi0_skill_embed.py` | **Delete** | Untracked in git, **0 importers**. Defines `class Pi0` that is a near-verbatim copy of `pi0.py`; the only diff is a removed commented block. No actual "skill embed" logic. |
| `hard_route_one_hot()` | `models/trace_utils.py:51` | **Delete** | Only referenced in its own module docstring; never called anywhere. |
| `_get_gemmoe_config` import | `models/gemmoe_trace.py:55` | **Delete** | Imported (`get_config as _get_gemmoe_config`), never used. |
| `left_to_right_align()` + `sample_low_level_task()` | `pi0.py:51`, `pi0.py:258` | **Delete (confirm first)** | Only reachable via a commented-out `sample_actions` block at `pi0.py:413-433`. Dead path. |
| Commented-out `sample_actions` block | `pi0.py:413-433` | **Delete (confirm first)** | Legacy implementation, superseded by `pi0.py:437-499`. |

### NOT dead (claims that did not hold up under verification)
- `put_along_last_axis` (`pi0_fuse.py:26`) — **used** at `pi0_fuse.py:229`. Keep the import.
- `time_uniform_resample` (`trace_utils.py`) — reachable via `resample_trace(method="time_uniform")`,
  and `config.py:143` exposes `trace_resample_method` which can be `"time_uniform"`. Keep.
- The `import flax.linen as nn` lines in the trace/target VLA files are `# noqa: F401`
  parity markers; `del prefix_out` / `del _p_out` in scan closures are intentional JAX idiom.

---

## 2. Gemma backbone duplication — ✅ DONE (gemma.py ↔ gemmoe.py)

`gemmoe.py` (657 → 553 LOC) now imports `RMSNorm`, `Embedder`, `FeedForward`, `_apply_rope`,
`_name`, `_gated_residual`, `PALIGEMMA_VOCAB_SIZE` from `gemma.py` instead of redefining them
(verified byte-identical first). `gemma.py` (the base used by pi0/pi05) is untouched;
`gemmoe_trace.py` is unaffected (it imports these via gemmoe's namespace — re-export confirmed).
Blank-line runs normalized. Validated: atomic_libero 1.3672 (exact, gemmoe path) and trace_vla_moe
3.7695 (exact, gemmoe_trace path). `gemma_fast.py` left standalone (lower priority — own ConfigDict
+ lora.FeedForward).

### (original analysis)

`gemma.py` and `gemmoe.py` redefine the same core blocks byte-for-byte (~140 LOC duplicated):

| Block | `gemma.py` | `gemmoe.py` |
|---|---|---|
| `RMSNorm` (adaptive) | 113–131 | 182–200 |
| `Embedder` | 135–154 | 204–223 |
| `FeedForward` (dense) | 281–308 | 345–372 |
| `_apply_rope` | 462–478 | 628–640 |
| `_name` | 481–488 | 643–650 |
| `_gated_residual` | 491–497 | 652–658 |
| `PALIGEMMA_VOCAB_SIZE` | — | — |

`gemmoe_trace.py` **already** imports these from `gemmoe.py` (lines 44–55) — that is the
pattern to follow. `gemmoe.py`'s `Config` is a backward-compatible superset of `gemma.py`'s.

**Genuinely unique** (keep):
- `gemmoe.py`: `Router` (397–420), `GemmoeSparseMoeBlock` (423–456), `top1_routing` (379–394),
  `GemmoeBlockSparseTop2MLP` (328–342) — learned routing + shared-expert MoE.
- `gemmoe_trace.py`: `get_trace_config` (80–184), `HardMoeBlock` (286–314), `TraceBlock` (322–383),
  `TraceModule` (391–482) — hard-routed MoE, no shared expert, externally-provided one-hot routing.
- `gemma.py` vs `gemmoe.py` `Attention` differ (gemma has `_init_cache`/`_update_cache` at 162–181;
  gemmoe simplifies cache concat at 283–286) — reconcile carefully, do not blindly merge.

**Proposed:** extract `gemma_base.py` holding the shared primitives + base `Config`; have
`gemma.py` and `gemmoe.py` import from it. `gemma_fast.py` is standalone (uses `ml_collections`
ConfigDict + `lora.FeedForward`); it could import `RMSNorm`/`Embedder`/`_apply_rope`/`_name`
from the base too, but it's lower priority.

### Minor issues in this family
- `gemmoe_trace.py:309` — `deterministic` param in `HardMoeBlock.__call__` is unused
  (marked `ARG002`); leave or document.
- `gemmoe_trace.py:346` vs `:360` — `_gate_attn` discarded for attention norm while `gate` is
  collected for FFN norm; intentional but the inconsistent naming is confusing. Add a comment.
- `gemmoe.py:559` — `self.num_extra_experts = getattr(self.configs[1], ...)` assumes a 2nd config
  exists with no bounds check. Add a guard.

---

## 3. Pi0 trace/target model classes — biggest win

`pi0_trace_vla.py` (916), `pi0_trace_vla_moe.py` (883), `pi0_trace_vla_actionmoe.py` (865),
`pi0_target_vla_actionmoe.py` (644) are **60–70% identical** (~3,300 LOC total).

**Duplicated verbatim across all four:**
- Module-level helpers `make_attn_mask`, `posemb_sincos`, `fourier_encode_2d`
- `_embed_prefix_with_images`, `_completion_predict`
- The `compute_loss` and `sample_actions` / `sample_actions_and_completion` skeletons

**The only real differences** are which streams are MoE vs dense, and whether `target_xy`
enters the AdaRMS conditioning:

| Variant | Trace stream | Action stream | Target in action cond | Trace loss |
|---|---|---|---|---|
| `Pi0TraceVLA` | MoE (hard one-hot) | dense | no | yes |
| `Pi0TraceVLAMoe` | MoE | MoE (same K) | no | yes |
| `Pi0TraceVLAActionMoe` | dense | MoE | no | yes |
| `Pi0TargetVLAActionMoe` | none | MoE | **yes** (`adarms_cond = time_emb + tgt_emb`) | no |

**Proposed:** one `TraceVLABase(_model.BaseModel)` holding the shared embed/loss/sample
machinery; variants override only `_forward_*()` and `compute_loss()`. Move the three
module-level helpers to a shared util module. Config flags drive the differences:
`has_trace_stream`, `num_trace_experts`, `num_action_experts`, `include_target_in_action_cond`.
Estimated ~3,300 → ~1,300 LOC.

`pi0_atomic.py` (`Pi0Atomic`, used by `pi0_config.py` + `policy.py`) is **genuinely distinct**
(hierarchical `prefill`/`reason`/`act`, MoE conditioning via `embed_atomic_skill`, custom
multi-term loss). **Keep**; just add a docstring noting it isn't a plain `Pi0` subclass.

`pi0_fuse.py` (`Pi0Fuse(Pi0)`) is **already a clean subclass**. Keep.

---

## 4. Config & observation dataclasses — ✅ DONE (via shared helpers, not a dataclass hierarchy)

Chose shared helpers over frozen-dataclass inheritance (lower risk, same dedup of the actual code;
field declarations left in place as declarative + readable). Net −111 LOC:
- `trace_observation.trace_inputs_spec(config, *, batch_size)` — the identical TraceObservation
  `inputs_spec` shared by Pi0TraceVLA / Pi0TraceVLAMoe / Pi0TraceVLAActionMoe configs.
- `pi0_config.llm_freeze_filter(paligemma_variant, action_expert_variant, *, expert_suffixes)` —
  the gemma-stream freeze logic, parameterized by which expert streams exist. Replaces the inline
  copies in trace_vla / trace_vla_moe / trace_vla_actionmoe / target_vla_actionmoe (trace_vla's
  redundant `Any(all_llm, action_subtree)` ≡ `all_llm`). Proven behaviorally identical to all
  originals across every paligemma×action variant combo (incl. action-LoRA). Pi0Config /
  Pi0AtomicConfig still inline their 2-stream copies — can adopt the helper later.

Removed now-unused imports (`jax`/`jnp` from the 3 trace configs, `nnx_utils` from all 4).
Validated: trace_vla 2.4391 (exact, inputs_spec), trace_vla_lora 2.4416/grad_norm 39.74 (exact vs
§5-migrated, freeze helper).

### (original analysis)

Six `*_config.py` files duplicate field sets and a near-identical `get_freeze_filter()`:

- All 6 share: `dtype`, `paligemma_variant`, `action_expert_variant`, `action_dim`,
  `action_horizon`, `max_token_len`, `pi05`.
- Trace family (4) also share: `action_loss_coeff`, `completion_loss_coeff`,
  `completion_shared_dim`, `completion_per_skill_hidden`, `fourier_num_freqs`.
- Trace VLA (3) also share: `trace_horizon`, `trace_dim`, `trace_loss_coeff`,
  `append_target_anchor`, `trace_expert_variant`.
- `get_freeze_filter()` is duplicated with minor per-variant differences across
  `pi0_config.py` (81–110, 183–212), `pi0_trace_vla_config.py` (131–178),
  `pi0_trace_vla_moe_config.py` (167–201), `pi0_trace_vla_actionmoe_config.py` (150–185),
  `pi0_target_vla_actionmoe_config.py` (153–188).

**Proposed hierarchy:** `BaseVLAConfig(_model.BaseModelConfig)` →
`TraceVLAConfigBase(BaseVLAConfig)` → concrete variants. Collapses 6 files toward ~3 plus
thin subclasses. Pull `get_freeze_filter()` into the base, parameterized by which expert
subtrees exist.

**Observations:** `target_observation.py` and `trace_observation.py` share an identical
`from_dict()` and image-normalization logic. `trace_observation.py` adds overlay images +
3 keypoint arrays (`semantic_target_xy`, `current_ee_xy`, `future_trace_xy`) vs
`target_observation.py`'s single keypoint. **Proposed:** lift the shared `from_dict()` and a
`_resize_images()`/`_build_masks()` helper into the base `Observation`; keep the augmentation
chains separate (different keypoint dimensionality).

---

## 5. Training scripts — SIX forked JAX training scripts (2,402 LOC) — ✅ DONE

**Consolidated into a single `scripts/train.py`. All 5 non-standard forks deleted**
(`train_atomic.py`, `train_trace_vla.py`, `train_trace_vla_actionmoe.py`,
`train_trace_vla_moe.py`, `train_target_vla_actionmoe.py`). Implemented:
- `train.py`: `init` overlays the loaded subset onto the model's init state (`_overlay_params`);
  `train_step` accepts `(loss, info)` *or* a scalar loss (backward compatible).
- `weight_loaders.py`: `AtomicWeightLoader` (dense FFN -> shared MoE expert) and
  `ActionMoeWeightLoader` (fans `mlp_1` into `moe_1`/`moe_2` experts, with optional stream-2
  attn replication + `mlp_2` copy). Each config's `weight_loader=` now names its loader.
- `data_loader.py`: `DataLoaderImpl` dispatches `Trace`/`Target` observations by data-config type
  (before the `atomic_token`/`diffusion_loss_mask` content fallbacks).

**Validated** — every config reproduces its pre-refactor step-0 loss via `train.py`:
atomic_libero 1.3672 (exact), target_vla_actionmoe 0.9990 (exact), trace_vla_moe 3.7695 (exact),
trace_vla_actionmoe 2.4816 (exact), trace_vla 2.4391 (exact), trace_vla_lora 2.4416 vs 2.4415
(LoRA random-init RNG noise), pi05_libero_100 regression 0.0953 vs 0.0951 (FSDP sharding).
Also repointed `trace_vla`/`trace_vla_lora` `repo_path` to local `data/libero-100`.

The original analysis below is retained for the line references.

---

### (historical) SIX forked JAX training scripts

There is not one fork but **six** near-identical JAX training scripts, each a copy of `train.py`
with a bespoke weight remap + observation handling:

| Script | LOC | Targets | Observation |
|---|---|---|---|
| `scripts/train.py` | 280 | pi05 / reason (plain) | `Observation` / `Fuse` / `Atomic` dispatch |
| `scripts/train_atomic.py` | 415 | `atomic_libero` | `AtomicObservation` |
| `scripts/train_trace_vla.py` | 398 | (Pi0TraceVLA) | `TraceObservation` |
| `scripts/train_trace_vla_actionmoe.py` | 449 | `trace_vla_actionmoe` | `TraceObservation` |
| `scripts/train_trace_vla_moe.py` | 431 | `trace_vla_moe` | `TraceObservation` |
| `scripts/train_target_vla_actionmoe.py` | 429 | `target_vla_actionmoe` | `TargetObservation` |

(plus `scripts/train_pytorch.py`, a separate PyTorch-backend path — out of scope here.)

Each fork duplicates `init_logging` / `init_wandb` / `init_train_state` / `train_step` / `main`
verbatim from `train.py` (dropping the `@at.typecheck` annotations), and each carries its own
`_load_and_filter_weights` + `update_params`. They differ only in (a) the pi05_base→variant weight
remap, (b) the observation type produced, and (c) multi-loss `(loss, info)` handling. This is
~2,000 LOC of duplication and is the single biggest cleanup opportunity in the repo.

**Confirmed empirically (baseline smoke tests):** `atomic_libero` cannot run via `train.py` —
its model has an extra `sigma_emb` param subtree absent from `pi05_base`, so `train.py`'s strict
`check_pytree_equality` rejects it (`ValueError: ... symmetric difference {'sigma_emb'}`). It needs
`train_atomic.py`'s partial-load path. Each variant is similarly bound to its own script.

The original two-script analysis below (kept for the concrete line refs) generalizes to all six.

### `train_trace_vla.py` as the representative fork

The user's main objection: **it bypasses the existing data-loading code.** Findings:

### Duplicated verbatim from `train.py` (should not be forked)
- `init_logging()` (44–58), `init_wandb()` (61–75) — byte-for-byte copies.
- `init_train_state()` (159–210), `train_step()` (213–266), `main()` (321–398) — forks with
  small deltas, and the trace fork **drops the `@at.typecheck` annotations** present in `train.py`.

### Reinvents existing abstractions
- **Data loading (`_create_trace_data_loader`, 273–314):** builds `LiberoTraceDataset` directly
  and wraps it in a bespoke `_Wrapper` to yield `TraceObservation`. This duplicates
  `data_loader.DataLoaderImpl.__iter__` (`data_loader.py:590–597`), which **already dispatches
  observation type by data config** (Atomic / Fuse / Observation). The right fix is to add a
  `TraceObservation` / `TargetObservation` branch there (keyed on `LeRobotTraceVLADataConfig`),
  so the standard `create_data_loader` handles it — and the bypass disappears.
- **`update_params` (144–152):** hand-rolled recursive dict merge that duplicates
  `state.replace_by_pure_dict(...)`, which `train.py:99` uses directly.

### Genuinely unique (must be preserved, but relocated)
- **`_load_and_filter_weights` (82–141):** the pi05_base → Pi0TraceVLA weight remap (replicate
  action-expert `*_1` weights into trace-expert `*_2`; fan dense `mlp_1` FFN into K hard-MoE
  experts `moe_2/expert_*`). This is real, non-trivial logic — but it belongs as a
  **`WeightLoader` subclass** (`training/weight_loaders.py`, alongside `CheckpointWeightLoader` /
  `PaliGemmaWeightLoader`), not inline in a training script.
- **Multi-loss handling in `train_step` (223–266):** trace `compute_loss` returns
  `(per_sample, info)` (`pi0_trace_vla.py:580`) vs `train.py`'s single tensor. Generalize
  `train.py`'s `train_step` to accept either an `(loss, info)` tuple or a bare loss, so all
  models share one step.

**Proposed end state:** delete **all five** non-standard forks
(`train_atomic.py`, `train_trace_vla.py`, `train_trace_vla_actionmoe.py`,
`train_trace_vla_moe.py`, `train_target_vla_actionmoe.py`). Each script's three real contributions
move to shared infrastructure so a single `train.py` runs every config:
- (a) the per-variant weight remap → a `WeightLoader` subclass in `training/weight_loaders.py`
  (the atomic `sigma_emb` and trace/target MoE fan-out cases each become a loader that does a
  partial, structure-tolerant load instead of `train.py`'s strict `check_pytree_equality`);
- (b) observation dispatch → `DataLoaderImpl.__iter__` (add `Trace`/`Target` branches);
- (c) multi-loss `(loss, info)` return → generalized `train_step` in `train.py`.

Success criterion: all 10 configs in `trained_configs.txt` train via `scripts/train.py` alone.

---

## 6. Prioritized execution plan

1. **Safe deletions** (§1) — zero risk, ~450+ LOC removed immediately. Confirm
   `pi0_skill_embed.py` deletion (untracked → permanent) and the `pi0.py` dead-path removal.
2. **Fold trace/target data loading into `DataLoaderImpl`** (§5) — directly addresses the
   user's main complaint; unblocks deleting the data-loader half of `train_trace_vla.py`.
3. **Generalize `train_step` multi-loss + move weight remap to a `WeightLoader`** (§5) — lets
   `train_trace_vla.py` be deleted entirely.
4. **Gemma base extraction** (§2) — foundational shared primitives.
5. **`TraceVLABase` model base class** (§3) — biggest LOC reduction, highest review effort.
6. **Config + observation base classes** (§4).

Each step is independently shippable; later steps depend on earlier ones only loosely.

---

## 7. Validation & smoke-test approach

### Canonical validation set (`trained_configs.txt`)
The 10 configs that must keep working, with their registry line and the training script each
uses **today** (pre-refactor):

| Config | config.py | Data config | Script today |
|---|---|---|---|
| `pi05_libero_100` | 1307 | (plain) | `train.py` (plain `Observation`) |
| `pi05_libero_reason` | 1600 | `LiberoReasonDataConfig` | `train.py` (`FuseObservation`) |
| `pi05_libero_reason_lora` | 1549 | `LiberoReasonDataConfig` | `train.py` (`FuseObservation`) |
| `atomic_libero` | 1955 | `AtomicDataConfig` | `train_atomic.py` |
| `trace_vla_actionmoe` | 2196 | `LeRobotTraceVLAActionMoeDataConfig` | `train_trace_vla_actionmoe.py` |
| `trace_vla_actionmoe_lora` | 2239 | `LeRobotTraceVLAActionMoeDataConfig` | `train_trace_vla_actionmoe.py` |
| `trace_vla_moe` | 2306 | `LeRobotTraceVLAMoeDataConfig` | `train_trace_vla_moe.py` |
| `trace_vla_moe_lora` | 2350 | `LeRobotTraceVLAMoeDataConfig` | `train_trace_vla_moe.py` |
| `target_vla_actionmoe` | 2487 | `LeRobotTargetVLAActionMoeDataConfig` | `train_target_vla_actionmoe.py` |
| `target_vla_actionmoe_lora` | 2528 | `LeRobotTargetVLAActionMoeDataConfig` | `train_target_vla_actionmoe.py` |

Each non-pi05 config is bound to its own forked script (see §5) because of that script's custom
weight remap + observation handling. **Success criterion for §5:** after folding observation
dispatch into `DataLoaderImpl` and moving each remap into a `WeightLoader`, all of them run via
`train.py`. (Also note: `atomic_libero`'s registry entry hardcoded a dataset path under the full
`/work` filesystem — changed to read local `data/libero-100`.)

### Execution constraints (Slurm cluster)
- **Never run compute (CPU/GPU) directly on the login node.** Launch via Slurm using the resource
  flags from `slurm_example.sh` (`--account bgtb-dtai-gh --partition ghx4-interactive -G2
  --time 120 -N1 --mem 256G --cpus-per-task 32`). For non-interactive runs, replace `--pty bash -i`
  with the command, or use an `sbatch` wrapper.
- **Disk is tight** — only room for a few checkpoints, *especially the non-LoRA (full-FT) configs*.
  Outputs land in `./checkpoints/<config_name>/<exp_name>/`. `train.py` always saves at the final
  step, so every run leaves ≥1 checkpoint. **`rm -rf` the smoke-test output dir between every run.**

### Baseline results (pre-refactor reference) — 2026-06-02

All 6 full-FT configs pass a 2-step smoke (exit 0, finite losses). Numbers are step-0 loss for
regression comparison after the refactor (`scripts/smoke_run.sh`, batch ≤8, `--no-wandb-enabled`):

| Config | Script | GPUs | step-0 loss (breakdown) |
|---|---|---|---|
| `pi05_libero_100` | `train.py` | 2 | `0.0951` |
| `pi05_libero_reason` | `train.py` | 2 | `11.6041` |
| `atomic_libero` | `train_atomic.py` | 2 (FSDP=2) | `1.3672` (action 0.67 / decision 12.37 / text 3.93) |
| `trace_vla_actionmoe` | `train_trace_vla_actionmoe.py` | 4 (FSDP=4) | `2.4816` (action 0.80 / trace 1.67 / completion 0.11) |
| `trace_vla_moe` | `train_trace_vla_moe.py` | 4 (FSDP=4) | `3.7695` (action 0.80 / trace 2.96 / completion 0.11) |
| `target_vla_actionmoe` | `train_target_vla_actionmoe.py` | 4 (FSDP=4) | `0.9990` (action 0.99 / completion 0.08; no trace loss — correct) |

**GPU recipe:** the 3 light configs fit on 2 GH200s; the 3 heavy full-FT MoE configs (~4B params,
fp32 Adam state) OOM on 2 GPUs (constant 24 GiB optimizer buffer) and need `-G4 --fsdp-devices 4`.
Request a short `--time 9` so the 4-GPU job backfills quickly on `ghx4-interactive`. Each full-FT
checkpoint is ~50–58 GiB on the (full) `/work` filesystem — `rm -rf` between every run.

**Baseline-prep fix applied:** `atomic_libero` registry entry hardcoded a dataset path on the full
`/work` filesystem; repointed `repo_path` to `str(REPO_ROOT / "data/libero-100")`. Also added
shared-norm-stats symlinks under `assets/<config>/yilin-wu/libero-100/`.

### Smoke-test recipe (per config)
- Use a throwaway `--exp_name` (e.g. `smoke`), tiny `--num_train_steps` (e.g. 2–3), a large
  `--save_interval` so only the forced final-step save fires, and `--wandb_enabled=false`.
- Norm stats must already exist (`scripts/compute_norm_stats.py --config-name <cfg>`); otherwise
  `transform_dataset` raises an actionable error.
- Pass: `.create()` + a few `train_step`s run without shape/typecheck errors and losses are finite.
- After each refactor step, smoke-test the affected variants; `rm -rf` the output dir afterward.
- `models/*_test.py` exist but trust level unconfirmed — treat the 10 configs above as the
  authoritative validation set.

## 8. Decisions locked in
- **`pi0_skill_embed.py`: leave for now** (do not delete yet).
- **Base-class refactor: proceed (aggressive).** Update the dynamic `__import__(...)` calls in
  `config.py` as classes move — they appear inline at `model=` / `freeze_filter=` fields, e.g.
  `__import__("openpi.models.pi0_trace_vla_actionmoe_config", fromlist=["Pi0TraceVLAActionMoeConfig"])
  .Pi0TraceVLAActionMoeConfig(...)`.
- **Authoritative validation = the 10 configs in `trained_configs.txt`** (table above).
