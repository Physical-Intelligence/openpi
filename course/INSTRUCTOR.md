# Instructor Guide — How to Teach This Course

**Audience: a future Claude Code agent (or human TA) acting as Leo's instructor for the
VLA course in `/course`.** Read this first. It tells you the student, the pedagogy, how
to run a session, what each module must land, the misconceptions to pre-empt, and a
grading rubric with verified answer-key facts so you can check labs without re-deriving
them.

> Keep this guide truthful to the code. If you cite a `file:line`, verify it still exists
> before asserting it — the repo evolves. The structural claims (two experts, flow loss,
> adaRMS, KV cache) are stable; exact line numbers may drift.

---

## 0. The student & the goal

- **Who:** Leo — aspiring robotics engineer. Comfortable with **JAX, transformers, and
  flow/diffusion policies**. He explicitly wants *rigor* and the ability to **build a VLA
  from scratch**, not a tour.
- **Reference model:** π₀.₅ as implemented in this repo (`Pi0Config(pi05=True)`), using π₀
  as the stepping stone.
- **Format he chose:** lecture notes + build labs; brief prereq refreshers + deep
  VLA-specific theory; capstone = minimal π₀.₅ reimplementation (`labs/minipi05/`);
  written material + code stubs he completes himself. He runs real training on the CMU
  **Babel** cluster (see his memory; commands go in `COMMANDS.md`, gitignored).
- **Definition of done (his words):** he can sit at a blank file and write the two
  sequence-builders, the blockwise mask, two-expert attention, adaRMS, the flow loss with
  correct conventions, and the cached Euler sampler — and explain *why* each is shaped
  that way and how π₀.₅ differs from π₀.

**Do not** lecture him through prerequisites he has. **Do** push on the VLA-specific
"why," make him implement, and verify his implementations against openpi.

---

## 1. Teaching philosophy (how to run a session)

1. **Diagnose first.** Ask which module he's on and have him state the prior module's
   self-check answers from memory. If shaky, review before advancing.
2. **Socratic over expository.** Prefer "why is the prefix bidirectional but the suffix
   not?" over re-explaining. The modules already contain the exposition; your job is to
   probe understanding and unblock.
3. **Always tie to code.** Every concept has a `file:line` anchor. When he's confused,
   open the real source together, don't paraphrase.
4. **Lab-gated progression.** He should not move to module N+1 until lab N's `check_*()`
   functions pass (or, for discussion/stretch parts, he's implemented and tested the
   behavior). You verify by running the lab.
5. **Reinforce the gotchas** (§4) every time they're relevant — they cause the most
   silent bugs.
6. **Capstone is the proof.** Treat modules 01–07 as building the pieces; module 08 is
   integration. Don't let him skip straight to it.

A good session loop: *check prior self-checks → introduce the module's central question →
let him attempt the lab → review his code against the rubric → probe the "why" → assign
the next module + paper section.*

---

## 2. Per-module teaching notes

For each module: **must-land points**, **misconceptions to pre-empt**, **good probing
questions**, and the **lab pass bar**.

### Module 00 — Orientation
- **Land:** VLA = `π(action chunk | images, state, language)`; chunks (not single
  actions) for temporal consistency + amortized VLM cost; two design commitments (reuse a
  VLM; generate continuous actions with flow matching); π₀ vs π₀-FAST vs π₀.₅.
- **Misconception:** "π₀.₅ is a different model file." It's `Pi0` with `pi05=True`.
- **Probe:** the four inputs + one output with shapes; which variant is autoregressive.
- **Lab bar:** `lab00` is exploration — he should print the specs and *correctly state*
  the total image-token count he'll confirm in Lab 02.

### Module 01 — Data & representations
- **Land:** static-shape masking (missing camera/short prompt/low-DoF all handled by
  masking fixed tensors, never reshaping — JAX jit constraint); normalization makes
  flow-matching noise well-conditioned; quantile norm (π₀.₅) bounds to [-1,1] which the
  state binner *requires*; image preprocessing lives *inside* the model, normalization is
  a *data transform*.
- **Misconception:** "augmentation applies to all cameras." Wrist cams get only color
  jitter (rigid mount → no spatial aug).
- **Probe:** why π₀.₅ uses quantile norm and what downstream component needs [-1,1].
- **Lab bar:** `lab01` Part 1 prints `OK` (his quantile norm matches `transforms.Normalize`
  and round-trips); Part 2 builds a valid `Observation` with 3 canonical image keys in
  [-1,1].

### Module 02 — Vision-language backbone
- **Land:** So400m/14 on 224² → 16×16 = **256 tokens/view**, `pool_type="none"` keeps
  per-patch tokens (spatial grounding needed for manipulation); embedder `×sqrt(dim)`
  scaling; `embed_prefix` builds parallel `(tokens, input_mask, ar_mask)`; prefix is one
  big bidirectional block.
- **Misconception:** confusing `input_mask` (real vs padding) with `ar_mask` (AR-block
  boundaries).
- **Probe:** why not pool to one vector; distinguish the two masks in one sentence each.
- **Lab bar:** `lab02` prints `embed_prefix OK` — token count = `n_views*N + prompt_len`,
  `ar_mask` all-False, masked-out view has `input_mask=0`.

### Module 03 — The MoE transformer (the crux; spend the most time here)
- **Land:** "expert" = full per-token-group weights with **shared attention** (NOT
  token-routed MoE); per-expert QKV/output projections, shared softmax in the middle;
  this split is *exactly* what enables prefix KV-caching and gradient insulation; blockwise
  `make_attn_mask` via `cumsum(ar_mask)`; the attendance table (actions read everything;
  perception never reads actions); experts must share head geometry; the `_1` naming
  trick for clean checkpoint loading.
- **Misconception:** thinking attention is computed separately per expert. It's joint.
- **Probe:** hand-trace `make_attn_mask([F,F,T,F], [T,T,T,F])` → 4×4 matrix; why the
  action expert can be ~7× smaller.
- **Lab bar:** `lab03` Part 1 prints `OK` (matches `pi0.make_attn_mask`); Part 2 he
  demonstrates the **insulation property** (prefix output invariant to suffix contents).

### Module 04 — Flow-matching action head
- **Land:** `x_t = t·noise + (1-t)·actions`, `u_t = noise - actions = dx_t/dt`; **t=1 is
  noise, t=0 is data — opposite of the paper**; loss = `mean((v_θ - u_t)²)`; Beta(1.5,1)
  oversamples high noise; π₀ injects time by concat-MLP + has a suffix state token, π₀.₅
  injects time via adaRMS + no state token; one joint forward pass (no cache in training).
- **Misconception:** assuming textbook FM time direction; forgetting π₀ has a state token
  in the suffix (so length H+1) while π₀.₅ does not (length H).
- **Probe:** verify `u_t = dx_t/dt`; which end is noise; why no KV cache in `compute_loss`.
- **Lab bar:** `lab04` Parts 1–2 print `OK` (posemb matches; interpolant correct); Part 3
  he correctly produces the H vs H+1 / adarms-None difference.

### Module 05 — Inference: ODE + KV cache
- **Land:** sampling = Euler-integrate `dx/dt = v_θ` from t=1→0, `dt=-1/num_steps`, ~10
  steps suffice because OT paths are nearly straight; prefix cached once (depends only on
  obs), suffix re-run each step; three masks (suffix→suffix, suffix→prefix, combined) and
  suffix positions continue after the prefix; adaRMS recomputed each step (depends on t).
- **Misconception:** "why so few steps?" (straight paths) vs image diffusion's many.
- **Probe:** which experts/tokens run in cache-fill vs per-step; why prefix is cacheable
  but suffix is not (the mask).
- **Lab bar:** `lab05` — cached == uncached to fp tolerance; output `[b,H,action_dim]`;
  num_steps sweep shows error decreasing toward the reference.

### Module 06 — π₀.₅ & knowledge insulation
- **Land:** the 3 deltas (discrete state in prompt; adaRMS in action expert only; token
  budget 200 + quantile norm); co-training on heterogeneous data is the *generalization*
  driver; **KI = (a) train the VLM with a discrete action CE loss + (b) stop flow-expert
  gradients from rewriting the VLM**, enabled by the two-expert split; **the repo ships
  only the flow head** (base ckpt was KI-trained, but the open training loop omits the
  discrete-action loss + insulation `stop_gradient`); LoRA/freeze is a crude insulation.
- **Misconception:** "π₀.₅ generalizes because of adaRMS/architecture." It's mostly
  **data**; architecture buys efficiency.
- **Probe:** the two halves of KI and the problem each solves; what openpi omits.
- **Lab bar:** `lab06` Part 1a `OK` (digitize convention); adaRMS reduces to standard
  RMSNorm when `cond=None`; he writes a sensible answer to the stop_gradient discussion.

### Module 07 — Training at scale
- **Land:** freezing enforced at the *gradient* level via `nnx.DiffState`+`trainable_filter`
  (not post-hoc zeroing); EMA of weights is what you deploy (off for LoRA); LRs are small
  because it's fine-tuning a pretrained 3B+ model; AdamW(0.9,0.95)+clip 1.0+cosine warmup;
  FSDP shards params, data-parallel across groups; reading a `TrainConfig` like a sentence.
- **Misconception:** thinking grads are computed then masked. They're never computed for
  frozen params.
- **Probe:** how freezing is enforced; why deploy EMA; what `fsdp_devices` trades.
- **Lab bar:** `lab07` Part 1 `OK` (optimizer clips + small warmup LR); he overfits one
  batch and the loss falls.

### Module 08 — Capstone
- **Land:** integration of his own lab solutions into `minipi05`; build order is
  incremental with tests at each step; success = trains on a toy task + samples + passes
  `test_against_openpi.py`.
- **Coach, don't solve.** Point at the lab he already wrote for each piece. Insist on the
  cross-checks (mask, posemb, binning, cached==uncached, pi05-vs-pi0 suffix).
- **Stretch only after minimal works:** real KI (`stop_gradient` + discrete head), π₀
  mode, a real `pi05_libero` fine-tune on Babel, π₀-FAST contrast.

---

## 3. Grading rubric — verified answer-key facts

These were **run against the live code** (2026-06-11) and can be used to confirm a lab is
correct. Re-verify if the repo changed.

- **`make_attn_mask([[T,T,T,F]], [[F,F,T,F]])`** →
  ```
  [[1 1 0 0]
   [1 1 0 0]
   [1 1 1 0]
   [0 0 0 0]]
  ```
  (tokens 0,1 form a bidirectional block and cannot see token 2 which starts a new AR
  block; token 2 attends to 0,1,2; token 3 is padding → all zero.)
- **`posemb_sincos(linspace(0.001,0.999,8), 64)`** → shape `(8, 64)`, equals
  `pi0.posemb_sincos(..., min_period=4e-3, max_period=4.0)`.
- **State digitize** `np.digitize(s, np.linspace(-1,1,257)[:-1]) - 1` on
  `[-1.0,-0.5,0.0,0.5,0.999]` → `[0, 64, 128, 192, 255]`. (So bin(−1)=0, bin(0)=128.)
- **Quantile norm** `(x-q01)/(q99-q01+1e-6)*2-1` must match `transforms.Normalize(
  use_quantiles=True)` and invert via `Unnormalize`.
- **Suffix shape rule:** `pi05=True` → suffix length = `action_horizon`, `adarms_cond`
  not None. `pi05=False` → length = `action_horizon+1` (leading state token),
  `adarms_cond` is None.
- **Cached vs uncached sampling** must agree to ~`1e-4` when given identical `noise`.
- **adaRMS** with `cond=None` and zero-init scale must equal standard RMSNorm output.

How to check a lab: `cd /home/leo/openpi && uv run python course/labs/labXX_*.py` and
confirm the `OK` prints. For TODO/discussion parts, read his implementation and compare to
the relevant openpi source (he should match *behavior/shapes*, not weights).

---

## 4. Convention gotchas to reinforce (the silent-bug list)

1. **Time direction is inverted vs the paper:** in code, `t=1`=noise, `t=0`=data, so
   `dt<0` in sampling. (`pi0.py` says so in a comment.)
2. **`input_mask` ≠ `ar_mask`.** Real-vs-padding vs AR-block-boundary.
3. **π₀ has a suffix state token; π₀.₅ does not** (state is in the prompt).
4. **adaRMS only on the action expert** (`use_adarms=[False, True]`); the VLM never gets
   it.
5. **Mask shapes at inference** are the #1 bug source — keep the asserted shape checks.
6. **Mixed precision everywhere:** variance/RoPE/softmax in float32, cast back to bf16.
   Watch `.astype`.
7. **The model's contract ends at normalized actions;** the transform shell handles
   physical units and robot-specific keys.

---

## 5. Maintenance (keep the course teachable)

- If `openpi` refactors `pi0.py`/`gemma.py`, the **concepts** stay valid but update
  `file:line` anchors in the modules and the rubric values in §3.
- The two PDFs (`2410.24164` = π₀, `2504.16054` = π₀.₅) are the primary sources;
  `reading_guide.md` maps paper §→code→module.
- Cross-checks that need **no network**: `make_attn_mask`, `posemb_sincos`, quantile norm,
  state digitize. Anything touching the PaliGemma tokenizer or `gs://` checkpoints needs
  access; those checks are marked optional in the labs.
- Leo's preferences: rigor, implement-don't-watch, record run commands in `COMMANDS.md`,
  trains on Babel. Don't re-teach JAX/transformers/flow-matching basics.

If you're picking this up cold: read `README.md`, then this guide, then skim
`pi0.py` (`embed_prefix`/`embed_suffix`/`compute_loss`/`sample_actions`) and `gemma.py`
(`Attention`/`Block`/`RMSNorm`). That's ~600 lines and the entire model.
