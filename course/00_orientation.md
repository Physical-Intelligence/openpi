# Module 00 — Orientation: What a VLA Is, and the Shape of π₀.₅

## Learning objectives

By the end of this module you can:
- State precisely what a vision-language-action model maps from and to.
- Draw the π₀.₅ data flow end-to-end from memory: pixels + proprioception + language →
  action chunk.
- Locate every major component in the `openpi` source tree.
- Explain the three model variants in this repo (π₀, π₀-FAST, π₀.₅) and how they differ
  at the top level.

---

## 1. What problem are we solving?

A **VLA** is a policy `π(actions | observation, instruction)` where the observation is
high-dimensional and multimodal (camera images + robot proprioceptive state) and the
instruction is natural language. The output is **not a single action** but a short
**action chunk** — a sequence of future actions, predicted open-loop and executed
before re-planning.

Formally, π₀.₅ models:

```
a_{t:t+H}  ~  π( · | I_1..I_k, q_t, ℓ )
```

- `I_1..I_k`  — k camera views (base + wrists), each a 224×224 RGB image.
- `q_t`       — robot proprioceptive state (joint angles, gripper, etc.).
- `ℓ`         — a language instruction ("fold the towel").
- `a_{t:t+H}` — an action chunk of horizon `H` (50 for π₀, 10–15 for some π₀.₅ configs),
  each action a vector of dim `action_dim` (padded to 32).

Two design commitments define this whole model family and you should internalize them
now:

1. **Reuse a pretrained vision-language model (VLM).** The semantic backbone is
   **PaliGemma** (a SigLIP vision encoder + a Gemma 2B language model). The robot
   doesn't learn "what a towel is" from scratch — it inherits web-scale semantics.
   *(π₀ paper §3; the whole point of starting from a VLM.)*

2. **Generate continuous actions with flow matching, not token-by-token decoding.**
   Robot control needs smooth, high-frequency, high-dimensional actions. π₀ attaches a
   **flow-matching "action expert"** that denoises an entire action chunk in parallel.
   (π₀-FAST is the autoregressive-tokens alternative; π₀/π₀.₅ are the flow variants.)

> **Why chunks, not single actions?** Predicting `H` actions at once and executing them
> open-loop (a) reduces compounding error from per-step re-planning, (b) produces
> temporally consistent, non-jittery motion, and (c) amortizes the expensive VLM
> forward pass over many control steps. This idea comes from action-chunking
> policies (ACT / diffusion policy) and is load-bearing for real-time control.

---

## 2. The π₀.₅ forward pass, end to end

Memorize this sequence — the rest of the course is just zooming into each arrow.

```
                 ┌─────────────────────── PREFIX (VLM, "gemma" expert) ──────────────────────┐
 images  ──► SigLIP ViT ──► image tokens ┐
                                          ├─► concat ─► token seq ─►┐
 language ─► Gemma embedder ─► text tokens┘    (+ discrete state     │
                                                tokens for π₀.₅)     │
                                                                     ▼
                                              ┌────────────────────────────────────────┐
                                              │  Gemma transformer, 18 blocks, shared    │
                                              │  attention across two "experts":         │
                                              │   • prefix tokens use gemma_2b weights    │
 noisy actions x_t  ─► action_in_proj ─┐      │   • suffix tokens use gemma_300m weights  │
 timestep t        ─► time MLP (adaRMS)─┴─►suffix│  Blockwise attention mask connects them │
                 ┌──────── SUFFIX (action expert, "gemma_300m") ─────────┐               │
                                              └────────────────────────────────────────┘
                                                                     │
                                                  suffix outputs ────┘
                                                                     ▼
                                                      action_out_proj  ─►  v_t  (flow velocity)
```

In code this is two methods that build the two halves of the sequence, plus one LLM
call that runs them jointly:

- [`Pi0.embed_prefix`](../src/openpi/models/pi0.py) — [`pi0.py:105`](../src/openpi/models/pi0.py): images + language (+ state, for π₀.₅) → prefix tokens.
- [`Pi0.embed_suffix`](../src/openpi/models/pi0.py) — [`pi0.py:139`](../src/openpi/models/pi0.py): noisy actions + timestep → suffix tokens.
- [`Pi0.compute_loss`](../src/openpi/models/pi0.py) — [`pi0.py:188`](../src/openpi/models/pi0.py): training (one joint forward pass).
- [`Pi0.sample_actions`](../src/openpi/models/pi0.py) — [`pi0.py:216`](../src/openpi/models/pi0.py): inference (KV-cache the prefix, integrate the ODE on the suffix).

The "two experts sharing attention" idea is the structural heart of the model. Hold the
question *"why two experts and not one?"* in your head — Module 03 answers it, and the
short version is: the VLM is big and frozen-ish and processes perception/language; the
action expert is small, fast, and trained to denoise actions, and it gets to **attend
into** the VLM's representations without polluting them.

---

## 3. The three variants in this repo

All three live behind `Pi0Config` / `Pi0FASTConfig` and share the PaliGemma backbone.
The difference is entirely in **how actions are represented and produced**:

| Variant | Action representation | Head | Config | Where |
|---------|----------------------|------|--------|-------|
| **π₀** | continuous chunk | flow matching (concat-MLP time cond.) | `Pi0Config(pi05=False)` | `pi0.py` |
| **π₀-FAST** | discrete tokens (FAST/DCT codes) | autoregressive decoding | `Pi0FASTConfig` | `pi0_fast.py` |
| **π₀.₅** | continuous chunk | flow matching (**adaRMS** time cond.) | `Pi0Config(pi05=True)` | `pi0.py` |

π₀ and π₀.₅ are *the same Python class* `Pi0`; the `pi05` boolean toggles three things
you'll meet in Module 06:

1. **State placement.** π₀ feeds robot state as a continuous token in the *suffix*
   ([`pi0.py:151`](../src/openpi/models/pi0.py)). π₀.₅ discretizes state and folds it
   into the *language* tokens of the prefix ([`tokenizer.py:22`](../src/openpi/models/tokenizer.py)).
2. **Time conditioning.** π₀ concatenates the timestep embedding with the action
   embedding and MLP-mixes them ([`pi0.py:170`](../src/openpi/models/pi0.py)). π₀.₅ uses
   **adaptive RMSNorm** (adaRMS) to inject the timestep as scale/shift/gate inside every
   transformer block ([`gemma.py:112`](../src/openpi/models/gemma.py)).
3. **Token budget.** `max_token_len` defaults to 48 for π₀, 200 for π₀.₅ (room for the
   discrete state string) — [`pi0_config.py:38`](../src/openpi/models/pi0_config.py).

> Note from the repo README: openpi currently implements **only the flow-matching head**
> for π₀.₅ training/inference, even though the paper's full recipe ("knowledge
> insulation") also uses a discrete autoregressive action loss during pretraining. We
> cover the full method conceptually in Module 06 and implement the part that ships.

---

## 4. Map of the source tree

Spend ten minutes with `find src/openpi -name '*.py'` open. The pieces you care about:

```
src/openpi/
├── models/
│   ├── model.py          # Observation/Actions structs, preprocessing, BaseModel ABC
│   ├── pi0_config.py     # Pi0Config (the pi05 toggle, freeze filters, input specs)
│   ├── pi0.py            # THE model: embed_prefix/suffix, compute_loss, sample_actions
│   ├── pi0_fast.py       # autoregressive variant (contrast/reference only)
│   ├── gemma.py          # Gemma transformer w/ multi-expert attention + (ada)RMSNorm
│   ├── siglip.py         # SigLIP/ViT vision encoder
│   ├── tokenizer.py      # PaliGemma tokenizer (+ discrete state for pi05), FAST tokenizer
│   └── lora.py           # LoRA-wrapped Einsum/FeedForward
├── transforms.py         # data pipeline: repack, normalize, tokenize, pad
├── training/
│   ├── config.py         # TrainConfig + all named configs (pi05_droid, pi05_libero, ...)
│   ├── optimizer.py      # schedules + optax chains
│   ├── data_loader.py    # LeRobot dataset → batches
│   ├── sharding.py       # FSDP/data-parallel mesh + activation constraints
│   └── checkpoints.py    # orbax save/restore
└── policies/             # per-robot input/output adapters + Policy.infer wrapper
scripts/train.py          # the training loop (train_step, EMA, logging)
```

You do **not** need `models_pytorch/` for this course — it's a parallel PyTorch port of
the same model. We work in the JAX/Flax (NNX + Linen-bridge) implementation, which is
canonical.

---

## 5. A note on the NNX / Linen split (so it doesn't confuse you later)

`openpi` is mid-migration between Flax's two APIs:

- The **top-level `Pi0` model is `flax.nnx`** (object-oriented, mutable modules) — see
  `class Pi0(_model.BaseModel)` at [`pi0.py:66`](../src/openpi/models/pi0.py).
- The **Gemma transformer and SigLIP are still `flax.linen`** (functional, `setup`/
  `@nn.compact`), wrapped via `nnx_bridge.ToNNX` at [`pi0.py:73`](../src/openpi/models/pi0.py).

This is why you'll see `llm.lazy_init(...)` and `method="embed"` calls — the bridge
exposes Linen modules as NNX submodules. For the course you can treat the bridge as
plumbing; just know *which* API a file uses before you read it. Your from-scratch
capstone may use either; the labs use NNX for the outer model and Linen for the
transformer to mirror the repo.

---

## Self-check (answer before moving on)

1. What are the four inputs and the one output of π₀.₅, with shapes?
Inputs:
- prompt: (batch_size, max token len)
- state: (batch_size, 32)
- observations: a dictionary of (Batch_size, image_resolution, 3) 
- obs and prompt masks
Output:
- actions: (Batch_size, 50, 32) 

2. Why does the model output a *chunk* of actions instead of one action?
- Makes actions more stable and reduces latency
- non jittery

3. Name the two "experts" and which weights each uses. Do they share attention?
- 300M action expert and 2B Gemma VLM using SigLip. They share KV cache,
4. Give the three things the `pi05` flag changes.
   1. Uses Adaptive RMS norm to fuse states in the action expert.
   2. the state input is part of the discrete language tokens rather than a continuous input in the suffix
5. Which is autoregressive: π₀, π₀.₅, or π₀-FAST?
- pi0fast

## Lab 00

Open [`labs/lab00_orientation.py`](labs/lab00_orientation.py). It's a guided
exploration: instantiate the real `Pi0Config`, build a tiny model, run `fake_obs`
through it, and print the shape of every intermediate. The goal is to *see* the data
flow you drew above before you implement any of it.

Next: [Module 01 — Data & representations](01_data_and_representations.md).
