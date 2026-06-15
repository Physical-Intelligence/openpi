# Module 08 — Capstone: A Minimal π₀.₅ From Scratch

The proof that you understand a model is that you can rebuild it. In this capstone you
assemble everything from Modules 01–07 into **`minipi05`**: a small, single-package,
clean-room π₀.₅ in JAX/Flax that trains on a toy dataset and runs flow-matching inference
— with `openpi` as the answer key, not a dependency to copy.

## Goal & success criteria

Build `labs/minipi05/` such that:

1. **Architecture.** It has: a (tiny) ViT image encoder, a text/state token embedder, a
   **two-expert** transformer with shared attention, blockwise masking, RoPE, RMSNorm +
   **adaRMS**, a flow-matching action head.
2. **π₀.₅ semantics.** State is discretized into the prompt; time is injected via adaRMS
   into the action expert only; quantile-normalized actions.
3. **Trains.** `compute_loss` is the conditional flow-matching MSE; a training loop drives
   the loss down on a toy "reach-to-target" dataset.
4. **Samples.** `sample_actions` integrates the ODE from noise with a cached prefix and
   produces sensible action chunks; cached and uncached sampling agree.
5. **Verified.** A test reproduces the openpi behaviors you've checked module-by-module
   (mask semantics, suffix shapes, π₀ vs π₀.₅ differences, loss→0 on a memorizable batch).

Keep it **small**: width ~128, depth ~4, 1–2 heads, a downsampled image so patches are
few. The point is correctness and clarity, not scale. You should be able to train it on
CPU or a single GPU in minutes.

---

## Suggested structure

```
labs/minipi05/
├── config.py        # MiniConfig: dims, horizons, pi05 flag
├── vit.py           # tiny patch-embed ViT  → image tokens          (Module 02)
├── tokenizer.py     # prompt+discrete-state → token ids             (Modules 01,06)
├── transformer.py   # two-expert Block, attention, RoPE, (ada)RMS   (Module 03)
├── masks.py         # make_attn_mask                                (Module 03)
├── model.py         # MiniPi05: embed_prefix/suffix, compute_loss, sample_actions
├── data.py          # toy dataset + quantile normalization          (Module 01)
├── train.py         # optimizer, train_step, EMA, loop              (Module 07)
└── test_against_openpi.py   # cross-checks vs the real classes
```

You've already built most of these as lab stubs in Modules 01–07. The capstone is
largely **integrating your own lab solutions** and making the full forward/backward/sample
loop run.

---

## Build order (incremental, test each step)

1. **`masks.py` + `transformer.py`** (Module 03). Get a two-expert block running on
   random `[x_prefix, x_suffix]` with a hand-built mask. Test: prefix outputs are
   invariant to suffix contents (the insulation property).
2. **`vit.py` + `tokenizer.py`** (Modules 02, 06). Image → tokens; prompt+state → ids →
   embeddings. Test: token counts, and that state binning matches openpi's `np.digitize`.
3. **`model.embed_prefix` / `embed_suffix`** (Modules 02, 04, 06). Assemble the prefix and
   the adaRMS suffix. Test: shapes + `adarms_cond` is non-None, no state token in suffix.
4. **`model.compute_loss`** (Module 04). One joint forward pass, flow MSE. Test: finite
   scalar, gradients reach action projections, loss→0 on one memorizable batch.
5. **`data.py`** (Module 01). A toy task where the "right" action chunk is a deterministic
   function of the observation (e.g. move gripper toward a target whose location is given
   in the image / state), with quantile normalization. This makes success *visible*.
6. **`train.py`** (Module 07). AdamW + cosine warmup + clip + EMA. Overfit, then
   generalize on held-out toy episodes.
7. **`model.sample_actions`** (Module 05). Uncached first, then KV-cached; assert they
   agree. Evaluate: sampled chunks should solve the toy task.
8. **`test_against_openpi.py`**. Import the real `openpi` classes and assert your
   components match on shared sub-behaviors (see below).

---

## Cross-checks against openpi (the "answer key")

You won't match weights (different init), but you can match **behaviors and shapes**:

- `make_attn_mask`: byte-for-byte equal to `pi0.make_attn_mask` on the same inputs.
- `posemb_sincos`: equal to `pi0.posemb_sincos`.
- State binning: equal to `PaligemmaTokenizer.tokenize(prompt, state)` digitization.
- adaRMS with `cond=None` reduces to standard RMSNorm (compare to `gemma.RMSNorm`).
- Suffix length & `adarms_cond` presence differ correctly between `pi05=True/False`.
- `sample_actions` returns `[b, action_horizon, action_dim]`; cached == uncached.

A good final test instantiates a **real** `Pi0Config(pi05=True).create(...)` on the
`dummy` Gemma variant, runs `fake_obs()` through it, and confirms your mental model of
every intermediate shape matches.

---

## Stretch goals (toward the full paper)

Only after the minimal version works:

1. **Knowledge insulation (real).** Add a discrete action head (cross-entropy on
   FAST-style or simple binned action tokens) trained on the VLM expert, and
   `stop_gradient` the prefix representations going into the flow expert. Observe that
   language-conditioning quality holds up better than flow-only on a toy task with
   distractor instructions. (Module 06.)
2. **π₀ mode.** Flip your `pi05` flag off: continuous state token in the suffix,
   concat-MLP time conditioning, z-score norm. One model, both behaviors — exactly like
   `Pi0`.
3. **Real fine-tune.** Graduate from `minipi05` to the actual repo: fine-tune
   `pi05_libero` or a small custom dataset with `scripts/train.py`, on your Babel
   cluster. Record commands in `COMMANDS.md`. Compare your understanding of each config
   field to what you built.
4. **π₀-FAST contrast.** Read [`pi0_fast.py`](../src/openpi/models/pi0_fast.py) and write
   a paragraph on the exact trade-offs vs. your flow model — you'll now read it fluently.

---

## What "done" looks like

You can sit down at a blank file and, without looking, write: the two sequence-builders,
the blockwise mask, the two-expert attention, the adaRMS norm, the flow loss with correct
noise/time conventions, and the cached Euler sampler — and explain *why* each is shaped
the way it is and how π₀.₅ differs from π₀. At that point you can build a VLA from
scratch, which was the goal.

---

## Self-check (final)

1. Without looking: write `x_t`, `u_t`, the loss, and which end is noise.
2. Draw the prefix/suffix attention block structure and the resulting attendance table.
3. Name every place the timestep enters a π₀.₅ forward pass.
4. Explain why the prefix can be KV-cached and the suffix re-run, in terms of the mask.
5. Explain knowledge insulation to a colleague in three sentences.

Congratulations — that's the course. Keep [`reading_guide.md`](reading_guide.md) handy as
you go back to the papers; everything in them should now map to code you can write.
