# Module 07 — Training at Scale

You can now build the model and its loss. This module is about turning that into a real
training run: the loop, optimizer, EMA, parameter freezing / LoRA, sharding, and the
config system that wires it all. Read [`scripts/train.py`](../scripts/train.py),
[`training/optimizer.py`](../src/openpi/training/optimizer.py),
[`training/config.py`](../src/openpi/training/config.py).

## Learning objectives

- Read and reproduce `train_step`: grad, frozen-param filtering, optimizer update, EMA.
- Configure the optimizer + LR schedule the way π₀ does (AdamW, cosine warmup, grad clip).
- Explain weight loading from a pretrained checkpoint and `get_freeze_filter` (full / LoRA).
- Understand FSDP sharding and where activation constraints live.
- Read a real `TrainConfig` (e.g. `pi05_libero`) and know what every field does.

---

## 1. The training step

[`train_step` at `train.py:137`](../scripts/train.py). Distilled:

```python
def loss_fn(model, rng, obs, actions):
    return jnp.mean(model.compute_loss(rng, obs, actions, train=True))   # Module 04

train_rng = jax.random.fold_in(rng, state.step)                          # per-step rng
diff_state = nnx.DiffState(0, config.trainable_filter)                   # only diff trainable params
loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, obs, actions)

params  = state.params.filter(config.trainable_filter)
updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
new_params = optax.apply_updates(params, updates)
nnx.update(model, new_params)

if state.ema_decay is not None:                                          # EMA of weights
    ema = tree_map(lambda old, new: ema_decay*old + (1-ema_decay)*new, state.ema_params, new_params)
```

Things to internalize:
- **`fold_in(rng, step)`** gives every step an independent but reproducible rng (for the
  noise/time/augmentation draws inside `compute_loss`). Determinism matters for resume.
- **`trainable_filter`** ([`config.py:636`](../src/openpi/training/config.py)) =
  `All(Param, Not(freeze_filter))`. Gradients are only computed and applied for trainable
  params — this is how freezing/LoRA is enforced at the gradient level via
  `nnx.DiffState`, not by zeroing grads after the fact.
- **EMA** ([`train.py:169`](../scripts/train.py)): a shadow copy of weights updated as an
  exponential moving average (`ema_decay=0.99` default,
  [`config.py:576`](../src/openpi/training/config.py)). EMA weights are usually what you
  evaluate/deploy — they're smoother and generalize better. LoRA configs set
  `ema_decay=None` to disable it ([`config.py:846`](../src/openpi/training/config.py)).
- **Metrics** ([`train.py:186`](../scripts/train.py)): loss, global grad norm, kernel
  param norm. Watch `grad_norm` — with `clip_by_global_norm(1.0)` it should sit near the
  clip threshold early then settle.

The whole step is `jit`-compiled and runs under the device mesh (sharding below).

---

## 2. Optimizer & schedule

[`optimizer.py`](../src/openpi/training/optimizer.py). The π₀ defaults:

- **AdamW**, `b1=0.9, b2=0.95, eps=1e-8`, tiny `weight_decay=1e-10`, **grad clip 1.0**
  ([`optimizer.py:65`](../src/openpi/training/optimizer.py)). The clip is wrapped first in
  the `optax.chain` so clipping happens before the Adam update
  ([`optimizer.py:85`](../src/openpi/training/optimizer.py)).
- **Cosine decay with warmup** ([`optimizer.py:15`](../src/openpi/training/optimizer.py)):
  warm up 1k steps to `peak_lr=2.5e-5`, cosine-decay to `2.5e-6` over 30k steps.
  (`RsqrtDecaySchedule` is the alternative, used for some long runs.)

These LRs are *small* — you're fine-tuning a pretrained 3B+ model, not training from
scratch. A weight-decay mask excludes biases/norms/embeddings from decay (built from the
same kernel filter used for the param-norm metric, [`train.py:178`](../scripts/train.py)).

---

## 3. Weight loading & freezing (full vs. LoRA)

**Loading pretrained weights.** A `TrainConfig` names a `weight_loader`, e.g.
`CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params")`
([`config.py:760`](../src/openpi/training/config.py)). After the model is created with
random init, the loader overwrites the matching subtree (the PaliGemma + action-expert
weights). The clean `_name` suffix scheme from Module 03 is what lets the VLM weights drop
in by name; the freshly-initialized action-expert weights (`_1`) are left as-is or also
loaded from the base checkpoint.

**Freezing.** `Pi0Config.get_freeze_filter` ([`pi0_config.py:88`](../src/openpi/models/pi0_config.py))
builds an `nnx` filter over parameter paths:
- LoRA on the VLM (`gemma_2b_lora`) → freeze all `llm` params *except* the action expert
  and except `lora` params. So only LoRA adapters + the action expert train.
- The path regexes (`.*llm.*`, `.*llm.*_1.*`, `.*lora.*`) encode "freeze the big VLM,
  train small adapters." This is both a memory win (fits LoRA fine-tuning on a 4090,
  README table) and a **practical form of knowledge insulation** (Module 06): the VLM
  can't be dragged by the flow gradient because it isn't updated.

**LoRA itself** ([`lora.py`](../src/openpi/models/lora.py)): low-rank `A·B` adapters added
to the Gemma einsums/FFN, configured by `LoRAConfig(rank, alpha)`
([`gemma.py:96`](../src/openpi/models/gemma.py)). You know LoRA; just note it's wired
through the `lora.Einsum`/`lora.FeedForward` wrappers so the same Gemma code serves full
and LoRA training.

---

## 4. Sharding (FSDP + data parallel)

`fsdp_devices` ([`config.py:617`](../src/openpi/training/config.py)) controls a 2D device
mesh in [`sharding.py`](../src/openpi/training/sharding.py): shard model params/optimizer
state across `fsdp_devices` (FSDP, reduces per-GPU memory) and run data-parallel across
the remaining device groups. Activation sharding constraints are sprinkled through the
forward pass (`sharding.activation_sharding_constraint`, e.g.
[`gemma.py:294`](../src/openpi/models/gemma.py),
[`siglip.py:86`](../src/openpi/models/siglip.py)) to keep intermediate tensors laid out
sensibly. Single-node only in this repo (README). For the course you'll train tiny models
on 1 device, so `fsdp_devices=1`, but you should understand *why* the constraints are
there: a 3B model + optimizer state doesn't fit on one card at full precision.

---

## 5. Reading a real config

Look at `pi05_libero` ([`config.py:893`](../src/openpi/training/config.py)):
```python
TrainConfig(
    name="pi05_libero",
    model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
    data=LeRobotLiberoDataConfig(...),
    weight_loader=CheckpointWeightLoader(".../pi05_base/params"),
    ema_decay=0.999,
    num_train_steps=30_000,
)
```
Read it like a sentence: "π₀.₅ with a 10-step action horizon (note `discrete_state_input=
False` here — LIBERO feeds state continuously), fine-tuned from the π₀.₅ base checkpoint
on LIBERO data, with EMA 0.999, for 30k steps." Every named config is just this struct
with different knobs. `tyro` turns the dataclass into a CLI:
```
uv run scripts/train.py pi05_libero --exp-name my_run
```
(Record your exact Babel commands in `COMMANDS.md` per your convention.)

---

## 6. The full loop (around `main`)

[`train.py:194`](../scripts/train.py): init logging/wandb → build mesh & sharding → create
model + apply weight loader → build optimizer & `TrainState` (with EMA params) →
`init_train_state` → loop `num_train_steps` calling the jitted `train_step`, logging every
`log_interval`, checkpointing every `save_interval` via
[`checkpoints.py`](../src/openpi/training/checkpoints.py) (orbax, keeps `keep_period`).
Resume reads the wandb run id and restores `TrainState`.

---

## Self-check

1. How is parameter freezing enforced — by zeroing gradients, or earlier? Where?
2. What is EMA of weights and why deploy it instead of the raw params? When is it off?
3. Why are the learning rates so small (~2.5e-5)?
4. How does LoRA fine-tuning approximate knowledge insulation?
5. What two kinds of parallelism does `fsdp_devices` trade between?
6. Read `pi05_droid` ([`config.py:716`](../src/openpi/training/config.py)) and state in
   one sentence what it trains.

## Lab 07

Open [`labs/lab07_training.py`](labs/lab07_training.py). You will:
1. Build an optax optimizer (AdamW + clip + cosine-warmup schedule) matching
   `optimizer.py` defaults.
2. Write a minimal `train_step` (value_and_grad on `compute_loss`, apply updates, EMA)
   for your toy model — no sharding.
3. Implement a `freeze_filter` that trains only the action expert (freezes the VLM) and
   confirm via param counts that the VLM params receive zero updates over a step.
4. Overfit a single synthetic batch for a few hundred steps and watch the flow loss fall
   — your first end-to-end VLA training run.

Next: [Module 08 — Capstone: minimal π₀.₅ from scratch](08_capstone_minimal_pi05.md).
