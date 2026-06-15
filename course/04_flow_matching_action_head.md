# Module 04 — The Flow-Matching Action Head

This is where π₀ stops being "a VLM" and becomes "a policy." You already know flow
matching as a generative-modeling tool; here we use it **conditionally** to generate
action chunks. Read [`pi0.py:139`–`214`](../src/openpi/models/pi0.py) with these notes.

## Learning objectives

- State the conditional flow-matching objective π₀ uses, with its exact noise/time
  conventions (which differ from the textbook and from the paper — careful!).
- Implement `embed_suffix`: actions + timestep → suffix tokens, both the π₀ (concat-MLP)
  and π₀.₅ (adaRMS) ways.
- Implement `compute_loss` end-to-end.
- Explain the beta-distributed time sampling and why it's used.

---

## 1. Flow matching refresher, in π₀'s conventions

We want to sample action chunks `a ∈ R^{H×d}` from `p(a | observation)`. Flow matching
learns a time-dependent velocity field `v_θ(x, t)` that transports a simple noise
distribution into the data distribution along a probability path. We use the **linear
(optimal-transport) interpolant** between data and Gaussian noise.

**Critical convention warning** (the code says so itself,
[`pi0.py:226`](../src/openpi/models/pi0.py)): π₀ uses the *diffusion* convention where

- `t = 1` is **pure noise**,
- `t = 0` is the **data** (the target actions).

This is the **opposite** of the π₀ paper's text. Always check which end is noise when
reading. In this code:

```
x_t = t · noise + (1 - t) · actions          # interpolant   (pi0.py:199)
u_t = noise - actions                         # target velocity = dx_t/dt  (pi0.py:200)
```

Sanity check the derivative: `d/dt [t·ε + (1-t)·a] = ε - a = u_t`. ✓ The velocity is
**constant along each sample's path** (straight-line / OT interpolant), which is what
makes few-step Euler integration accurate at inference (Module 05).

The network predicts `v_t = v_θ(x_t, t, observation)` and we regress it onto `u_t`:

```
L(θ) = E_{a, ε, t} ‖ v_θ(x_t, t, obs) − (ε − a) ‖²
```

That's it. The "denoiser" learns the straight-line direction from noise to data,
*conditioned on the observation tokens via attention*.

---

## 2. Time sampling: Beta(1.5, 1)

[`pi0.py:197`](../src/openpi/models/pi0.py):
```python
time = jax.random.beta(rng, 1.5, 1, batch_shape) * 0.999 + 0.001
```
Instead of `t ~ Uniform[0,1]`, π₀ draws `t ~ Beta(1.5, 1)`, which puts **more mass near
t→1 (high noise)**, then squashes to `[0.001, 0.999]` to avoid the exact endpoints.

Why bias toward high noise? The high-noise regime is where the velocity field is hardest
to learn and matters most for sample quality — early integration steps from pure noise
set the trajectory's coarse shape. Over-sampling there improves the parts of the path
that need the most modeling capacity. (This is a known flow/diffusion trick; the exact
schedule is an empirical choice.)

---

## 3. `embed_suffix`: actions + time → suffix tokens

[`pi0.py:139`](../src/openpi/models/pi0.py). This is the second of the two
sequence-builders, and it's **where π₀ vs π₀.₅ diverges most**.

### Shared first step
```python
action_tokens = self.action_in_proj(noisy_actions)        # [b, H, d_expert]
time_emb = posemb_sincos(timestep, d_expert, min_period=4e-3, max_period=4.0)
```
- `action_in_proj` ([`pi0.py:92`](../src/openpi/models/pi0.py)): a Linear from
  `action_dim`(32) → action-expert width(1024). One token **per action in the chunk** —
  the suffix has `action_horizon` action tokens, predicted in parallel (not
  autoregressively).
- `posemb_sincos` ([`pi0.py:47`](../src/openpi/models/pi0.py)): sine-cosine embedding of
  the scalar timestep `t`, with periods tuned to the `[0,1]` range
  (`min_period=4e-3 … max_period=4.0`). This is how the network knows "how noisy" its
  input is.

### π₀ path (no adaRMS) — [`pi0.py:170`](../src/openpi/models/pi0.py)
```python
time_tokens = repeat(time_emb, "b d -> b H d")              # broadcast time to every action
x = concat([action_tokens, time_tokens], -1)               # [b, H, 2d]
x = swish(action_time_mlp_in(x)); x = action_time_mlp_out(x)  # mix down to d
action_expert_tokens = x ; adarms_cond = None
```
Time is injected by **concatenation + MLP mixing** into the token features. No
conditioning signal is passed to the transformer norms.

Also, for **π₀ only**, a continuous **state token** is prepended to the suffix
([`pi0.py:151`](../src/openpi/models/pi0.py)) via `state_proj`, with `ar_mask=[True]` so
it starts the suffix block.

### π₀.₅ path (adaRMS) — [`pi0.py:162`](../src/openpi/models/pi0.py)
```python
time_emb = swish(time_mlp_in(time_emb)); time_emb = swish(time_mlp_out(time_emb))
action_expert_tokens = action_tokens          # actions go in UN-mixed with time
adarms_cond = time_emb                          # time is passed as conditioning instead
```
Here the action tokens carry **no** time information directly; instead `adarms_cond` (the
processed time embedding) is handed to the transformer, where **every RMSNorm in the
action expert modulates its scale/shift/gate by the timestep**
([`gemma.py:127`](../src/openpi/models/gemma.py), Module 03/06). This is DiT-style
conditioning and is one of the two defining π₀.₅ changes. There is **no** separate state
token in the π₀.₅ suffix — state lives in the prefix language tokens.

### The suffix masks
[`pi0.py:182`](../src/openpi/models/pi0.py): `ar_mask = [True] + [False]*(H-1)`. The
first action token starts a fresh AR block (so the prefix can't attend forward into
actions), and the H action tokens then attend bidirectionally among themselves —
consistent with predicting the whole chunk jointly.

Return: `(tokens, input_mask, ar_mask, adarms_cond)` — note the 4th element, which is
`None` for π₀ and the time embedding for π₀.₅.

---

## 4. `compute_loss`: one joint forward pass

[`pi0.py:188`](../src/openpi/models/pi0.py). Put it all together:

```python
preprocess_rng, noise_rng, time_rng = split(rng, 3)
obs = preprocess_observation(preprocess_rng, obs, train=train)   # augment + resize

noise = normal(noise_rng, actions.shape)                          # ε
time  = beta(time_rng, 1.5, 1, batch) * 0.999 + 0.001            # t
x_t = time[...,None,None] * noise + (1 - time[...,None,None]) * actions
u_t = noise - actions

prefix = embed_prefix(obs)                                        # images+lang(+state)
suffix = embed_suffix(obs, x_t, time)                             # noisy actions + time
input_mask = concat([prefix_mask, suffix_mask], 1)
ar_mask    = concat([prefix_ar,   suffix_ar],   0)
attn_mask  = make_attn_mask(input_mask, ar_mask)
positions  = cumsum(input_mask, 1) - 1

(prefix_out, suffix_out), _ = self.PaliGemma.llm(
    [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions,
    adarms_cond=[None, adarms_cond])                              # only suffix gets adaRMS

v_t = self.action_out_proj(suffix_out[:, -H:])                    # last H tokens → velocity
return mean((v_t - u_t)**2, axis=-1)                              # per-(sample,step) MSE
```

Points that matter:
- **One forward pass** runs both experts jointly; training does *not* use the KV cache
  (that's an inference-only optimization). Both halves are present so attention is fully
  connected per the mask.
- `adarms_cond=[None, adarms_cond]`: expert 0 (VLM) never gets adaRMS; expert 1 (action)
  gets it for π₀.₅ (and `None` for π₀).
- `action_out_proj` ([`pi0.py:100`](../src/openpi/models/pi0.py)): Linear from expert
  width → `action_dim`, applied to the **last H** suffix outputs (the action tokens;
  for π₀ this correctly skips the leading state token).
- The loss is plain **MSE between predicted and target velocity**, averaged over the
  action-dim, returned per (batch, horizon) so the trainer can mean it
  ([`train.py:150`](../scripts/train.py)).

That `jnp.mean((v_t - u_t)**2)` is the entire training signal for the policy. Everything
else in the repo exists to feed it well-shaped, well-normalized data and to optimize it
at scale.

---

## 5. Why flow matching instead of tokenized actions?

(Contrast with π₀-FAST, [`pi0_fast.py`](../src/openpi/models/pi0_fast.py).)
- **Continuous & high-dimensional**: a 50×32 chunk is one parallel prediction, not
  1600 autoregressive decode steps. Far faster inference.
- **Smooth multimodal distributions**: flow matching represents the full continuous
  action distribution; discretization quantizes it.
- **Trade-off**: FAST (autoregressive tokens) can follow language slightly more crisply
  and reuses the LLM's native decoding; π₀/π₀.₅ trade a little of that for speed and
  motion quality. π₀.₅'s knowledge insulation (Module 06) recovers language-following by
  *also* training a discrete action head during pretraining.

---

## Self-check

1. Write `x_t` and `u_t` and verify `u_t = dx_t/dt`. Which end (`t=0`/`t=1`) is noise here?
2. Why `Beta(1.5,1)` instead of uniform time?
3. Contrast how π₀ vs π₀.₅ inject the timestep into the action expert. Where does the
   state token go in each?
4. Why is there no KV cache in `compute_loss`?
5. The suffix has `ar_mask=[True,False,...,False]`. What two things does that achieve?
6. What exactly is regressed in the loss, and over what is it averaged?

## Lab 04

Open [`labs/lab04_flow_head.py`](labs/lab04_flow_head.py). You will:
1. Implement `posemb_sincos` and check it against `pi0.posemb_sincos`.
2. Implement both `embed_suffix` paths (π₀ and π₀.₅) and verify shapes + that the π₀.₅
   path returns a non-None `adarms_cond` while π₀ returns `None`, and that π₀ prepends a
   state token (so suffix length = H+1) while π₀.₅ does not (length = H).
3. Implement the flow-matching loss `compute_loss` for a toy model and confirm: (a) it's
   a finite scalar, (b) gradients flow to `action_in_proj`/`action_out_proj`, (c) on a
   trivially memorizable single batch the loss drives toward ~0.

Next: [Module 05 — Inference: ODE integration + KV cache](05_inference_flow_and_kvcache.md).
