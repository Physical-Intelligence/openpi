# Module 05 — Inference: ODE Integration + the KV Cache

Training learns the velocity field; inference *integrates* it from noise to actions. The
engineering trick that makes this real-time is caching the expensive prefix once and
re-running only the tiny action expert per step. Read
[`sample_actions` at `pi0.py:216`](../src/openpi/models/pi0.py).

## Learning objectives

- Implement sampling as Euler integration of the learned ODE from `t=1` (noise) to `t=0`.
- Explain and implement the prefix KV cache and the suffix-only forward passes.
- Build the three attention masks needed for cached generation and get their shapes right.
- Reason about the speed/accuracy trade-off of `num_steps`.

---

## 1. Sampling = solving an ODE

Given the trained velocity field `v_θ(x, t, obs)`, generating an action chunk means
solving the ODE `dx/dt = v_θ(x, t)` from `t=1` (noise) down to `t=0` (actions). π₀ uses
plain **forward Euler** with a fixed step:

```python
dt = -1.0 / num_steps                  # negative: we integrate 1 → 0   (pi0.py:228)
x_t = noise                            # start at pure noise
for _ in range(num_steps):
    v_t = v_θ(x_t, t, obs)
    x_t = x_t + dt * v_t               # Euler step                      (pi0.py:271)
    t   = t + dt
return x_t                             # ≈ actions at t≈0
```

In the code this is a `jax.lax.while_loop` ([`pi0.py:273`](../src/openpi/models/pi0.py))
with `cond: t >= -dt/2` (a float-robust "stop at 0"). `num_steps` defaults to **10**
([`pi0.py:221`](../src/openpi/models/pi0.py)) — ten network evaluations to produce a full
action chunk.

Why so few steps suffice: the OT/linear interpolant from Module 04 makes the true paths
nearly straight, so Euler with a handful of steps is accurate. This is a major practical
advantage of flow matching over many-step diffusion samplers for control.

> Note the sign convention again: `dt < 0` because this code integrates from `t=1`
> (noise) to `t=0` (data). If you flip the convention you flip the sign.

---

## 2. The KV cache: compute the prefix once

The prefix (hundreds of image + language tokens through the **2B** VLM) is **constant**
across all integration steps — it depends only on the observation, not on `x_t` or `t`.
Recomputing it 10× would dominate cost. So:

**Step A — fill the cache** ([`pi0.py:233`](../src/openpi/models/pi0.py)):
```python
prefix_tokens, prefix_mask, prefix_ar = embed_prefix(obs)
prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar)
positions = cumsum(prefix_mask, 1) - 1
_, kv_cache = self.PaliGemma.llm([prefix_tokens, None],          # suffix expert = None
                                 mask=prefix_attn_mask, positions=positions)
```
Running with `[prefix, None]` executes **only expert 0** over the prefix and returns its
per-layer keys and values (`KVCache`, [`gemma.py:336`](../src/openpi/models/gemma.py)).
No action expert, no action tokens yet.

**Step B — per integration step, run only the suffix** ([`pi0.py:239`](../src/openpi/models/pi0.py)):
```python
def step(carry):
    x_t, t = carry
    suffix_tokens, suffix_mask, suffix_ar, adarms_cond = embed_suffix(obs, x_t, broadcast(t))
    (_, suffix_out), _ = self.PaliGemma.llm(
        [None, suffix_tokens],            # prefix expert = None; use cache instead
        mask=full_attn_mask, positions=positions,
        kv_cache=kv_cache, adarms_cond=[None, adarms_cond])
    v_t = action_out_proj(suffix_out[:, -H:])
    return x_t + dt * v_t, t + dt
```
`[None, suffix_tokens]` runs **only expert 1** (the 300M action expert) over the H action
tokens, while attention reads the cached prefix K/V. This is the payoff of the
architecture from Module 03: the big network runs once; each of the 10 steps is a cheap
small-expert pass.

> **This is why π₀ can control a robot at high frequency.** Per-step cost ≈ one tiny
> action-expert forward over ~H tokens attending to a cached prefix — not a full
> VLM forward.

---

## 3. Three masks for cached generation

When you query with only the suffix but attend to prefix + suffix keys, you must build
the attention mask by hand ([`pi0.py:244`](../src/openpi/models/pi0.py)):

1. **suffix→suffix** `[b, S_suf, S_suf]`: from `make_attn_mask(suffix_mask, suffix_ar)` —
   how action tokens attend to each other.
2. **suffix→prefix** `[b, S_suf, S_pre]`: `repeat(prefix_mask, "b p -> b s p")` — action
   tokens may attend to every *real* prefix token (bidirectional into the prefix).
3. **combined** `[b, S_suf, S_pre + S_suf]`: concat of the two along the key axis
   ([`pi0.py:252`](../src/openpi/models/pi0.py)). The asserted shape check at
   [`pi0.py:253`](../src/openpi/models/pi0.py) is worth keeping in your own code — mask
   shape bugs are the #1 source of silent garbage here.

**Positions** for the suffix continue *after* the prefix
([`pi0.py:259`](../src/openpi/models/pi0.py)):
```python
positions = sum(prefix_mask, -1)[:, None] + cumsum(suffix_mask, -1) - 1
```
i.e., start counting from however many real prefix tokens there were, so RoPE sees the
suffix as genuinely following the prefix.

Notice `adarms_cond` is recomputed **every step** (it depends on `t`), so for π₀.₅ the
timestep is re-injected through adaRMS at each integration step — exactly mirroring
training.

---

## 4. From `sample_actions` to a deployable policy

`sample_actions` returns *normalized* actions in model space. The serving layer
(`policies/policy.py`, `Policy.infer`) wraps it: apply input transforms → tokenize →
`sample_actions` → `Unnormalize` → robot-specific output transform → action chunk in
physical units. The websocket server (`serving/websocket_policy_server.py`) streams
these to the robot. You don't need to reimplement serving for the capstone, but know that
**the model's contract ends at normalized actions**; the transform shell (Module 01)
handles the rest.

`num_steps` is your speed/quality dial: more steps = more accurate ODE solve = slower.
10 is the default sweet spot for these straight-ish paths; try 1, 4, 10, 50 in the lab
and watch the action error vs. a long-integration reference.

---

## Self-check

1. Why is `dt` negative, and what are the start/end values of `t`?
2. Precisely which tokens/experts run in Step A vs. Step B, and what is reused between
   steps?
3. Why can the prefix be cached but the suffix cannot?
4. Give the shapes of the three attention masks and how `positions` is computed for the
   suffix.
5. Why do ~10 Euler steps suffice here when image-diffusion often needs 50–1000?
6. For π₀.₅, what gets recomputed each step and why?

## Lab 05

Open [`labs/lab05_sampling.py`](labs/lab05_sampling.py). You will:
1. Implement `sample_actions` *without* a KV cache first (recompute prefix each step) —
   simplest correct version — and confirm it returns `[b, H, action_dim]`.
2. Add the prefix KV cache and the three-mask suffix-only path; assert your cached
   result matches the no-cache result to within float tolerance.
3. Sweep `num_steps ∈ {1,2,4,10,50}` and plot action error vs. a `num_steps=200`
   reference to see the accuracy/speed trade-off.
4. Compare end-to-end against the real `Pi0.sample_actions` on a `dummy` model + fixed
   noise (pass `noise=` explicitly so both integrate the same path).

Next: [Module 06 — π₀.₅ specifics & knowledge insulation](06_pi05_and_knowledge_insulation.md).
