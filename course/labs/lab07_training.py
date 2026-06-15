"""Lab 07 — Training at scale (in miniature).

Build the optimizer, a minimal train_step with EMA, a freeze filter that trains only the
action expert, and overfit a single batch.

Run:  uv run python course/labs/lab07_training.py
"""

import jax
import jax.numpy as jnp
import optax


# ----------------------------------------------------------------------------
# Part 1: optimizer matching openpi defaults (Module 07 §2)
# ----------------------------------------------------------------------------
def make_optimizer(num_steps: int = 30_000):
    """AdamW(b1=0.9, b2=0.95, eps=1e-8, wd=1e-10) wrapped with global-norm clip 1.0,
    cosine warmup schedule: warmup 1000 -> peak 2.5e-5 -> decay to 2.5e-6.
    Mirror optimizer.py. Return an optax.GradientTransformation."""
    # TODO(you): build the schedule and the optax.chain(clip, adamw).
    raise NotImplementedError


def check_part1():
    tx = make_optimizer()
    params = {"w": jnp.ones((4, 4))}
    state = tx.init(params)
    grads = {"w": jnp.ones((4, 4)) * 100.0}  # huge -> should be clipped
    updates, _ = tx.update(grads, state, params)
    gnorm = optax.global_norm(updates)
    # With clip 1.0 and a tiny warmup LR, the update norm should be very small.
    assert gnorm < 1.0, gnorm
    print("Part 1 OK: optimizer builds, clips, and uses a small warmup LR.")


# ----------------------------------------------------------------------------
# Part 2: minimal train_step with EMA (Module 07 §1)
# ----------------------------------------------------------------------------
# Implement train_step(params, ema_params, opt_state, tx, batch) -> new state + loss,
# where loss_fn = mean(model.compute_loss(...)) for YOUR mini model. Update EMA as
# ema = decay*ema + (1-decay)*new_params with decay=0.99.

def check_part2():
    print("Part 2: wire your mini model's compute_loss into a train_step + EMA update.")


# ----------------------------------------------------------------------------
# Part 3: freeze filter — train only the action expert (Module 07 §3)
# ----------------------------------------------------------------------------
# Using optax.multi_transform (or a mask), make the VLM/prefix params have a zero
# (set_to_zero) update while the action-expert params train. Assert prefix params are
# unchanged after one step.

def check_part3():
    print("Part 3: implement a freeze mask; assert VLM params get zero update.")


# ----------------------------------------------------------------------------
# Part 4: overfit one batch (Module 07 lab goal)
# ----------------------------------------------------------------------------
def check_part4():
    print("Part 4: overfit a single synthetic (obs, actions) batch for ~300 steps; "
          "the flow loss should fall toward ~0. This is your first VLA training run.")


if __name__ == "__main__":
    check_part1()
    check_part2()
    check_part3()
    check_part4()
