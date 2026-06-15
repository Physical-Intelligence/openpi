"""Lab 05 — Inference: ODE integration + KV cache.

Implement Euler integration of the learned velocity field, first without a cache, then
with a prefix KV cache, and confirm they agree.

Run:  uv run python course/labs/lab05_sampling.py

This lab is written against YOUR mini model (Lab 03/04 components). The structure mirrors
Pi0.sample_actions (pi0.py:216). Fill in the velocity function with your model.
"""

import jax
import jax.numpy as jnp


def velocity_fn(model, params, obs, x_t, t):
    """Return v_t = v_theta(x_t, t, obs), shape [b, H, action_dim].
    Build suffix from (x_t, t), run the transformer over [prefix, suffix], project the
    last H suffix outputs. (No cache version recomputes the prefix each call.)"""
    # TODO(you): wire to your Lab 02/03/04 components.
    raise NotImplementedError


def sample_actions_nocache(model, params, obs, rng, H, action_dim, num_steps=10):
    """Euler-integrate from t=1 (noise) to t=0 (data). dt = -1/num_steps."""
    # TODO(you):
    #   x = normal(rng, [b, H, action_dim]); t = 1.0; dt = -1/num_steps
    #   for _ in range(num_steps): x = x + dt * velocity_fn(...); t += dt
    #   return x
    raise NotImplementedError


def sample_actions_cached(model, params, obs, rng, H, action_dim, num_steps=10):
    """Same integration, but fill a prefix KV cache ONCE, then per step run only the
    suffix expert attending to the cache. Build the 3 masks from Module 05 §3.
    Assert result matches sample_actions_nocache to fp tolerance."""
    # TODO(you): implement the cached path.
    raise NotImplementedError


def check():
    # TODO(you): once implemented, fix the noise (pass the same rng) so both paths
    # integrate identical trajectories, then:
    #   a = sample_actions_nocache(...); b = sample_actions_cached(...)
    #   assert jnp.allclose(a, b, atol=1e-4)
    #   assert a.shape == (batch, H, action_dim)
    # Then sweep num_steps in {1,2,4,10,50} and compare to a num_steps=200 reference.
    print("Implement the two samplers, assert cached == uncached, then sweep num_steps.")


if __name__ == "__main__":
    check()
