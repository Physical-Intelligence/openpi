"""Lab 04 — The flow-matching action head.

Implement posemb_sincos, both embed_suffix paths, and the flow-matching loss.
Verify against openpi where possible.

Run:  uv run python course/labs/lab04_flow_head.py
"""

import jax
import jax.numpy as jnp

from openpi.models import pi0 as _pi0


# ----------------------------------------------------------------------------
# Part 1: timestep embedding (Module 04 §3)
# ----------------------------------------------------------------------------
def posemb_sincos(pos, embedding_dim, min_period=4e-3, max_period=4.0):
    """Sine-cosine embedding of scalar positions. Match pi0.posemb_sincos."""
    # TODO(you): implement (see pi0.py:47). pos: [b] -> [b, embedding_dim].
    raise NotImplementedError


def check_part1():
    t = jnp.linspace(0.001, 0.999, 8)
    mine = posemb_sincos(t, 64)
    ref = _pi0.posemb_sincos(t, 64, min_period=4e-3, max_period=4.0)
    assert jnp.allclose(mine, ref, atol=1e-5), "must match pi0.posemb_sincos"
    print("Part 1 OK: posemb_sincos matches openpi.")


# ----------------------------------------------------------------------------
# Part 2: the flow interpolant + loss (Module 04 §1, §4)
# ----------------------------------------------------------------------------
def flow_targets(rng, actions):
    """Return (x_t, u_t, time) using the openpi conventions:
        time ~ Beta(1.5, 1) * 0.999 + 0.001          # t=1 noise, t=0 data
        x_t  = t*noise + (1-t)*actions
        u_t  = noise - actions
    """
    # TODO(you): implement. actions: [b, H, d].
    raise NotImplementedError


def check_part2():
    rng = jax.random.key(0)
    actions = jax.random.normal(rng, (4, 10, 32))
    x_t, u_t, t = flow_targets(rng, actions)
    assert x_t.shape == actions.shape and u_t.shape == actions.shape
    assert t.shape == (4,)
    assert jnp.all((t > 0) & (t < 1))
    # At t->0, x_t -> actions; verify the interpolant endpoints make sense:
    noise = u_t + actions  # since u_t = noise - actions
    recon = t[:, None, None] * noise + (1 - t[:, None, None]) * actions
    assert jnp.allclose(recon, x_t, atol=1e-5), "x_t must equal the interpolant"
    print("Part 2 OK: flow interpolant + targets correct.")


# ----------------------------------------------------------------------------
# Part 3: embed_suffix, both paths (Module 04 §3)
# ----------------------------------------------------------------------------
# Implement a function that, given (noisy_actions, time, pi05: bool), returns
#   (suffix_tokens, input_mask, ar_mask, adarms_cond)
# matching Pi0.embed_suffix semantics:
#   - action_in_proj(noisy_actions) -> action tokens
#   - time_emb = posemb_sincos(time, d)
#   - if pi05: time MLP -> adarms_cond; tokens = action tokens; NO state token
#   - else:    prepend a state token (ar=[True]); concat time, MLP-mix; adarms_cond=None
#   - ar_mask over actions = [True] + [False]*(H-1)
#
# Test target: pi05=True -> suffix length H, adarms_cond is not None.
#              pi05=False -> suffix length H+1, adarms_cond is None.

def check_part3():
    print("Part 3: implement embed_suffix(both paths) and assert the length/adarms "
          "differences described above.")


if __name__ == "__main__":
    check_part1()
    check_part2()
    check_part3()
    print("\nCapstone hook: your compute_loss = mean((action_out_proj(suffix_out[:, -H:]) "
          "- u_t)**2). On a single memorizable batch it should drive toward ~0.")
