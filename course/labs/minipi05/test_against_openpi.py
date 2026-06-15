"""Capstone cross-checks: your minipi05 components vs. the real openpi.

These tests assert *behavioral* equivalence on shared sub-functions (you won't match
weights). Implement the imported names in your minipi05 package, then run:

    uv run python course/labs/minipi05/test_against_openpi.py

Each check is independent; comment out the ones whose component you haven't built yet.
"""

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import pi0 as _pi0


def test_make_attn_mask():
    from masks import make_attn_mask  # your implementation
    input_mask = jnp.array([[True, True, True, False]])
    mask_ar = jnp.array([[False, False, True, False]])
    assert jnp.array_equal(
        make_attn_mask(input_mask, mask_ar), _pi0.make_attn_mask(input_mask, mask_ar)
    )
    print("OK make_attn_mask == openpi")


def test_posemb():
    from model import posemb_sincos  # your implementation
    t = jnp.linspace(0.001, 0.999, 8)
    assert jnp.allclose(
        posemb_sincos(t, 64), _pi0.posemb_sincos(t, 64, min_period=4e-3, max_period=4.0),
        atol=1e-5,
    )
    print("OK posemb_sincos == openpi")


def test_state_binning():
    from tokenizer import discretize_state  # your implementation
    state = np.array([-1.0, 0.0, 0.999], dtype=np.float32)
    ref = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
    assert np.array_equal(discretize_state(state), ref)
    print("OK discretize_state == openpi convention")


def test_sample_shapes_and_cache():
    # Build your MiniPi05, run sample_actions with a fixed noise, and assert:
    #   - output shape [b, action_horizon, action_dim]
    #   - cached and uncached integration agree to fp tolerance
    print("TODO: assert sample shape + cached==uncached for your MiniPi05")


def test_pi05_vs_pi0_suffix():
    # Build MiniPi05 with pi05=True and pi05=False; assert:
    #   pi05=True  -> suffix length == action_horizon, adarms_cond is not None
    #   pi05=False -> suffix length == action_horizon + 1, adarms_cond is None
    print("TODO: assert pi0 vs pi0.5 suffix differences for your MiniPi05")


if __name__ == "__main__":
    # Run the ones you've implemented:
    test_make_attn_mask()
    test_posemb()
    test_state_binning()
    test_sample_shapes_and_cache()
    test_pi05_vs_pi0_suffix()
    print("\nAll implemented checks passed. Finish the TODOs to complete the capstone.")
