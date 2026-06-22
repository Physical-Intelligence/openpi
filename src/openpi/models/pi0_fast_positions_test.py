# src/openpi/models/pi0_fast_positions_test.py
import jax.numpy as jnp


def compute_positions(prefill_len, step):
    # mirrors patched logic in pi0_fast.py:
    prefill_len = jnp.asarray(prefill_len)  # shape [B]
    return (prefill_len[:, None] + step).tolist()  # first next token at L


def test_next_token_is_contiguous_zero_indexed():
    # If prefix tokens are 0..L-1, next must be L
    assert compute_positions([3, 5], 0) == [[3], [5]]
    assert compute_positions([0, 1], 0) == [[0], [1]]
    # advancing decode step should increment positions
    assert compute_positions([3], 2) == [[5]]
