import jax.numpy as jnp


def next_positions(prefill_len, step):
    prefill_len = jnp.asarray(prefill_len)  # shape [B]
    return prefill_len[:, None] + step + 1  # current behavior in repo


# toy batch with different prefix lengths
print("current:", next_positions([3, 5], 0).tolist())  # shows [[4],[6]] but should be [[3],[5]]
