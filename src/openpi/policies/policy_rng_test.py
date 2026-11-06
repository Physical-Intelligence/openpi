from flax import nnx
import jax
import numpy as np
import pytest

from openpi.policies import policy


class _Model(nnx.Module):
    def sample_actions(self, key, observation):
        return jax.random.uniform(key)


@pytest.mark.parametrize("make_key", [jax.random.key, jax.random.PRNGKey])
def test_policy_accepts_explicit_rng(make_key):
    key = make_key(42)
    instance = policy.Policy(_Model(), rng=key)
    np.testing.assert_array_equal(jax.random.key_data(instance._rng), jax.random.key_data(key))  # noqa: SLF001
