from flax import nnx
import jax
import numpy as np
import pytest

from openpi.policies import policy


class _Model(nnx.Module):
    def sample_actions(self, key, observation):
        shape = (observation.state.shape[0], 2, observation.state.shape[-1])
        return observation.state[:, None, :] + jax.random.normal(key, shape)


@pytest.mark.parametrize("make_key", [jax.random.key, jax.random.PRNGKey])
def test_policy_accepts_explicit_rng(make_key):
    key = make_key(42)
    instance = policy.Policy(_Model(), rng=key)
    np.testing.assert_array_equal(jax.random.key_data(instance._rng), jax.random.key_data(key))  # noqa: SLF001


@pytest.mark.parametrize("make_key", [jax.random.key, jax.random.PRNGKey])
def test_policy_infer_replays_seed_and_advances_rng(make_key):
    observation = {"image": {}, "image_mask": {}, "state": np.zeros(3, dtype=np.float32)}
    instance = policy.Policy(_Model(), rng=make_key(42))
    replay = policy.Policy(_Model(), rng=make_key(42))

    first_actions = instance.infer(observation)["actions"]
    np.testing.assert_array_equal(first_actions, replay.infer(observation)["actions"])

    second_actions = instance.infer(observation)["actions"]
    np.testing.assert_array_equal(second_actions, replay.infer(observation)["actions"])
    assert not np.array_equal(first_actions, second_actions)
