import jax
import jax.numpy as jnp
import openpi.base.array_typing as at
import openpi.models.model as _model
import openpi.models.common as _common
from functools import partial

from flax.training import common_utils
from typing import Iterator


def fake_init_data_loader(
    model: _model.Model, local_batch_size: int, dp_sharding: jax.sharding.Sharding
) -> Iterator[tuple[_common.Observation, _common.Actions]]:
    """Returns a faked (and infinite) data loader with input spec."""
    observation_spec, action_spec = _model.create_inputs_spec(model, batch_size=local_batch_size)
    print(observation_spec)
    print(action_spec)

    def make_from_spec(spec: jax.ShapeDtypeStruct, rng: at.KeyArrayLike):
        if spec.dtype == jnp.float32:
            return jax.random.uniform(rng, shape=spec.shape, minval=-1.0, maxval=1.0)
        if spec.dtype == jnp.int32:
            return jax.random.randint(rng, shape=spec.shape, minval=0, maxval=2048)
        return jnp.zeros(shape=spec.shape, dtype=spec.dtype)

    def _to_dp_array(local_arr: jax.Array):
        global_shape = (local_arr.shape[0] * jax.process_count(), *local_arr.shape[1:])
        return jax.make_array_from_process_local_data(dp_sharding, local_arr, global_shape)

    def data_loader():
        rng = jax.random.key(0)
        while True:
            rng, data_rng = jax.random.split(rng)
            observation = jax.tree.map(partial(make_from_spec, rng=data_rng), observation_spec)
            action = jax.tree.map(partial(make_from_spec, rng=data_rng), action_spec)
            # Obs(stacked on leaves) -> Obs(global array)
            yield jax.tree.map(_to_dp_array, observation), jax.tree.map(_to_dp_array, action)

    return data_loader()
