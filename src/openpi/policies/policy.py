from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        return self._infer_batch([obs], noise=noise, batched=False)

    @override
    def infer_batch(self, obs_batch: Sequence[dict], *, noise: np.ndarray | None = None) -> dict:
        obs_batch = _validate_obs_batch(obs_batch)
        return self._infer_batch(obs_batch, noise=noise, batched=True)

    def _infer_batch(self, obs_batch: Sequence[dict], *, noise: np.ndarray | None, batched: bool) -> dict:
        # Make a copy since transformations may modify the inputs in place.
        transformed = [self._input_transform(_copy_tree(obs)) for obs in obs_batch]

        inputs = _stack_trees(transformed)
        batch_size = len(transformed)

        if self._is_pytorch_model:
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device), inputs)
            sample_rng_or_pytorch_device = self._pytorch_device
        else:
            inputs = jax.tree.map(jnp.asarray, inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)

        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            sample_kwargs["noise"] = self._prepare_noise(noise, batch_size=batch_size, batched=batched)

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time

        outputs = _to_numpy_tree(outputs, is_pytorch=self._is_pytorch_model)
        outputs = self._apply_output_transforms_batch(outputs, batch_size=batch_size)
        if not batched:
            outputs = jax.tree.map(lambda x: x[0, ...], outputs)

        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        if batched:
            outputs["policy_timing"]["batch_size"] = batch_size
        return outputs

    def _prepare_noise(self, noise: np.ndarray, *, batch_size: int, batched: bool):
        noise = (
            torch.from_numpy(np.asarray(noise)).to(self._pytorch_device)
            if self._is_pytorch_model
            else jnp.asarray(noise)
        )

        if noise.ndim == 2:
            if batched:
                raise ValueError(
                    "Batched noise must have shape (batch_size, action_horizon, action_dim); "
                    f"got unbatched noise with shape {tuple(noise.shape)}"
                )
            return noise[None, ...]

        if noise.ndim != 3:
            raise ValueError(
                "Noise must have shape (action_horizon, action_dim) for infer or "
                f"(batch_size, action_horizon, action_dim) for infer_batch; got shape {tuple(noise.shape)}"
            )

        if noise.shape[0] != batch_size:
            raise ValueError(f"Noise batch size {noise.shape[0]} does not match observation batch size {batch_size}")

        return noise

    def _apply_output_transforms_batch(self, outputs: dict, *, batch_size: int) -> dict:
        output_batch = [_index_tree(outputs, i) for i in range(batch_size)]
        transformed = [self._output_transform(output) for output in output_batch]
        return _stack_trees(transformed)

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)
        self._record({"inputs": obs, "outputs": results})
        return results

    @override
    def infer_batch(self, obs_batch: Sequence[dict]) -> dict:
        results = self._policy.infer_batch(obs_batch)
        self._record({"inputs": obs_batch, "outputs": results})
        return results

    def _record(self, data: dict) -> None:
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))


def _validate_obs_batch(obs_batch: Sequence[dict]) -> list[dict]:
    if isinstance(obs_batch, (dict, str, bytes)) or not isinstance(obs_batch, Sequence):
        raise TypeError("obs_batch must be a non-empty sequence of observation dictionaries")

    obs_batch = list(obs_batch)
    if not obs_batch:
        raise ValueError("obs_batch must contain at least one observation")
    if not all(isinstance(obs, dict) for obs in obs_batch):
        raise TypeError("obs_batch must contain only observation dictionaries")
    return obs_batch


def _copy_tree(tree):
    return jax.tree.map(lambda x: x.copy() if isinstance(x, np.ndarray) else x, tree)


def _stack_trees(trees: Sequence[dict]) -> dict:
    try:
        return jax.tree.map(_stack_leaves, *trees)
    except Exception as e:
        raise ValueError(
            "Could not stack policy batch; all observations must have matching structure and shapes"
        ) from e


def _stack_leaves(*leaves):
    if isinstance(leaves[0], (str, bytes)):
        raise TypeError("String leaves cannot be batched; tokenize prompts before stacking")
    return np.stack(leaves, axis=0)


def _index_tree(tree: dict, index: int) -> dict:
    return jax.tree.map(lambda x: x[index, ...], tree)


def _to_numpy_tree(tree: dict, *, is_pytorch: bool) -> dict:
    if is_pytorch:
        return jax.tree.map(lambda x: np.asarray(x.detach().cpu()), tree)
    return jax.tree.map(np.asarray, tree)
