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
            self._staged_inference = hasattr(model, "_stage1_token_prep")
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)
            # Staged inference for per-stage timing (each stage separately JIT-compiled)
            if hasattr(model, "_stage1_embed_prefix"):
                self._jit_stage1 = nnx_utils.module_jit(model._stage1_embed_prefix)
                self._jit_stage2 = nnx_utils.module_jit(model._stage2_fill_kv_cache)
                self._jit_stage3 = nnx_utils.module_jit(model._stage3_denoise)
                self._staged_inference = True
            else:
                self._staged_inference = False

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)

        if self._staged_inference and self._is_pytorch_model:
            # --- Staged PyTorch inference with per-stage timing ---
            def _sync():
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            t_total = time.monotonic()

            t0 = time.monotonic()
            with torch.no_grad():
                state, prefix_embs, prefix_pad_masks, prefix_att_2d_masks_4d, prefix_position_ids = self._model._stage1_token_prep(observation)
            _sync()
            token_prep_ms = (time.monotonic() - t0) * 1000

            t1 = time.monotonic()
            with torch.no_grad():
                past_key_values = self._model._stage2_llm_backbone(prefix_embs, prefix_pad_masks, prefix_att_2d_masks_4d, prefix_position_ids)
            _sync()
            llm_backbone_ms = (time.monotonic() - t1) * 1000

            t2 = time.monotonic()
            step_noise = sample_kwargs.get("noise", None)
            if step_noise is None:
                bsize = observation.state.shape[0]
                actions_shape = (bsize, self._model.config.action_horizon, self._model.config.action_dim)
                step_noise = self._model.sample_noise(actions_shape, sample_rng_or_pytorch_device)
            with torch.no_grad():
                actions = self._model._stage3_action_expert(state, prefix_pad_masks, past_key_values, step_noise)
            _sync()
            action_expert_ms = (time.monotonic() - t2) * 1000

            total_ms = (time.monotonic() - t_total) * 1000

            outputs = {
                "state": inputs["state"],
                "actions": actions,
            }
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
            outputs = self._output_transform(outputs)
            outputs["policy_timing"] = {"infer_ms": total_ms}
            outputs["stage_timing"] = {
                "token_prep_ms": token_prep_ms,
                "llm_backbone_ms": llm_backbone_ms,
                "action_expert_ms": action_expert_ms,
                "total_ms": total_ms,
            }
        elif self._staged_inference:
            # --- Staged JAX inference with per-stage timing ---
            t_total = time.monotonic()

            t0 = time.monotonic()
            stage1_out = self._jit_stage1(observation)
            jax.block_until_ready(stage1_out)
            token_prep_ms = (time.monotonic() - t0) * 1000

            obs_prep, prefix_tokens, prefix_mask, prefix_attn_mask, positions = stage1_out

            t1 = time.monotonic()
            kv_cache = self._jit_stage2(prefix_tokens, prefix_attn_mask, positions)
            jax.block_until_ready(kv_cache)
            llm_backbone_ms = (time.monotonic() - t1) * 1000

            t2 = time.monotonic()
            step_noise = sample_kwargs.get("noise", None)
            if step_noise is None:
                batch_size = observation.state.shape[0]
                step_noise = jax.random.normal(
                    sample_rng_or_pytorch_device,
                    (batch_size, self._model.action_horizon, self._model.action_dim),
                )
            actions = self._jit_stage3(obs_prep, step_noise, prefix_mask, kv_cache)
            jax.block_until_ready(actions)
            action_expert_ms = (time.monotonic() - t2) * 1000

            total_ms = (time.monotonic() - t_total) * 1000

            outputs = {
                "state": inputs["state"],
                "actions": actions,
            }
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
            outputs = self._output_transform(outputs)
            outputs["policy_timing"] = {"infer_ms": total_ms}
            outputs["stage_timing"] = {
                "token_prep_ms": token_prep_ms,
                "llm_backbone_ms": llm_backbone_ms,
                "action_expert_ms": action_expert_ms,
                "total_ms": total_ms,
            }
        else:
            # --- Original inference path (unchanged) ---
            start_time = time.monotonic()
            outputs = {
                "state": inputs["state"],
                "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
            }
            model_time = time.monotonic() - start_time
            if self._is_pytorch_model:
                outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
            else:
                outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
            outputs = self._output_transform(outputs)
            outputs["policy_timing"] = {"infer_ms": model_time * 1000}

        return outputs

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

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
