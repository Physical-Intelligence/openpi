from __future__ import annotations

import logging
import math
from typing import Any

import jax
import numpy as np
import torch
from openpi_client import base_policy as _base_policy

from openpi.collect.data_collector import EpisodeDataCollector, InferenceEmbeddings

logger = logging.getLogger(__name__)


def _find_inner_model(policy: Any) -> Any:
    """Walk wrapper chain until a PyTorch model with hook targets is found."""
    obj = policy
    while obj is not None:
        if hasattr(obj, "_model"):
            model = obj._model
            required_attrs = ("action_in_proj", "action_out_proj", "paligemma_with_expert")
            if not all(hasattr(model, attr) for attr in required_attrs):
                raise ValueError("CollectionPolicy only supports the PyTorch PI0Pytorch inference path.")
            return model
        obj = getattr(obj, "_policy", None)
    raise ValueError("CollectionPolicy: cannot find _model in policy chain.")


def _find_wrapped_attr(policy: Any, attr_name: str) -> Any:
    """Walk wrapper chain until an object exposing attr_name is found."""
    obj = policy
    while obj is not None:
        if hasattr(obj, attr_name):
            return getattr(obj, attr_name)
        obj = getattr(obj, "_policy", None)
    raise ValueError(f"CollectionPolicy: cannot find {attr_name!r} in policy chain.")


class CollectionPolicy(_base_policy.BasePolicy):
    """Observer wrapper that records embeddings while preserving the normal infer path."""

    def __init__(self, policy: Any, collector: EpisodeDataCollector) -> None:
        self._policy = policy
        self._collector = collector
        self._collecting = False
        self._inner_model = _find_inner_model(policy)
        self._input_transform = _find_wrapped_attr(policy, "_input_transform")

    @property
    def metadata(self) -> dict[str, Any]:
        return self._policy.metadata

    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
        infer_kwargs = {"noise": noise} if noise is not None else {}
        if not self._collecting:
            return self._policy.infer(obs, **infer_kwargs)

        vision_captures: list[torch.Tensor] = []
        lang_capture: list[torch.Tensor | None] = [None]
        action_in_captures: list[torch.Tensor] = []
        action_out_captures: list[torch.Tensor] = []

        def _vision_hook(module, inp, out):
            del module, inp
            vision_captures.append(out.detach().clone())

        def _lang_hook(module, inp, out):
            del module, inp
            lang_capture[0] = (out * math.sqrt(out.shape[-1])).detach().clone()

        def _action_in_hook(module, inp, out):
            del module, out
            action_in_captures.append(inp[0].detach().clone())

        def _action_out_hook(module, inp, out):
            del module, inp
            action_out_captures.append(out.detach().clone())

        model = self._inner_model
        handles = [
            model.paligemma_with_expert.paligemma.multi_modal_projector.register_forward_hook(_vision_hook),
            model.paligemma_with_expert.paligemma.language_model.embed_tokens.register_forward_hook(_lang_hook),
            model.action_in_proj.register_forward_hook(_action_in_hook),
            model.action_out_proj.register_forward_hook(_action_out_hook),
        ]
        robot_state_np = self._extract_robot_state(obs)
        try:
            result = self._policy.infer(obs, **infer_kwargs)
        finally:
            for handle in handles:
                handle.remove()

        try:
            self._record(robot_state_np, vision_captures, lang_capture[0], action_in_captures, action_out_captures)
        except Exception:
            logger.exception("CollectionPolicy: failed to record inference embeddings; skipping step.")

        return result

    def on_episode_start(self, experiment: str, task: str, episode_id: int) -> None:
        self._collector.on_episode_start(experiment, task, episode_id)
        self._collecting = True

    def on_episode_end(self, success: bool) -> None:
        self._collector.on_episode_end(success)
        self._collecting = False

    def on_task_begin(self) -> None:
        if hasattr(self._policy, "on_task_begin"):
            self._policy.on_task_begin()

    def on_task_end(self) -> None:
        if self._collecting and self._collector.has_pending_data():
            logger.warning("CollectionPolicy: connection closed mid-episode, flushing partial data.")
            self._collector.on_episode_end(success=False)
            self._collecting = False
        if hasattr(self._policy, "on_task_end"):
            self._policy.on_task_end()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._policy, name)

    def _record(
        self,
        robot_state_np: np.ndarray,
        vision_captures: list[torch.Tensor],
        lang_emb: torch.Tensor | None,
        action_in_captures: list[torch.Tensor],
        action_out_captures: list[torch.Tensor],
    ) -> None:
        vision_embs = [vision.squeeze(0).cpu().to(torch.float16).numpy() for vision in vision_captures]

        if lang_emb is None:
            raise RuntimeError("CollectionPolicy: embed_tokens hook did not fire.")
        prompt_emb_np = lang_emb.squeeze(0).cpu().to(torch.float16).numpy()

        num_steps = len(action_in_captures)
        if num_steps < 2 or len(action_out_captures) != num_steps:
            raise RuntimeError(
                "CollectionPolicy: expected equal action hook counts >= 2, got "
                f"action_in={num_steps}, action_out={len(action_out_captures)}."
            )
        dt = -1.0 / num_steps
        noise_action_steps = [
            action_in_captures[i].squeeze(0).cpu().numpy().astype(np.float32) for i in range(1, num_steps)
        ]

        x_last = action_in_captures[-1].squeeze(0).cpu().float()
        v_last = action_out_captures[-1].squeeze(0).cpu().float()
        clean_action_np = (x_last + dt * v_last).numpy().astype(np.float32)

        self._collector.record_inference(
            InferenceEmbeddings(
                vision_embs=vision_embs,
                prompt_emb=prompt_emb_np,
                robot_state=robot_state_np,
                noise_action_steps=noise_action_steps,
                clean_action=clean_action_np,
            )
        )

    def _extract_robot_state(self, obs: dict) -> np.ndarray:
        """Apply the wrapped policy's input transform and read normalized state."""
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if "state" not in inputs:
            raise RuntimeError("CollectionPolicy: transformed inputs are missing the 'state' field.")
        return np.asarray(inputs["state"], dtype=np.float32).flatten()
