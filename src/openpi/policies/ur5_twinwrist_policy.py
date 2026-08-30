"""Transforms for the UR5 + two-axis wrist + gripper platform."""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

ACTION_DIM = 9


def _image(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim != 3:
        raise ValueError(f"expected an image, got {value.shape}")
    if value.shape[0] == 3 and value.shape[-1] != 3:
        value = einops.rearrange(value, "c h w -> h w c")
    if np.issubdtype(value.dtype, np.floating):
        value = np.clip(value * (255.0 if value.max(initial=0) <= 1.0 else 1.0), 0, 255)
    value = value.astype(np.uint8, copy=False)
    if value.shape[-1] != 3:
        raise ValueError(f"expected RGB image, got {value.shape}")
    return value


@dataclasses.dataclass(frozen=True)
class UR5TwinWristInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation.state"], dtype=np.float32)
        if state.shape != (ACTION_DIM,) or not np.isfinite(state).all():
            raise ValueError(f"observation.state must be finite shape (9,), got {state.shape}")
        result = {
            "state": state,
            "image": {
                "base_0_rgb": _image(data["observation.images.front"]),
                "left_wrist_0_rgb": _image(data["observation.images.side"]),
                "right_wrist_0_rgb": _image(data["observation.images.top"]),
            },
            "image_mask": dict.fromkeys(("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"), np.True_),
        }
        if "action" in data:
            actions = np.asarray(data["action"], dtype=np.float32)
            if actions.shape[-1] != ACTION_DIM or not np.isfinite(actions).all():
                raise ValueError(f"action must be finite with final dimension 9, got {actions.shape}")
            result["actions"] = actions
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class UR5TwinWristOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.shape[-1] < ACTION_DIM:
            raise ValueError(f"model returned only {actions.shape[-1]} action dimensions")
        return {"actions": actions[..., :ACTION_DIM]}
