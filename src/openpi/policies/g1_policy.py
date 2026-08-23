import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    if image.ndim != 3:
        raise ValueError(f"Expected one HWC or CHW head image, got shape {image.shape}")
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.shape[-1] != 3:
        raise ValueError(f"Expected an RGB head image, got shape {image.shape}")
    return image


@dataclasses.dataclass(frozen=True)
class G1Inputs(transforms.DataTransformFn):
    """Map one G1 head camera and low-dimensional state into pi0.5 slots."""

    model_type: _model.ModelType
    state_dim: int = 29
    task_action_dim: int = 21

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"], dtype=np.float32)
        head_image = _parse_image(data["head_image"])

        if state.shape != (self.state_dim,):
            raise ValueError(f"Expected {self.state_dim} G1 state values, got shape {state.shape}")
        if not np.isfinite(state).all():
            raise ValueError("G1 state contains non-finite values")

        if self.model_type not in (_model.ModelType.PI0, _model.ModelType.PI05):
            raise ValueError(f"Unsupported G1 model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": head_image,
                "left_wrist_0_rgb": np.zeros_like(head_image),
                "right_wrist_0_rgb": np.zeros_like(head_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1] != self.task_action_dim:
                raise ValueError(f"Expected {self.task_action_dim} G1 task actions, got shape {actions.shape}")
            if not np.isfinite(actions).all():
                raise ValueError("G1 actions contain non-finite values")
            inputs["actions"] = actions

        if "prompt" in data:
            prompt = data["prompt"]
            inputs["prompt"] = prompt.decode("utf-8") if isinstance(prompt, bytes) else prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class G1Outputs(transforms.DataTransformFn):
    task_action_dim: int = 21

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., : self.task_action_dim]}
