import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_so101_example() -> dict:
    """Creates a random input example for the SO-101 policy."""
    return {
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.rand(6),  # 5 joints + 1 gripper
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:  # CHW -> HWC
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class SO101Inputs(transforms.DataTransformFn):
    """Input transforms for SO-101 single arm robot.

    Expected inputs:
    - state: [6] (5 joints + 1 gripper)
    - images: dict with wrist camera
    - actions: [action_horizon, 6]
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Parse wrist image
        wrist_image = _parse_image(data["observation/wrist_image"])

        # State: 5 joints + 1 gripper = 6 dims
        state = np.asarray(data["observation/state"])

        # Map to model's expected 3-camera format (pad missing cameras)
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": np.zeros((224, 224, 3), dtype=np.uint8),  # No base camera
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(wrist_image),  # Pad
            },
            "image_mask": {
                "base_0_rgb": np.False_,  # Masked out
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class SO101Outputs(transforms.DataTransformFn):
    """Output transforms - strip padding from actions."""

    def __call__(self, data: dict) -> dict:
        # Return only first 6 dims (5 joints + gripper)
        return {"actions": np.asarray(data["actions"][:, :6])}

