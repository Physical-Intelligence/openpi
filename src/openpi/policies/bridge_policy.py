"""
Bridge dataset policy transforms for OpenPI.

Following the third-party open-pi-zero implementation:
- Actions: [x, y, z, roll, pitch, yaw, gripper] (7D end-effector position)
- State: [x, y, z, roll, pitch, yaw, gripper] (7D proprioceptive state with POS_EULER encoding)
- Images: image_0 (primary/static camera), image_1 (secondary camera), NOT wrist
- Normalization: Quantile-based to [-1, 1] range (not z-score)
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_bridge_example() -> dict:
    """Creates a random input example for the Bridge policy."""
    return {
        "observation/image": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "observation/state": np.random.rand(7),
        "prompt": "pick up the object",
    }


def _parse_image(image) -> np.ndarray:
    """Parse image to uint8 (H,W,C) format."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        # Convert from CHW to HWC
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class BridgeInputs(transforms.DataTransformFn):
    """Transform Bridge dataset observations into OpenPI model format.

    Following the third-party open-pi-zero implementation:
    - Uses image_0 (primary/static view) and image_1 (secondary view)
    - image_1 is NOT a wrist camera, it's a secondary static camera
    - Expects 7D proprioceptive state (POS_EULER: xyz + euler angles + gripper)
    """

    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Bridge state is 7D: [x, y, z, roll, pitch, yaw, gripper]
        state = np.asarray(data["observation/state"])

        # Parse images to uint8 (H,W,C) format
        # image_0 = primary/static camera, image_1 = secondary camera (NOT wrist)
        primary_image = _parse_image(data["observation/image"])
        secondary_image = _parse_image(data.get("observation/secondary_image", data.get("observation/wrist_image")))

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                # PI0 uses: base_0_rgb (primary), left_wrist_0_rgb (secondary), right_wrist_0_rgb (unused)
                # Note: "left_wrist_0_rgb" is used for the secondary camera, even though it's not a wrist camera
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (primary_image, secondary_image, np.zeros_like(primary_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                # PI0-FAST expects: base_0_rgb (primary), base_1_rgb (secondary), wrist_0_rgb (unused)
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (primary_image, secondary_image, np.zeros_like(primary_image))
                # We don't mask out padding images for FAST models.
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class BridgeOutputs(transforms.DataTransformFn):
    """Transform model outputs back to Bridge action space.

    Bridge actions are 7D, so we only return the first 7 dimensions.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first 7 dims (Bridge action space)
        return {"actions": np.asarray(data["actions"][:, :7])}
