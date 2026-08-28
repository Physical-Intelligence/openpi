# transforms.py: Input/output transforms for single right-arm ALOHA (WidowX) robots.
# Handles 7-dim state/actions (6 joints + gripper) with 2 cameras (cam_high + cam_right_wrist).

import dataclasses

import einops
import numpy as np

from openpi import transforms


ACTION_DIM = 7  # 6 joints + 1 gripper (right arm)


def make_single_arm_example() -> dict:
    """Creates a random input example for single-arm ALOHA inference."""
    return {
        "state": np.ones((ACTION_DIM,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class AlohaSingleArmInputs(transforms.DataTransformFn):
    """Inputs for single right-arm ALOHA policy.

    Expected inputs (after repack):
    - images: dict with cam_high and cam_right_wrist, each [channel, height, width]
    - state: [7] (6 joints + 1 gripper)
    - actions: [action_horizon, 7] (training only)
    """

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"], dtype=np.float32)

        in_images = data["images"]
        base_image = _convert_image(in_images["cam_high"])

        if "cam_right_wrist" in in_images:
            wrist_image = _convert_image(in_images["cam_right_wrist"])
        else:
            wrist_image = np.zeros_like(base_image)

        # Map to model's 3-image slots following DROID pattern:
        # single wrist camera goes in left_wrist slot, right_wrist slot is zero-padded
        images = {
            "base_0_rgb": base_image,
            "left_wrist_0_rgb": wrist_image,
            "right_wrist_0_rgb": np.zeros_like(base_image),
        }
        image_masks = {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.False_,
        }

        inputs = {
            "state": state,
            "image": images,
            "image_mask": image_masks,
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class AlohaSingleArmOutputs(transforms.DataTransformFn):
    """Outputs for single right-arm ALOHA policy. Slices padded actions back to 7-dim."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :ACTION_DIM])}


def _convert_image(img) -> np.ndarray:
    """Convert image to uint8 HWC format."""
    img = np.asarray(img)
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = einops.rearrange(img, "c h w -> h w c")
    return img
