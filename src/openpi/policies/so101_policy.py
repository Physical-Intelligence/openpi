"""Data transforms for SO101 single-arm robot (LeRobot v3 format).

SO101 is a 5-DOF arm + gripper = 6D joint-space:
  - observation.state: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper] (6D)
  - action: same 6D joint positions
  - observation.images.front, observation.images.wrist
"""

import dataclasses

import numpy as np

from openpi import transforms


_SO101_ACTION_DIM = 6


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        import einops

        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _get_key(data: dict, *keys: str):
    for key in keys:
        if key in data:
            return data[key]
    raise KeyError(f"Missing keys: {keys}")


@dataclasses.dataclass(frozen=True)
class SO101Inputs(transforms.DataTransformFn):
    """Transforms SO101 dataset observations into model input format.

    Handles both training (LeRobot keys with dots/slashes) and inference
    (keys as sent by the robot driver).
    """

    default_prompt: str = "stack the rings"

    def __call__(self, data: dict) -> dict:
        front = _parse_image(
            _get_key(data, "observation.images.front", "observation/images/front", "image", "images.front")
        )
        try:
            wrist = _parse_image(
                _get_key(data, "observation.images.wrist", "observation/images/wrist", "wrist_image", "images.wrist")
            )
        except KeyError:
            wrist = np.zeros_like(front)

        state = np.asarray(
            _get_key(data, "observation.state", "observation/state", "state"),
            dtype=np.float32,
        )

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": front,
                "left_wrist_0_rgb": wrist,
                "right_wrist_0_rgb": np.zeros_like(front),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        if "prompt" in data:
            prompt = data["prompt"]
            inputs["prompt"] = prompt.decode("utf-8") if isinstance(prompt, bytes) else prompt
        elif "task" in data:
            task = data["task"]
            inputs["prompt"] = task.decode("utf-8") if isinstance(task, bytes) else str(task)
        else:
            inputs["prompt"] = self.default_prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class SO101Outputs(transforms.DataTransformFn):
    """Slices model output back to SO101's native 6D action space."""

    action_dim: int = _SO101_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        return {"actions": actions[:, : self.action_dim].astype(np.float32)}
