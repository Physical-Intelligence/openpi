import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi_client import image_tools


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class UR5Inputs(transforms.DataTransformFn):
    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        mask_padding = self.model_type == _model.ModelType.PI0

        # RepackTransform should produce these keys
        #   see: LeRobotUR5DataConfig in `openpi/training/config.py`
        state_key = "state"
        actions_key = "actions"
        image_key = "base_camera"
        wrist_image_key = "wrist_camera"

        assert state_key in data, f"'{state_key}' must be in data after repack transform"
        assert actions_key in data, f"'{actions_key}' must be in data after repack transform"
        assert image_key in data, f"'{image_key}' must be in data after repack transform"
        assert wrist_image_key in data, f"'{wrist_image_key}' must be in data after repack transform"

        # First, concatenate the joints and gripper into the state vector.
        # Pad to the expected input dimensionality of the model (same as action_dim).
        # TODO: split joint positions and gripper position?
        # state = np.concatenate([data["joints"], data["gripper"]])
        # state = transforms.pad_to_dim(state, self.action_dim)
        state = transforms.pad_to_dim(data[state_key], self.action_dim)

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        base_image = _parse_image(data[image_key])
        wrist_image = _parse_image(data[wrist_image_key])

        base_image = image_tools.resize_with_pad(base_image, 224, 224)
        wrist_image = image_tools.resize_with_pad(wrist_image, 224, 224)

        # Create inputs dict.
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Since there is no right wrist, replace with zeros
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # Since the "slot" for the right wrist is not used, this mask is set
                # to False
                "right_wrist_0_rgb": np.False_ if mask_padding else np.True_,
            },
        }

        # Pad actions to the model action dimension.
        if actions_key in data:
            # The robot produces 7D actions (6 DoF + 1 gripper), and we pad these.
            actions = transforms.pad_to_dim(data[actions_key], self.action_dim)
            inputs["actions"] = actions

        # Pass the prompt (aka language instruction) to the model.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UR5Outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Since the robot has 7 action dimensions (6 DoF + gripper), return the first 7 dims
        return {"actions": np.asarray(data["actions"][:, :7])}
