import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


YUANYOU2_STATE_DIM = 14
YUANYOU2_ACTION_DIM = 14

# Yuanyou2 14-dim layout:
# [left_arm_6, left_gripper_1, right_arm_6, right_gripper_1]
LEFT_ARM_SLICE = slice(0, 6)
LEFT_GRIPPER_INDEX = 6
RIGHT_ARM_SLICE = slice(7, 13)
RIGHT_GRIPPER_INDEX = 13


def make_yuanyou2_example() -> dict:
    """Creates a random input example for the Yuanyou2 policy."""
    return {
        "observation/state": np.random.rand(YUANYOU2_STATE_DIM),
        "observation/images/head": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/images/left_wrist": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/images/right_wrist": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        # Only used during training / transform tests.
        # Shape is usually [action_horizon, action_dim].
        "actions": np.random.rand(10, YUANYOU2_ACTION_DIM),
        "prompt": "pick a cube and place it on another cube",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class Yuanyou2Inputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data["observation/images/head"])
        left_wrist_image = _parse_image(data["observation/images/left_wrist"])
        right_wrist_image = _parse_image(data["observation/images/right_wrist"])
        # base_image = _parse_image(data["observation/image"])
        # wrist_image = _parse_image(data["observation/wrist_image"])

        state = np.asarray(data["observation/state"])

        if state.shape[-1] != YUANYOU2_STATE_DIM:
            raise ValueError(
                f"Expected observation/state dim {YUANYOU2_STATE_DIM}, got {state.shape[-1]}. "
                "Yuanyou2 state layout should be: "
                "[left_arm_6, left_gripper, right_arm_6, right_gripper]."
            )

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_,  # if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        # if "actions" in data:
        #     inputs["actions"] = data["actions"]
        if "actions" in data:
            actions = np.asarray(data["actions"])

            if actions.shape[-1] != YUANYOU2_ACTION_DIM:
                raise ValueError(
                    f"Expected actions dim {YUANYOU2_ACTION_DIM}, got {actions.shape[-1]}. "
                    "Yuanyou2 action layout should be: "
                    "[left_arm_6, left_gripper, right_arm_6, right_gripper]."
                )

            inputs["actions"] = actions

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs
    
@dataclasses.dataclass(frozen=True)
class Yuanyou2Outputs(transforms.DataTransformFn):
    """
    Converts model outputs back to Yuanyou2 action format.

    Yuanyou2 action layout:
    [left_arm_6, left_gripper, right_arm_6, right_gripper]
    """

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, :YUANYOU2_ACTION_DIM])

        if actions.shape[-1] != YUANYOU2_ACTION_DIM:
            raise ValueError(
                f"Expected output action dim {YUANYOU2_ACTION_DIM}, got {actions.shape[-1]}"
            )

        return {"actions": actions}
    
    
def decode_yuanyou2_action(action_14: np.ndarray) -> dict:
    """
    Decode one Yuanyou2 14-dim action into arm and gripper commands.

    Input layout:
    [left_arm_6, left_gripper, right_arm_6, right_gripper]

    Returns:
        {
            "left_arm": np.ndarray, shape (6,),
            "left_gripper": float,
            "right_arm": np.ndarray, shape (6,),
            "right_gripper": float,
        }
    """
    action_14 = np.asarray(action_14)

    if action_14.shape[-1] != YUANYOU2_ACTION_DIM:
        raise ValueError(
            f"Expected action dim {YUANYOU2_ACTION_DIM}, got {action_14.shape[-1]}"
        )

    return {
        "left_arm": action_14[LEFT_ARM_SLICE],
        "left_gripper": float(action_14[LEFT_GRIPPER_INDEX]),
        "right_arm": action_14[RIGHT_ARM_SLICE],
        "right_gripper": float(action_14[RIGHT_GRIPPER_INDEX]),
    }
# @dataclasses.dataclass(frozen=True)
# class Yuanyou2Outputs(transforms.DataTransformFn):
#     """
#     This class is used to convert outputs from the model back the the dataset specific format. It is
#     used for inference only.

#     For your own dataset, you can copy this class and modify the action dimension based on the comments below.
#     """

#     def __call__(self, data: dict) -> dict:
#         # Only return the first N actions -- since we padded actions above to fit the model action
#         # dimension, we need to now parse out the correct number of actions in the return dict.
#         # For Libero, we only return the first 14 actions (since the rest is padding).
#         # For your own dataset, replace `14` with the action dimension of your dataset.
#         return {"actions": np.asarray(data["actions"][:, :YUANYOU2_ACTION_DIM])}
