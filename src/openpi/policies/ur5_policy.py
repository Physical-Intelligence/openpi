import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_ur5_example() -> dict:
    """Creates a random input example for the UR5 policy.

    The keys and shapes here reflect what the inference environment sends to the policy
    server at runtime (i.e. after the repack transform has already been applied).
    """
    return {
        "joints": np.random.rand(6).astype(np.float32),
        "gripper": np.random.rand(1).astype(np.float32),
        "base_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "wrist_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class UR5Inputs(transforms.DataTransformFn):
    """Converts UR5 environment observations to the format expected by the model.

    This class is used for both training (applied after the repack transform) and
    inference (applied to the dict sent by the robot client). For your own robot,
    copy this class and adjust the keys and dimensions to match your setup.

    The UR5 has 6 revolute joints and a parallel-jaw gripper, giving a 7-dimensional
    state and action space. It is equipped with one base (third-person) camera and one
    wrist camera; there is no second wrist camera, so that image slot is zero-padded.
    """

    # Determines which model will be used. Do not change this for your own robot.
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # Concatenate 6 joint angles and 1 gripper position into a 7-dim state vector.
        # Modify the keys and slicing to match your robot's proprioceptive state.
        state = np.concatenate([data["joints"], data["gripper"]])

        # Possibly need to parse images to uint8 (H, W, C) since LeRobot stores images
        # as float32 (C, H, W). This conversion is a no-op during policy inference, where
        # images already arrive as uint8 (H, W, C).
        # If your robot uses different key names, change "base_rgb" and "wrist_rgb" here.
        base_image = _parse_image(data["base_rgb"])
        wrist_image = _parse_image(data["wrist_rgb"])

        # Do not change the output keys below — these are fixed by the model.
        # Pi0 supports three image slots: one third-person view and two wrist views.
        # If a slot is unused, fill it with zeros and set its mask to False (pi0) or
        # True (pi0-FAST, which attends to all slots regardless of the mask).
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # UR5 has no right wrist camera; pad the slot with zeros.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # Mask the unused slot for pi0; pi0-FAST uses all slots.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Actions are only present during training, not inference.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the language instruction through to the model.
        # The output key must always be "prompt".
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UR5Outputs(transforms.DataTransformFn):
    """Converts model output actions back to the UR5 action space.

    This class is used for inference only. For your own robot, replace ``7`` with
    the number of action dimensions your robot actually uses.
    """

    def __call__(self, data: dict) -> dict:
        # The model produces actions padded to its full internal action_dim.
        # Slice out the first 7 dimensions (6 joint targets + 1 gripper command).
        return {"actions": np.asarray(data["actions"][:, :7])}
