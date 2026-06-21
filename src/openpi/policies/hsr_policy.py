import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms
from PIL import Image


def make_hsr_example() -> dict:
    """Creates a random input example for the HSR policy."""
    return {
        "head_rgb": np.random.randint(256, size=(640, 480, 3), dtype=np.uint8),
        "hand_rgb": np.random.randint(256, size=(640, 480, 3), dtype=np.uint8),
        "state": np.ones((8,)),
        #  STATE_NAMES = ["arm_lift_joint", "arm_flex_joint", "arm_roll_joint", "wrist_flex_joint", "wrist_roll_joint","hand_motor_joint(gripper)", "head_pan_joint", "head_tilt_joint"]
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class HSRInputs(transforms.DataTransformFn):
    """Inputs for the HSR policy.

    Expected inputs:

    - head_rgb:[H, W, 3]
    - hand_rgb: [H, W, 3]
    - state: [8] #  7 joints (arm 5 + head 2) and 1 gripper
        #  STATE_NAMES = ["arm_lift_joint", "arm_flex_joint", "arm_roll_joint", "wrist_flex_joint", "wrist_roll_joint","hand_motor_joint(gripper)", "head_pan_joint", "head_tilt_joint"]
    - actions: [action_horizon, 11] # Actions are only available during training. 7 joints (arm 5 + head 2) and 1 gripper and 3 twist actions
        #  ACTION_NAMES = ["arm_lift_joint", "arm_flex_joint", "arm_roll_joint", "wrist_flex_joint", "wrist_roll_joint","hand_motor_joint(gripper)", "head_pan_joint", "head_tilt_joint" , "base_x", "base_y", "base_t"]
    """

    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # If true, this will convert the joint and gripper values from the standard HSR space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True
    # If true, apply gripper conversion between HSR and pi0 angular space.
    convert_gripper: bool = False

    def __call__(self, data: dict) -> dict:
        data = _decode_hsr(
            data,
            adapt_to_pi=self.adapt_to_pi,
            convert_gripper=self.convert_gripper,
        )

        # Get the state. We are padding from 14 to the model action dim.
        state = transforms.pad_to_dim(data["state"], self.action_dim)

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": np.zeros_like(
                    data["hand_rgb"]
                ),  # No top-down base camera exists, so this channel is zero-filled.
                "left_wrist_0_rgb": data["hand_rgb"],
                "right_wrist_0_rgb": data["head_rgb"],
            },
            "image_mask": {
                "base_0_rgb": np.False_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            actions = _encode_actions_inv(
                actions,
                adapt_to_pi=self.adapt_to_pi,
                convert_gripper=self.convert_gripper,
            )
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class HSROutputs(transforms.DataTransformFn):
    """Outputs for the HSR policy."""

    # If true, this will convert the joint and gripper values from the standard HSR space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True
    # If true, apply gripper conversion between pi0 angular space and HSR space.
    convert_gripper: bool = False

    def __call__(self, data: dict) -> dict:
        # Only return meaningful actions.
        actions = np.asarray(data["actions"][:, :16])
        # NaN/Inf guard — replace non-finite values with 0 to prevent eval rejection.
        if not np.all(np.isfinite(actions)):
            actions = np.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
        actions = _decode_actions_inv(actions, adapt_to_pi=self.adapt_to_pi)
        return {
            "actions": _encode_actions(
                actions,
                adapt_to_pi=self.adapt_to_pi,
                convert_gripper=self.convert_gripper,
            )
        }


def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def _gripper_to_angular(value):
    # HSR transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with pi0 which is pretrained in
    # angular space.
    #
    # These values are coming from the lite6 OpenParallelGripper:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = (
        _unnormalize(value, min_val=0, max_val=0.032)
        + 0.01844  # TODO: Re-check this offset; current value works in practice.
    )  # Aloha calibration used 0.01844 as the observed minimum.

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (
            2 * horn_radius * linear_position
        )
        return np.arcsin(np.clip(value, -1.0, 1.0))

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return _normalize(value, min_val=0.4, max_val=1.5)


def _gripper_from_angular(value):
    # Convert from the gripper position used by pi0 to the gripper position that is used by lite6 OpenParallelGripper.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = _unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the OpenParallelGripper code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return _normalize(
        value, min_val=1.0, max_val=0.0
    )  # CAUTION: HSR gripper uses open=1.0, close=0.0, so min/max are intentionally reversed.


def _gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = _unnormalize(
        value, min_val=1.0, max_val=0.0
    )  # CAUTION: HSR gripper uses open=1.0, close=0.0, so min/max are intentionally reversed.
    return _normalize(value, min_val=0.4, max_val=1.5)


def _decode_hsr(data: dict, *, adapt_to_pi: bool = False, convert_gripper: bool = False) -> dict:
    # state is ["arm_lift_joint", "arm_flex_joint", "arm_roll_joint", "wrist_flex_joint", "wrist_roll_joint","hand_motor_joint(gripper)", "head_pan_joint", "head_tilt_joint"]
    # dim sizes: [8, 1]

    state = np.asarray(data["state"])
    state = _decode_state(state, adapt_to_pi=adapt_to_pi, convert_gripper=convert_gripper)

    if "actions" in data:
        actions = np.asarray(data["actions"])
        actions = _decode_actions(actions, adapt_to_pi=adapt_to_pi)
        data["actions"] = actions

    def convert_image(img):
        img = np.asarray(img)
        # Convert to uint8 if using float images.
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        # Convert from [channel, height, width] to [height, width, channel].
        if img.shape[0] == 3:
            img = einops.rearrange(img, "c h w -> h w c")

        size = (224, 224)  # Match pi0 input image resolution.
        img = Image.fromarray(img)
        img = img.resize(size, Image.Resampling.BICUBIC)
        return np.array(img)

    image_keys = ["head_rgb", "hand_rgb"]
    for key in image_keys:
        data[key] = convert_image(data[key])
    data["state"] = state

    return data


def _decode_state(
    state: np.ndarray, *, adapt_to_pi: bool = False, convert_gripper: bool = False
) -> np.ndarray:
    if adapt_to_pi:
        # expand state to 14 dimensions
        new_state = np.zeros(shape=(14))
        aligned_ids = [0, 1, 2, 3, 4, 6, 11, 12]
        # state is ["arm_lift_joint", "arm_flex_joint", "arm_roll_joint", "wrist_flex_joint", "wrist_roll_joint", None, "hand_motor_joint(gripper)", None, None, None, None, "head_pan_joint", "head_tilt_joint", None]
        new_state[aligned_ids] = state
        if convert_gripper:
            # Reverse the gripper transformation that is being applied by the HSR runtime.
            new_state[6] = _gripper_to_angular(new_state[6])

        return new_state

    return state


def _decode_actions(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # expand actions to 16 dimensions
        new_actions = np.zeros(shape=(actions.shape[0], 16))
        aligned_ids = [0, 1, 2, 3, 4, 6, 11, 12, 13, 14, 15]
        # action is ["arm_lift_joint", "arm_flex_joint", "arm_roll_joint", "wrist_flex_joint", "wrist_roll_joint", None, "hand_motor_joint(gripper)", None, None, None, None, "head_pan_joint", "head_tilt_joint", "base_x", "base_y", "base_t"]
        new_actions[:, aligned_ids] = actions
        # new_actions[:, 6] = _gripper_to_angular(new_actions[:, 6])  # Keep disabled based on prior lite6 behavior.

        return new_actions

    return actions


def _decode_actions_inv(
    actions: np.ndarray, *, adapt_to_pi: bool = False
) -> np.ndarray:
    if adapt_to_pi:
        # compress actions to 11 dimensions from 16 dimensions
        aligned_ids = [0, 1, 2, 3, 4, 6, 11, 12, 13, 14, 15]
        actions = actions[:, aligned_ids]
        # action is ["arm_lift_joint", "arm_flex_joint", "arm_roll_joint", "wrist_flex_joint", "wrist_roll_joint","hand_motor_joint(gripper)", "head_pan_joint", "head_tilt_joint" , "base_x", "base_y", "base_t"]

    return actions


def _encode_actions(
    actions: np.ndarray, *, adapt_to_pi: bool = False, convert_gripper: bool = False
) -> np.ndarray:
    if adapt_to_pi:
        if convert_gripper:
            actions[:, 5] = _gripper_from_angular(actions[:, 5])
    return actions


def _encode_actions_inv(
    actions: np.ndarray, *, adapt_to_pi: bool = False, convert_gripper: bool = False
) -> np.ndarray:
    if adapt_to_pi:
        if convert_gripper:
            actions[:, 6] = _gripper_from_angular_inv(actions[:, 6])
    return actions


@dataclasses.dataclass(frozen=True)
class ProprioDropout(transforms.DataTransformFn):
    """Zero out proprioceptive state with probability `drop_rate` during training.
    Forces the model to rely on visual information. Training-only transform."""
    drop_rate: float = 0.4

    def __call__(self, data: dict) -> dict:
        if "state" in data and self.drop_rate > 0:
            if np.random.random() < self.drop_rate:
                data["state"] = np.zeros_like(data["state"])
        return data
