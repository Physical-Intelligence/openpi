import dataclasses
import einops
import numpy as np
import openpi.models.model as _model
import openpi.transforms as _transforms


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class UR5eInputs(_transforms.DataTransformFn):

    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # Expect a sim joint state vector otherwise fall back to concatenating if the state was provided as
        # a tuple of joints/gripper.       
        if "state" in data:
            state = np.asarray(data["state"], dtype=np.float32)
        else:
            joints = np.asarray(data.get("joints", ()), dtype=np.float32)
            gripper = np.asarray(data.get("gripper", ()), dtype=np.float32)
            state = np.concatenate([joints, gripper])
        if state.ndim > 1:
            # Some loaders return the state as (T, D); we only need the current state for conditioning.
            state = state[0]

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        base_image = _parse_image(data.get("image"))
        wrist_image = _parse_image(data.get("wrist_image"))

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
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        # Pass the prompt (aka language instruction) to the model.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs
    


@dataclasses.dataclass(frozen=True)
class UR5eOutputs(_transforms.DataTransformFn):

    def __call__(self, data: dict) -> dict:
        # Since the robot has 7 action dimensions (6 DoF + gripper), return the first 7 dims
        return {"actions": np.asarray(data["actions"][:, :7])}
    
