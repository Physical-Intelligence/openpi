import dataclasses

import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    """
    Parse and convert the image to the expected format.

    Args:
        image: The input image to be parsed.
    Returns:
        np.ndarray: The parsed image in the expected format.
    """
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    return image


@dataclasses.dataclass(frozen=True)
class PikkiInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the Pikki expected format.

    For Pikki robot dataset, this class handles the conversion of robot-specific data formats.
    """

    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        mask_padding = self.model_type == _model.ModelType.PI0
        state = transforms.pad_to_dim(data["observation/state"], self.action_dim)
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_ if mask_padding else np.True_,
            },
        }

        if "actions" in data:
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class PikkiOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model to the Pikki expected format.

    For Pikki robot dataset, this class handles the conversion of robot-specific data formats.
    """

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :6])}
