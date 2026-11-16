import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_galbot_example() -> dict:
    """Creates a random input example for the Galbot policy."""
    return {
        "observation/state": np.random.rand(77).astype(np.float32),
        "observation/first_person_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/left_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/right_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
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
class GalbotInputs(transforms.DataTransformFn):
    """Convert Galbot LeRobot / inference inputs to the OpenPI model format.

    Expected keys after repack (training) or from the policy client (inference):
    - observation/state: [77]
    - observation/first_person_image, observation/left_wrist_image, observation/right_wrist_image
    - actions: [action_horizon, 20] (training only)
    - prompt: language instruction
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        first_person = _parse_image(data["observation/first_person_image"])
        left_wrist = _parse_image(data["observation/left_wrist_image"])
        right_wrist = _parse_image(data["observation/right_wrist_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": first_person,
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Actions are only available during training. Padding 20D -> model action_dim
        # is handled later by PadStatesAndActions in ModelTransformFactory.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class GalbotOutputs(transforms.DataTransformFn):
    """Slice model actions back to the Galbot 20D action space (inference only).

    Receives the key ``actions`` (plural) from Policy.infer / Unnormalize, matching
    LiberoOutputs and AlohaOutputs.
    """

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :20])}
