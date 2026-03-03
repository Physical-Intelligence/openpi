"""RM75 platform policy configuration.

Follows the libero_policy.py / droid_policy.py pattern:
- RM75Inputs: DataTransformFn mapping RM75 observations → model inputs
- RM75Outputs: DataTransformFn mapping model outputs → RM75 actions
- LeRobotRM75DataConfig: DataConfigFactory for RM75 LeRobot datasets
- make_rm75_example(): Creates a random input example dict for testing

RM75 robot platform:
- 7-DoF arm + 1D gripper = 8D action space (absolute joint position)
- State vector: joint_position(7) + gripper_position(1) = 8D
- Cameras: wrist RGB + scene/base RGB
"""

import dataclasses
import pathlib
from typing import Any

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.research.shared.action_transforms import RM75_DELTA_MASK
from openpi.research.shared.action_transforms import RM75AbsoluteActions
from openpi.research.shared.action_transforms import RM75DeltaActions


def _resolve_training_config_types() -> tuple[Any, Any]:
    try:
        from openpi.training.config import DataConfigFactory as RealDataConfigFactory
        from openpi.training.config import ModelTransformFactory as RealModelTransformFactory
    except ImportError as training_import_error:
        import_error = training_import_error

        @dataclasses.dataclass(frozen=True)
        class FallbackDataConfigFactoryBase:
            repo_id: str
            assets: Any = dataclasses.field(default_factory=lambda: None)
            base_config: Any = None

            def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> Any:
                raise ImportError("DataConfigFactory unavailable due to import cycle") from import_error

            def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> Any:
                raise ImportError("DataConfigFactory unavailable due to import cycle") from import_error

        @dataclasses.dataclass(frozen=True)
        class FallbackModelTransformFactory:
            def __call__(self, model_config: _model.BaseModelConfig) -> Any:
                raise ImportError("ModelTransformFactory unavailable due to import cycle") from import_error

        return FallbackDataConfigFactoryBase, FallbackModelTransformFactory

    return RealDataConfigFactory, RealModelTransformFactory


_RM75DataConfigFactoryBase, _RM75ModelTransformFactory = _resolve_training_config_types()


def make_rm75_example() -> dict:
    """Creates a random input example for the RM75 policy."""
    return {
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/scene_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/joint_velocity": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
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
class RM75Inputs(transforms.DataTransformFn):
    """Convert RM75 observations to model input format.

    Maps wrist camera image and proprioceptive state to the model's
    expected input dictionary. Handles PI0, PI05, and PI0_FAST model types
    with appropriate image slot naming and masking.
    """

    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Parse wrist image to uint8 (H,W,C) — handles float32 (C,H,W) from LeRobot.
        wrist_image = _parse_image(data["observation/wrist_image"])
        scene_image = _parse_image(data["observation/scene_image"])

        # Build 8D state vector: joint_position(7) + gripper_position(1).
        state = np.concatenate(
            [
                data["observation/joint_position"],
                data["observation/gripper_position"],
            ]
        )
        # Joint velocity is recorded in the dataset for future use but not included in the state vector

        # Map images to model slots based on model type.
        # RM75 has wrist and scene cameras — pad missing slots with zeros.
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (scene_image, wrist_image, np.zeros_like(wrist_image))
                # Mask out zero-padded images; scene and wrist are real.
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (scene_image, np.zeros_like(wrist_image), wrist_image)
                # FAST models expect all image slots masked true.
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass prompt (language instruction) through.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RM75Outputs(transforms.DataTransformFn):
    """Convert model outputs back to RM75 action format.

    Slices to the first 8 dimensions: 7 joint positions + 1 gripper.
    """

    def __call__(self, data: dict) -> dict:
        # Slice to first 8 dims (7 joint + 1 gripper), discarding padding.
        return {"actions": np.asarray(data["actions"][:, :8])}


@dataclasses.dataclass(frozen=True)
class LeRobotRM75DataConfig(_RM75DataConfigFactoryBase):
    """Data config factory for RM75 robot datasets in LeRobot format.

    Wires up:
    - RepackTransform for key mapping (identity — dataset keys match inference keys)
    - RM75Inputs / RM75Outputs for observation→model and model→action transforms
    - Delta action conversion (absolute joint → delta for training, delta → absolute for inference)
    """

    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> Any:
        # Repack: map LeRobot dot-notation keys to pipeline slash-notation keys.
        repack_transform = transforms.Group(
            inputs=[
                transforms.RepackTransform(
                    {
                        "observation/wrist_image": "observation.wrist_image",
                        "observation/scene_image": "observation.scene_image",
                        "observation/joint_position": "observation.joint_position",
                        "observation/joint_velocity": "observation.joint_velocity",
                        "observation/gripper_position": "observation.gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # Data transforms: RM75-specific input/output mapping.
        data_transforms = transforms.Group(
            inputs=[RM75Inputs(model_type=model_config.model_type)],
            outputs=[RM75Outputs()],
        )

        # RM75 uses absolute joint position actions — convert to delta for training.
        # Joint dims (0:7) get delta conversion; gripper dim (7) stays absolute.
        data_transforms = data_transforms.push(
            inputs=[RM75DeltaActions(mask=RM75_DELTA_MASK)],
            outputs=[RM75AbsoluteActions(mask=RM75_DELTA_MASK)],
        )

        # Model transforms (tokenization, etc.) — standard, no customization needed.
        model_transforms = _RM75ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
