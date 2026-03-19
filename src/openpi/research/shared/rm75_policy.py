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
import logging
import pathlib
from typing import Any

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.research.shared.action_transforms import RM75_DELTA_MASK
from openpi.research.shared.action_transforms import RM75AbsoluteActions
from openpi.research.shared.action_transforms import RM75DeltaActions
from openpi.research.shared.illumination_augment import IlluminationAugmentationConfig
from openpi.research.shared.illumination_augment import RM75IlluminationAugmentation


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


def _first_present(data: dict, keys: tuple[str, ...], *, field: str):
    for key in keys:
        if key in data:
            return data[key]
    raise KeyError(f"Missing {field}. Tried keys: {keys}")


def _resolve_action_sequence_keys(repo_id: str) -> tuple[str, ...]:
    """Pick the correct LeRobot action key for delta timestamp expansion."""
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

        feature_keys = set(LeRobotDatasetMetadata(repo_id).features.keys())
        if "actions" in feature_keys:
            return ("actions",)
        if "action" in feature_keys:
            return ("action",)
    except Exception as exc:  # noqa: BLE001
        logging.info("Failed to inspect LeRobot metadata for %s: %s", repo_id, exc)

    # Backwards-compatible default used by existing RM75 conversion.
    return ("actions",)


@dataclasses.dataclass(frozen=True)
class RM75LeRobotCanonicalize(transforms.DataTransformFn):
    """Canonicalize LeRobot sample keys for RM75.

    Supports both:
    - Legacy RM75 schema from convert_rm75_data_to_lerobot.py:
      observation.wrist_image, observation.scene_image, observation.joint_position,
      observation.joint_velocity, observation.gripper_position, actions
    - LeRobot v2.1 state-style schema from RoboCOIN-like pipelines:
      observation.images.wrist_image, observation.images.scene_image, observation.state, action
    """

    def __call__(self, data: dict) -> dict:
        wrist_image = _first_present(
            data,
            ("observation.images.wrist_image", "observation.wrist_image"),
            field="wrist image",
        )
        scene_image = _first_present(
            data,
            ("observation.images.scene_image", "observation.scene_image"),
            field="scene image",
        )

        if "observation.state" in data:
            state = np.asarray(data["observation.state"], dtype=np.float32)
            if state.shape[-1] != 8:
                raise ValueError(f"Expected observation.state last dim 8, got {state.shape}")
            joint_position = state[..., :7]
            gripper_position = state[..., 7:]
            joint_velocity = np.zeros_like(joint_position, dtype=np.float32)
        else:
            joint_position = np.asarray(
                _first_present(data, ("observation.joint_position",), field="joint position"),
                dtype=np.float32,
            )
            gripper_position = np.asarray(
                _first_present(data, ("observation.gripper_position",), field="gripper position"),
                dtype=np.float32,
            )
            if "observation.joint_velocity" in data:
                joint_velocity = np.asarray(data["observation.joint_velocity"], dtype=np.float32)
            else:
                joint_velocity = np.zeros_like(joint_position, dtype=np.float32)

        actions = np.asarray(_first_present(data, ("actions", "action"), field="action sequence"), dtype=np.float32)

        canonical = {
            "observation/wrist_image": wrist_image,
            "observation/scene_image": scene_image,
            "observation/joint_position": joint_position,
            "observation/joint_velocity": joint_velocity,
            "observation/gripper_position": gripper_position,
            "actions": actions,
        }
        if "prompt" in data:
            canonical["prompt"] = data["prompt"]
        return canonical


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

    illumination_augmentation: IlluminationAugmentationConfig | None = None

    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> Any:
        # Canonicalize keys across both RM75 LeRobot schema variants.
        repack_transform = transforms.Group(inputs=[RM75LeRobotCanonicalize()])

        # Data transforms: RM75-specific input/output mapping.
        data_transforms = transforms.Group(
            inputs=[RM75Inputs(model_type=model_config.model_type)],
            outputs=[RM75Outputs()],
        )
        if self.illumination_augmentation is not None:
            data_transforms = data_transforms.push(
                inputs=[RM75IlluminationAugmentation(self.illumination_augmentation)]
            )

        # RM75 uses absolute joint position actions — convert to delta for training.
        # Joint dims (0:7) get delta conversion; gripper dim (7) stays absolute.
        data_transforms = data_transforms.push(
            inputs=[RM75DeltaActions(mask=RM75_DELTA_MASK)],
            outputs=[RM75AbsoluteActions(mask=RM75_DELTA_MASK)],
        )

        # Model transforms (tokenization, etc.) — standard, no customization needed.
        model_transforms = _RM75ModelTransformFactory()(model_config)
        action_sequence_keys = _resolve_action_sequence_keys(self.repo_id)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=action_sequence_keys,
        )
