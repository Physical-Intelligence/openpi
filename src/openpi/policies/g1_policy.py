import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    if image.ndim != 3:
        raise ValueError(f"Expected one HWC or CHW head image, got shape {image.shape}")
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.shape[-1] != 3:
        raise ValueError(f"Expected an RGB head image, got shape {image.shape}")
    return image


def _translate_pair(rgb: np.ndarray, depth: np.ndarray, dx: int, dy: int) -> tuple[np.ndarray, np.ndarray]:
    """Translate an aligned RGB-D pair with edge padding."""
    height, width = rgb.shape[:2]
    pad_x, pad_y = abs(dx), abs(dy)

    def translate(image: np.ndarray) -> np.ndarray:
        padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode="edge")
        start_x = pad_x - dx
        start_y = pad_y - dy
        return padded[start_y : start_y + height, start_x : start_x + width].copy()

    return translate(rgb), translate(depth)


@dataclasses.dataclass(frozen=True)
class G1RgbdAugment(transforms.DataTransformFn):
    """Apply bounded real-camera perturbations to a parsed RGB-D pair."""

    max_translation_px: int = 6
    max_brightness_delta: float = 12.0
    max_contrast_delta: float = 0.10
    max_channel_gain_delta: float = 0.05
    max_depth_noise_std: float = 3.0
    occlusion_probability: float = 0.25
    max_occlusion_fraction: float = 0.12

    def __call__(self, data: dict) -> dict:
        images = data.get("image")
        masks = data.get("image_mask", {})
        if not isinstance(images, dict) or "base_0_rgb" not in images:
            raise ValueError("G1 RGB-D augmentation requires parsed policy images")
        if not bool(masks.get("left_wrist_0_rgb", False)):
            raise ValueError("G1 RGB-D augmentation requires valid aligned depth")

        rgb = _parse_image(images["base_0_rgb"])
        depth = _parse_image(images["left_wrist_0_rgb"])
        if rgb.shape != depth.shape:
            raise ValueError("G1 RGB-D augmentation requires matching RGB and depth shapes")

        dx = int(np.random.randint(-self.max_translation_px, self.max_translation_px + 1))
        dy = int(np.random.randint(-self.max_translation_px, self.max_translation_px + 1))
        rgb, depth = _translate_pair(rgb, depth, dx, dy)

        rgb_float = rgb.astype(np.float32)
        contrast = 1.0 + float(np.random.uniform(-self.max_contrast_delta, self.max_contrast_delta))
        brightness = float(np.random.uniform(-self.max_brightness_delta, self.max_brightness_delta))
        channel_gain = np.random.uniform(
            1.0 - self.max_channel_gain_delta,
            1.0 + self.max_channel_gain_delta,
            size=(1, 1, 3),
        ).astype(np.float32)
        rgb = np.clip(
            (rgb_float - 127.5) * contrast * channel_gain + 127.5 + brightness,
            0,
            255,
        ).astype(np.uint8)

        depth_float = depth[..., 0].astype(np.float32)
        depth_noise_std = float(np.random.uniform(0.0, self.max_depth_noise_std))
        if depth_noise_std > 0.0:
            depth_float += np.random.normal(0.0, depth_noise_std, size=depth_float.shape).astype(np.float32)
        depth_plane = np.clip(depth_float, 0, 255).astype(np.uint8)
        depth = np.repeat(depth_plane[..., None], 3, axis=-1)

        if float(np.random.random()) < self.occlusion_probability:
            height, width = rgb.shape[:2]
            min_fraction = min(0.04, self.max_occlusion_fraction)
            fraction = float(np.random.uniform(min_fraction, self.max_occlusion_fraction))
            cutout_height = max(1, int(round(height * fraction)))
            cutout_width = max(1, int(round(width * fraction)))
            top = int(np.random.randint(0, height - cutout_height + 1))
            left = int(np.random.randint(0, width - cutout_width + 1))
            rgb_fill = np.median(rgb.reshape(-1, 3), axis=0).astype(np.uint8)
            rgb[top : top + cutout_height, left : left + cutout_width] = rgb_fill
            depth[top : top + cutout_height, left : left + cutout_width] = 0

        images["base_0_rgb"] = rgb
        images["left_wrist_0_rgb"] = depth
        return data


@dataclasses.dataclass(frozen=True)
class G1Inputs(transforms.DataTransformFn):
    """Map G1 RGB-D and low-dimensional state into pi0.5 image slots.

    The optional aligned depth image is an 8-bit, three-channel inverse-depth
    visualization produced by the dataset/runtime adapter.  It occupies a
    pretrained auxiliary image slot so the pi0.5 checkpoint shape does not
    change.  Missing depth is accepted only for RGB-only configurations.
    """

    model_type: _model.ModelType
    state_dim: int = 29
    task_action_dim: int = 21
    use_depth_image: bool = False

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"], dtype=np.float32)
        head_image = _parse_image(data["head_image"])
        depth_image = None
        if self.use_depth_image:
            if "depth_image" not in data:
                raise ValueError("Expected an aligned G1 depth image")
            depth_image = _parse_image(data["depth_image"])
            if depth_image.shape != head_image.shape:
                raise ValueError(
                    "G1 RGB and aligned depth images must share one shape, got "
                    f"{head_image.shape} and {depth_image.shape}"
                )

        if state.shape != (self.state_dim,):
            raise ValueError(f"Expected {self.state_dim} G1 state values, got shape {state.shape}")
        if not np.isfinite(state).all():
            raise ValueError("G1 state contains non-finite values")

        if self.model_type not in (_model.ModelType.PI0, _model.ModelType.PI05):
            raise ValueError(f"Unsupported G1 model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": head_image,
                "left_wrist_0_rgb": depth_image if depth_image is not None else np.zeros_like(head_image),
                "right_wrist_0_rgb": np.zeros_like(head_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_ if depth_image is not None else np.False_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1] != self.task_action_dim:
                raise ValueError(f"Expected {self.task_action_dim} G1 task actions, got shape {actions.shape}")
            if not np.isfinite(actions).all():
                raise ValueError("G1 actions contain non-finite values")
            inputs["actions"] = actions

        if "prompt" in data:
            prompt = data["prompt"]
            inputs["prompt"] = prompt.decode("utf-8") if isinstance(prompt, bytes) else prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class G1Outputs(transforms.DataTransformFn):
    task_action_dim: int = 21

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., : self.task_action_dim]}
