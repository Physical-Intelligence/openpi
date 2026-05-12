"""Illumination-aware image augmentation for RM75 training."""

import dataclasses

import numpy as np

from openpi import transforms


def _sample_range(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    low, high = bounds
    return float(rng.uniform(low, high))


def _to_float01(image: np.ndarray) -> tuple[np.ndarray, np.dtype]:
    image = np.asarray(image)
    orig_dtype = image.dtype
    if np.issubdtype(orig_dtype, np.integer):
        return image.astype(np.float32) / 255.0, orig_dtype
    image = image.astype(np.float32)
    if image.max(initial=0.0) > 1.0 or image.min(initial=0.0) < 0.0:
        image = np.clip(image / 255.0, 0.0, 1.0)
    return image, orig_dtype


def _from_float01(image: np.ndarray, orig_dtype: np.dtype) -> np.ndarray:
    image = np.clip(image, 0.0, 1.0)
    if np.issubdtype(orig_dtype, np.integer):
        return np.rint(image * 255.0).astype(orig_dtype)
    return image.astype(orig_dtype)


@dataclasses.dataclass(frozen=True)
class IlluminationAugmentationConfig:
    """Parameter bundle for RM75 illumination augmentation."""

    apply_probability: float = 0.7
    shadow_elevation_deg_range: tuple[float, float] = (3.0, 15.0)
    shadow_strength_range: tuple[float, float] = (0.5, 0.9)
    shadow_softness_fraction: float = 0.05
    brightness_shift_range: tuple[float, float] = (-0.3, 0.3)
    contrast_scale_range: tuple[float, float] = (0.5, 1.5)
    color_temperature_scale: tuple[float, float, float] = (1.0, 0.97, 1.08)
    color_noise_std: float = 0.02
    seed: int | None = None


@dataclasses.dataclass(frozen=True)
class RM75IlluminationAugmentation(transforms.DataTransformFn):
    """Apply combined illumination perturbations to real RM75 images only."""

    config: IlluminationAugmentationConfig = dataclasses.field(default_factory=IlluminationAugmentationConfig)

    def __call__(self, data: dict) -> dict:
        if 'image' not in data:
            return data

        rng = np.random.default_rng(self.config.seed)
        if rng.random() > self.config.apply_probability:
            return data

        image_mask = data.get('image_mask', {})
        new_images = {}
        for key, image in data['image'].items():
            if bool(np.asarray(image_mask.get(key, True))):
                augmented = self._augment_image(np.asarray(image), rng)
                new_images[key] = augmented
                
                # --- PAPER FIGURE SAVING LOGIC ---
                import os
                save_dir = os.environ.get("SAVE_AUG_IMAGES_DIR")
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    # Limit to saving ~100 images per worker using a global variable to avoid FrozenInstanceError
                    global _AUG_IMAGES_SAVE_COUNT
                    if '_AUG_IMAGES_SAVE_COUNT' not in globals():
                        _AUG_IMAGES_SAVE_COUNT = 0
                    
                    if _AUG_IMAGES_SAVE_COUNT < 100:
                        import time
                        from PIL import Image
                        timestamp = int(time.time() * 1000)
                        
                        # Save Original
                        orig_pil = Image.fromarray(np.asarray(image).astype(np.uint8))
                        orig_pil.save(os.path.join(save_dir, f"{timestamp}_{key}_orig.png"))
                        
                        # Save Augmented
                        aug_pil = Image.fromarray(augmented.astype(np.uint8))
                        aug_pil.save(os.path.join(save_dir, f"{timestamp}_{key}_aug.png"))
                        
                        _AUG_IMAGES_SAVE_COUNT += 1
            else:
                new_images[key] = image
                
        data['image'] = new_images
        return data

    def _augment_image(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        float_image, orig_dtype = _to_float01(image)
        float_image = self._apply_low_angle_shadow(float_image, rng)
        float_image = self._apply_brightness_contrast(float_image, rng)
        float_image = self._apply_color_temperature(float_image, rng)
        return _from_float01(float_image, orig_dtype)

    def _apply_low_angle_shadow(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        height, width = image.shape[:2]
        elevation_deg = _sample_range(rng, self.config.shadow_elevation_deg_range)
        azimuth = float(rng.uniform(0.0, 2.0 * np.pi))
        strength = _sample_range(rng, self.config.shadow_strength_range)

        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        center_x = float(rng.uniform(0.0, max(width - 1, 1)))
        center_y = float(rng.uniform(0.0, max(height - 1, 1)))
        normal = np.array([np.cos(azimuth), np.sin(azimuth)], dtype=np.float32)
        signed_distance = (xx - center_x) * normal[0] + (yy - center_y) * normal[1]

        diagonal = float(np.hypot(height, width))
        softness = max(diagonal * self.config.shadow_softness_fraction, 1.0)
        transition = np.clip(0.5 - signed_distance / softness, 0.0, 1.0)
        elevation_scale = np.clip((15.0 - elevation_deg) / 12.0, 0.0, 1.0)
        mask = transition * (0.6 + 0.4 * elevation_scale)
        return image * (1.0 - strength * mask[..., None])

    def _apply_brightness_contrast(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        brightness = _sample_range(rng, self.config.brightness_shift_range)
        contrast = _sample_range(rng, self.config.contrast_scale_range)
        return np.clip(contrast * (image - 0.5) + 0.5 + brightness, 0.0, 1.0)

    def _apply_color_temperature(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        scales = np.asarray(self.config.color_temperature_scale, dtype=np.float32)
        noise = rng.normal(0.0, self.config.color_noise_std, size=(1, 1, image.shape[-1])).astype(np.float32)
        return np.clip(image * scales.reshape(1, 1, -1) + noise, 0.0, 1.0)
