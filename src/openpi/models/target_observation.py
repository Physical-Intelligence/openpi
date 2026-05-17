"""TargetObservation: observation for the TargetVLA family.

Mirrors the structure of ``trace_observation.TraceObservation`` but **drops every
trace-related field** because the target_vla variants completely remove trace
generation. What remains is just the inputs needed to condition the action
expert on the semantic-target keypoint (via AdaRMS) and to supervise the
per-skill completion head.

Fields kept (carried through the dataset / data transforms):

  - ``atomic_token``        : float scalar in {0..K-1} (skill -> hard-routed expert id)
  - ``semantic_target_xy``  : float[2] normalized [0, 1] semantic target pixel
                              (AdaRMS conditioning signal for the action stream)
  - ``has_skill``           : bool[1]   True iff the frame has a valid skill / target
                              annotation. Used to mask out the completion loss when
                              the annotation is missing (also used to silence the
                              completion head's gradient).
  - ``progress``            : float[1]  completion target in [0, 1]
  - ``diffusion_loss_mask`` : bool[1]   per the AtomicObservation convention

Fields removed vs ``TraceObservation``: ``current_ee_xy``, ``future_trace_xy``,
``has_trace``, ``has_overlay``, ``overlay_images``, ``overlay_image_masks``.
There is no planning forward pass and no overlay rendering in this family, so
none of those fields have any consumer.

Existing :class:`openpi.models.model.Observation` is *not* modified — this is a
fresh subclass that mirrors the additive pattern used by ``TraceObservation``.
"""
from __future__ import annotations

import dataclasses  # noqa: F401  (kept for parity with trace_observation; not used directly)
from typing import Generic, TypeVar

import augmax
import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import struct

from openpi.models import model as _model
from openpi.shared import image_tools as _img_tools
import openpi.shared.array_typing as at

ArrayT = TypeVar("ArrayT", bound=jax.Array | torch.Tensor | np.ndarray)


@at.typecheck
@struct.dataclass
class TargetObservation(_model.Observation, Generic[ArrayT]):
    """Observation for the TargetVLA family (no trace fields)."""

    atomic_token: at.Float[ArrayT, "*b"] | None = None
    semantic_target_xy: at.Float[ArrayT, "*b 2"] | None = None
    has_skill: at.Bool[ArrayT, "*b"] | None = None
    progress: at.Float[ArrayT, "*b"] | None = None
    diffusion_loss_mask: at.Bool[ArrayT, "*b"] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "TargetObservation[ArrayT]":
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")
        # Image normalization (uint8 -> [-1, 1] float32) — same convention as
        # ``Observation.from_dict`` / ``TraceObservation.from_dict``.
        for key in data["image"]:
            if data["image"][key].dtype == np.uint8:
                data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
            elif hasattr(data["image"][key], "dtype") and data["image"][key].dtype == torch.uint8:
                data["image"][key] = data["image"][key].to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
            atomic_token=data.get("atomic_token"),
            semantic_target_xy=data.get("semantic_target_xy"),
            has_skill=data.get("has_skill"),
            progress=data.get("progress"),
            diffusion_loss_mask=data.get("diffusion_loss_mask"),
        )


_BASE_IMAGE_KEY = "base_0_rgb"


def preprocess_target_observation(
    rng: at.KeyArrayLike | None,
    observation: TargetObservation,
    *,
    train: bool = False,
    image_keys=_model.IMAGE_KEYS,
    image_resolution: tuple[int, int] = _model.IMAGE_RESOLUTION,
) -> TargetObservation:
    """Resize and augment images, keeping the semantic-target keypoint aligned.

    When ``train=True`` the same random geometric transform (random crop +
    rotate, plus color jitter) is applied jointly to:

      - the base camera image (``base_0_rgb``),
      - the semantic-target keypoint (``semantic_target_xy``).

    Wrist images go through a *separate color-only chain* (no geometric
    transform), matching the wrist-policy used by both
    ``_model.preprocess_observation`` and
    ``trace_observation.preprocess_trace_observation``.

    Geometric augmentation does **not** apply to the wrist cameras because
    their viewpoint is calibrated to the robot's gripper, not to a global
    workspace frame; rotating the wrist image would break the action / state
    correspondence the action head already learned.

    Out-of-bounds keypoints (rare for LIBERO since the workspace is centered)
    are clamped to ``[0, 1]^2``.

    When ``train=False`` we only resize images to the target resolution; no
    augmax is applied. The semantic-target keypoint passes through unchanged.
    """
    H, W = image_resolution
    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    # ---- Step 1: bring all images to the target resolution. ----
    raw_images: dict = {}
    for key in image_keys:
        img = observation.images[key]
        if img.shape[1:3] != image_resolution:
            img = _img_tools.resize_with_pad(img, *image_resolution)
        raw_images[key] = img

    sem_xy = observation.semantic_target_xy

    # ---- Step 2: training-time augmentation. ----
    if train and rng is not None:
        # Convert to [0, 1] for augmax, matching ``_model.preprocess_observation``.
        for key in raw_images:
            raw_images[key] = raw_images[key] / 2.0 + 0.5

        # Per-batch RNGs for vmap.
        B = raw_images[_BASE_IMAGE_KEY].shape[0]
        sub_rngs = jax.random.split(rng, B)

        scale = jnp.asarray([W - 1, H - 1], dtype=jnp.float32)

        has_keypoint = sem_xy is not None

        if has_keypoint:
            # Joint geometric + color chain on (base, sem_keypoint). One KEYPOINTS
            # entry of shape (B, 1, 2) keeps the call signature aligned with the
            # multi-keypoint chain used by ``preprocess_trace_observation``.
            geom_chain = augmax.Chain(
                augmax.RandomCrop(int(W * 0.95), int(H * 0.95)),
                augmax.Resize(W, H),
                augmax.Rotate((-5, 5)),
                augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
                input_types=[augmax.InputType.IMAGE, augmax.InputType.KEYPOINTS],
            )

            sem_px = (sem_xy * scale)[:, None, :]  # (B, 1, 2) in pixel coords

            def _apply_geom(rng_b, img, kp):
                return geom_chain(rng_b, [img, kp])

            base_out, sem_out = jax.vmap(_apply_geom)(
                sub_rngs, raw_images[_BASE_IMAGE_KEY], sem_px
            )
            raw_images[_BASE_IMAGE_KEY] = base_out
            sem_xy = jnp.clip(sem_out[:, 0, :] / scale, 0.0, 1.0)
        else:
            # Defensive fallback: no keypoint provided — apply a plain image-only chain.
            base_chain = augmax.Chain(
                augmax.RandomCrop(int(W * 0.95), int(H * 0.95)),
                augmax.Resize(W, H),
                augmax.Rotate((-5, 5)),
                augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
                input_types=[augmax.InputType.IMAGE],
            )
            raw_images[_BASE_IMAGE_KEY] = jax.vmap(
                lambda rng_b, img: base_chain(rng_b, [img])[0]
            )(sub_rngs, raw_images[_BASE_IMAGE_KEY])

        # Color-only chain on wrist images (and any other non-base key).
        wrist_chain = augmax.Chain(
            augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
            input_types=[augmax.InputType.IMAGE],
        )
        for key in image_keys:
            if key == _BASE_IMAGE_KEY:
                continue
            raw_images[key] = jax.vmap(
                lambda rng_b, img: wrist_chain(rng_b, [img])[0]
            )(sub_rngs, raw_images[key])

        # Back to [-1, 1].
        for key in raw_images:
            raw_images[key] = raw_images[key] * 2.0 - 1.0

    # ---- Step 3: build masks (default ones if missing). ----
    batch_shape = observation.state.shape[:-1]
    out_image_masks: dict = {}
    for key in image_keys:
        if observation.image_masks is not None and key in observation.image_masks:
            out_image_masks[key] = jnp.asarray(observation.image_masks[key])
        else:
            out_image_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool_)

    return TargetObservation(
        images=raw_images,
        image_masks=out_image_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
        atomic_token=observation.atomic_token,
        semantic_target_xy=sem_xy,
        has_skill=observation.has_skill,
        progress=observation.progress,
        diffusion_loss_mask=observation.diffusion_loss_mask,
    )
