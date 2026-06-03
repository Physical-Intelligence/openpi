"""TraceObservation: extends Observation for the TraceVLA model.

Carries the extra per-sample fields produced by ``LiberoTraceDataset``:

  - ``atomic_token``           : float scalar in {0..K-1} (skill -> hard-routed expert id)
  - ``semantic_target_xy``     : float[2] normalized [0,1] semantic target pixel
  - ``current_ee_xy``          : float[2] normalized [0,1] EE pixel at this frame
  - ``future_trace_xy``        : float[N,2] normalized [0,1] resampled GT trace
                                  (only used as supervision target during training; passed
                                  in the actions slot of the model; here we just carry a
                                  placeholder shape so to_dict round-trips cleanly)
  - ``has_trace``              : bool[1]   True iff the trace is valid for this frame
  - ``has_overlay``            : bool[1]   True iff the base image already carries an overlay
                                  (set by the data loader's anchor-age augmentation)
  - ``diffusion_loss_mask``    : bool[1]   carried over for symmetry with AtomicObservation

Existing :class:`openpi.models.model.Observation` is *not* modified - this is a fresh subclass
that we register through a separate code path.
"""
from __future__ import annotations

import dataclasses
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
class TraceObservation(_model.Observation, Generic[ArrayT]):
    """Observation for the TraceVLA model."""

    atomic_token: at.Float[ArrayT, "*b"] | None = None
    semantic_target_xy: at.Float[ArrayT, "*b 2"] | None = None
    current_ee_xy: at.Float[ArrayT, "*b 2"] | None = None
    has_trace: at.Bool[ArrayT, "*b"] | None = None
    has_overlay: at.Bool[ArrayT, "*b"] | None = None
    progress: at.Float[ArrayT, "*b"] | None = None
    diffusion_loss_mask: at.Bool[ArrayT, "*b"] | None = None

    # Resampled GT trace target for the trace flow-matching loss.
    future_trace_xy: at.Float[ArrayT, "*b n 2"] | None = None

    # Overlay images (only `base_0_rgb` typically): used for execution-mode forward pass.
    # Wrist images are reused from `images`.
    overlay_images: dict[str, at.Float[ArrayT, "*b h w c"]] | None = None
    overlay_image_masks: dict[str, at.Bool[ArrayT, "*b"]] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "TraceObservation[ArrayT]":
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")
        # Image normalization (uint8 -> [-1, 1] float32) — same convention as Observation.from_dict.
        for key in data["image"]:
            if data["image"][key].dtype == np.uint8:
                data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
            elif hasattr(data["image"][key], "dtype") and data["image"][key].dtype == torch.uint8:
                data["image"][key] = data["image"][key].to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
        # Same conversion for overlay images, if present.
        overlay_images = data.get("overlay_image")
        if overlay_images is not None:
            for key in overlay_images:
                if overlay_images[key].dtype == np.uint8:
                    overlay_images[key] = overlay_images[key].astype(np.float32) / 255.0 * 2.0 - 1.0
                elif hasattr(overlay_images[key], "dtype") and overlay_images[key].dtype == torch.uint8:
                    overlay_images[key] = overlay_images[key].to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
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
            current_ee_xy=data.get("current_ee_xy"),
            has_trace=data.get("has_trace"),
            has_overlay=data.get("has_overlay"),
            progress=data.get("progress"),
            diffusion_loss_mask=data.get("diffusion_loss_mask"),
            future_trace_xy=data.get("future_trace_xy"),
            overlay_images=overlay_images,
            overlay_image_masks=data.get("overlay_image_mask"),
        )


_BASE_IMAGE_KEY = "base_0_rgb"


def preprocess_trace_observation(
    rng: at.KeyArrayLike | None,
    observation: TraceObservation,
    *,
    train: bool = False,
    image_keys=_model.IMAGE_KEYS,
    image_resolution: tuple[int, int] = _model.IMAGE_RESOLUTION,
    image_source_hw: tuple[int, int] | None = None,
) -> TraceObservation:
    """Resize and augment images for the TraceVLA model.

    Crucially, when ``train=True`` we apply the *same* random geometric transform
    (random crop + rotate, plus color jitter) to all of:

      - the base camera image (``base_0_rgb``),
      - the overlay version of the base image (``overlay_images['base_0_rgb']``),
      - the semantic-target keypoint (``semantic_target_xy``),
      - the current end-effector keypoint (``current_ee_xy``),
      - every waypoint of the supervised future trace (``future_trace_xy``).

    This keeps the image-space conditioning labels and the trace flow-matching target
    in correspondence with the visual content the VLM actually sees, fixing the
    misalignment that ``_model.preprocess_observation`` (image-only) would otherwise
    introduce. Wrist images receive a *separate* color-only chain (no geometric
    transform), matching ``_model.preprocess_observation``'s wrist policy.

    Out-of-bounds keypoints (e.g. caused by the random crop pulling a near-edge
    point past the boundary) are clamped back into the unit square. With 5%
    crop margin and ±5° rotation this is rare for LIBERO-style data because the
    workspace points sit well inside the camera frame.

    ``image_source_hw`` is the (height, width) of the original camera frame *before*
    ``resize_with_pad`` letterboxes it into ``image_resolution``. It only affects how
    the image-space keypoints (``semantic_target_xy``, ``current_ee_xy``,
    ``future_trace_xy``) are placed into pixel space for the geometric augmentation.
    When ``None`` (the default — used by square-image datasets such as LIBERO), the
    keypoints map linearly to ``[0, W-1] x [0, H-1]`` exactly as before. For a
    non-square source (e.g. the 480x640 table-tasks camera), ``resize_with_pad``
    confines the visible content to a letterboxed sub-rectangle of the square model
    image; we place the keypoints into that *same* rectangle so the random
    crop/resize/rotate moves them in lockstep with the visible content
    (label-preserving augmentation), then invert with the same fixed rectangle. The
    normalized coordinate convention is therefore unchanged: keypoints stay in the
    source camera frame's ``[0, 1]^2`` both in and out, so overlay rendering, the
    trace supervision target, and inference are all unaffected.

    When ``train=False`` we only resize images to the target resolution; no augmax
    is applied. Overlay images and trace keypoints pass through unchanged.
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

    raw_overlay: dict | None = None
    if observation.overlay_images is not None:
        raw_overlay = {}
        for key, img in observation.overlay_images.items():
            if img.shape[1:3] != image_resolution:
                img = _img_tools.resize_with_pad(img, *image_resolution)
            raw_overlay[key] = img

    sem_xy = observation.semantic_target_xy
    cur_ee_xy = observation.current_ee_xy
    future_trace_xy = observation.future_trace_xy

    # ---- Step 2: training-time augmentation. ----
    if train and rng is not None:
        # Convert to [0, 1] for augmax, matching `_model.preprocess_observation`.
        for key in raw_images:
            raw_images[key] = raw_images[key] / 2.0 + 0.5
        if raw_overlay is not None:
            for key in raw_overlay:
                raw_overlay[key] = raw_overlay[key] / 2.0 + 0.5

        # Per-batch RNGs for vmap. Using the same `rng` here means base, overlay, and
        # keypoints share identical transform parameters per batch element. Wrist
        # ColorJitter also derives from the same `rng` split, mirroring the
        # behavior of `_model.preprocess_observation` (color jitter is consistent
        # across cameras within a sample).
        B = raw_images[_BASE_IMAGE_KEY].shape[0]
        sub_rngs = jax.random.split(rng, B)

        # Keypoint <-> pixel mapping used to place the image-space keypoints into the
        # 224x224 model frame for the geometric augmentation (and to invert afterward).
        #   px = xy * kp_scale + kp_offset ;  xy = (px - kp_offset) / kp_scale
        # For a square source (``image_source_hw is None``) this is the original linear
        # map onto [0, W-1] x [0, H-1] (kp_offset = 0). For a non-square source we mirror
        # ``resize_with_pad``'s letterbox so keypoints land on the visible content
        # rectangle and track it under crop/resize/rotate; the inverse uses the same
        # fixed rectangle, preserving the source-frame [0, 1]^2 coordinate convention.
        if image_source_hw is None:
            kp_scale = jnp.asarray([W - 1, H - 1], dtype=jnp.float32)
            kp_offset = jnp.asarray([0.0, 0.0], dtype=jnp.float32)
        else:
            H_src, W_src = int(image_source_hw[0]), int(image_source_hw[1])
            ratio = max(W_src / W, H_src / H)
            content_w = int(round(W_src / ratio))
            content_h = int(round(H_src / ratio))
            pad_x = (W - content_w) // 2
            pad_y = (H - content_h) // 2
            kp_scale = jnp.asarray([content_w - 1, content_h - 1], dtype=jnp.float32)
            kp_offset = jnp.asarray([float(pad_x), float(pad_y)], dtype=jnp.float32)

        # ---- 2a. Geometric + color chain on base + overlay + keypoints ----
        has_keypoints = (
            sem_xy is not None and cur_ee_xy is not None and future_trace_xy is not None
        )
        has_overlay_base = raw_overlay is not None and _BASE_IMAGE_KEY in raw_overlay

        if has_keypoints:
            geom_input_types = [augmax.InputType.IMAGE]                   # base
            if has_overlay_base:
                geom_input_types.append(augmax.InputType.IMAGE)           # overlay
            geom_input_types.extend([augmax.InputType.KEYPOINTS] * 3)     # sem, ee, future_trace

            geom_chain = augmax.Chain(
                augmax.RandomCrop(int(W * 0.95), int(H * 0.95)),
                augmax.Resize(W, H),
                augmax.Rotate((-5, 5)),
                augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
                input_types=geom_input_types,
            )

            # Convert keypoints to pixel coordinates on the (possibly letterboxed)
            # content rectangle of the 224x224 model frame.
            sem_px = (sem_xy * kp_scale + kp_offset)[:, None, :]   # (B, 1, 2)
            ee_px = (cur_ee_xy * kp_scale + kp_offset)[:, None, :] # (B, 1, 2)
            ft_px = future_trace_xy * kp_scale + kp_offset         # (B, N, 2)

            inputs = [raw_images[_BASE_IMAGE_KEY]]
            if has_overlay_base:
                inputs.append(raw_overlay[_BASE_IMAGE_KEY])
            inputs.extend([sem_px, ee_px, ft_px])

            def _apply_geom(rng_b, *args):
                return geom_chain(rng_b, list(args))

            outs = jax.vmap(_apply_geom)(sub_rngs, *inputs)

            idx = 0
            raw_images[_BASE_IMAGE_KEY] = outs[idx]; idx += 1
            if has_overlay_base:
                raw_overlay[_BASE_IMAGE_KEY] = outs[idx]; idx += 1
            sem_aug = outs[idx]; idx += 1
            ee_aug = outs[idx]; idx += 1
            ft_aug = outs[idx]

            # Back to normalized [0, 1] (source camera frame), clamped to bounds.
            sem_xy = jnp.clip((sem_aug[:, 0, :] - kp_offset) / kp_scale, 0.0, 1.0)
            cur_ee_xy = jnp.clip((ee_aug[:, 0, :] - kp_offset) / kp_scale, 0.0, 1.0)
            future_trace_xy = jnp.clip((ft_aug - kp_offset) / kp_scale, 0.0, 1.0)
        else:
            # No keypoints provided — fall back to a plain image-only geom chain.
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
            if has_overlay_base:
                raw_overlay[_BASE_IMAGE_KEY] = jax.vmap(
                    lambda rng_b, img: base_chain(rng_b, [img])[0]
                )(sub_rngs, raw_overlay[_BASE_IMAGE_KEY])

        # ---- 2b. Color-only chain on wrist images (and any other non-base, non-wrist keys) ----
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

        # Convert images back to [-1, 1].
        for key in raw_images:
            raw_images[key] = raw_images[key] * 2.0 - 1.0
        if raw_overlay is not None:
            for key in raw_overlay:
                raw_overlay[key] = raw_overlay[key] * 2.0 - 1.0

    # ---- Step 3: build masks (default ones if missing). ----
    batch_shape = observation.state.shape[:-1]
    out_image_masks: dict = {}
    for key in image_keys:
        if observation.image_masks is not None and key in observation.image_masks:
            out_image_masks[key] = jnp.asarray(observation.image_masks[key])
        else:
            out_image_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool_)

    overlay_image_masks = observation.overlay_image_masks if raw_overlay is not None else None
    overlay_images = raw_overlay  # may be None

    return TraceObservation(
        images=raw_images,
        image_masks=out_image_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
        atomic_token=observation.atomic_token,
        semantic_target_xy=sem_xy,
        current_ee_xy=cur_ee_xy,
        has_trace=observation.has_trace,
        has_overlay=observation.has_overlay,
        progress=observation.progress,
        diffusion_loss_mask=observation.diffusion_loss_mask,
        future_trace_xy=future_trace_xy,
        overlay_images=overlay_images,
        overlay_image_masks=overlay_image_masks,
    )


def trace_inputs_spec(config, *, batch_size: int = 1) -> tuple["TraceObservation", _model.Actions]:
    """Shared ``inputs_spec`` for the TraceVLA config family.

    Pi0TraceVLAConfig / Pi0TraceVLAMoeConfig / Pi0TraceVLAActionMoeConfig all consume the identical
    TraceObservation schema (same dataset + transforms), so they delegate here. ``config`` is
    duck-typed: it only needs ``action_dim``, ``action_horizon``, ``max_token_len``,
    ``trace_horizon`` and ``trace_dim``.
    """
    image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
    image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)
    with at.disable_typechecking():
        observation_spec = TraceObservation(
            images={
                "base_0_rgb": image_spec,
                "left_wrist_0_rgb": image_spec,
                "right_wrist_0_rgb": image_spec,
            },
            image_masks={
                "base_0_rgb": image_mask_spec,
                "left_wrist_0_rgb": image_mask_spec,
                "right_wrist_0_rgb": image_mask_spec,
            },
            state=jax.ShapeDtypeStruct([batch_size, config.action_dim], jnp.float32),
            tokenized_prompt=jax.ShapeDtypeStruct([batch_size, config.max_token_len], jnp.int32),
            tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, config.max_token_len], jnp.bool_),
            token_ar_mask=jax.ShapeDtypeStruct([batch_size, config.max_token_len], jnp.int32),
            token_loss_mask=jax.ShapeDtypeStruct([batch_size, config.max_token_len], jnp.bool_),
            atomic_token=jax.ShapeDtypeStruct([batch_size], jnp.float32),
            semantic_target_xy=jax.ShapeDtypeStruct([batch_size, 2], jnp.float32),
            current_ee_xy=jax.ShapeDtypeStruct([batch_size, 2], jnp.float32),
            has_trace=jax.ShapeDtypeStruct([batch_size], jnp.bool_),
            has_overlay=jax.ShapeDtypeStruct([batch_size], jnp.bool_),
            progress=jax.ShapeDtypeStruct([batch_size], jnp.float32),
            diffusion_loss_mask=jax.ShapeDtypeStruct([batch_size], jnp.bool_),
            future_trace_xy=jax.ShapeDtypeStruct([batch_size, config.trace_horizon, config.trace_dim], jnp.float32),
            overlay_images={
                "base_0_rgb": image_spec,
            },
            overlay_image_masks={
                "base_0_rgb": image_mask_spec,
            },
        )
    action_spec = jax.ShapeDtypeStruct([batch_size, config.action_horizon, config.action_dim], jnp.float32)
    return observation_spec, action_spec
