import logging
import os
import pathlib
from typing import Any

import jax.numpy as jnp

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


def _infer_checkpoint_asset_id(assets_dir: pathlib.Path) -> str:
    """Infer the single asset_id stored under a checkpoint's `assets/` directory."""
    if not assets_dir.exists():
        raise FileNotFoundError(f"Checkpoint assets directory does not exist: {assets_dir}")

    candidates = sorted([p.name for p in assets_dir.iterdir() if p.is_dir()])
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        raise FileNotFoundError(f"No asset directories found under checkpoint assets dir: {assets_dir}")
    raise ValueError(
        "Cannot infer which asset_id to use from checkpoint assets directory "
        f"(multiple candidates found): {candidates}. "
        "Please specify the correct asset_id via the training config's data.assets.asset_id."
    )


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # Check if this is a PyTorch model by looking for model.safetensors
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    logging.info("Loading model...")
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        assets_dir = pathlib.Path(checkpoint_dir) / "assets"
        asset_id = data_config.asset_id

        # Fine-tunes can override `--data.repo-id` during training, so the checkpoint may contain a different
        # `assets/<asset_id>` than the static train config. In that case, infer the correct asset id from the
        # checkpoint itself.
        if asset_id is None or not (assets_dir / asset_id).is_dir():
            inferred = _infer_checkpoint_asset_id(assets_dir)
            if asset_id is None:
                logging.info("No asset_id set in config; inferred %r from checkpoint assets.", inferred)
            else:
                logging.warning(
                    "asset_id=%r not found under checkpoint assets (%s); using inferred %r instead.",
                    asset_id,
                    assets_dir,
                    inferred,
                )
            asset_id = inferred

        norm_stats = _checkpoints.load_norm_stats(assets_dir, asset_id)

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )
