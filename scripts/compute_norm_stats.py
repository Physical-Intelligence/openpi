"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.

For multi-embodiment configs (MultiEmbodimentDataConfig), statistics are computed
independently per embodiment and saved in a hierarchical directory structure::

    <assets_dir>/<config_name>/<embodiment_name>/norm_stats.json
"""

from __future__ import annotations

import pathlib

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def _compute_and_save_stats(
    data_loader: _data_loader.Dataset,
    num_batches: int,
    output_path: pathlib.Path,
    description: str = "Computing stats",
) -> None:
    """Iterate over a data loader, compute running statistics, and save."""
    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc=description):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: s.get_statistics() for key, s in stats.items()}

    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


def compute_single_config_stats(
    config: _config.TrainConfig,
    max_frames: int | None = None,
) -> None:
    """Compute norm stats for a standard (single-embodiment) config."""
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    output_path = config.assets_dirs / data_config.repo_id
    _compute_and_save_stats(data_loader, num_batches, output_path)


def compute_multi_embodiment_stats(
    config: _config.TrainConfig,
    max_frames: int | None = None,
) -> None:
    """Compute norm stats independently for each embodiment in a multi-embodiment config.

    Stats are saved to: <assets_dir>/<config_name>/<embodiment_name>/norm_stats.json
    """
    assert isinstance(config.data, _config.MultiEmbodimentDataConfig)
    embodiments = config.data.embodiments

    for emb in embodiments:
        print(f"\n{'='*60}")
        print(f"Computing norm stats for embodiment: {emb.name}")
        print(f"  data_path: {emb.data_path}")
        print(f"  tag_id: {emb.tag_id}")
        print(f"{'='*60}")

        if emb.data_path == "fake":
            print(f"Skipping fake embodiment '{emb.name}'")
            continue

        # Create a temporary DataConfig for this embodiment.
        repack = emb.get_repack_transforms()
        emb_data_config = _config.DataConfig(
            repo_id=emb.data_path,
            repack_transforms=repack,
            data_transforms=emb.data_transforms,
            action_sequence_keys=emb.action_sequence_keys,
            prompt_from_task=emb.prompt_from_task,
        )

        data_loader, num_batches = create_torch_dataloader(
            emb_data_config,
            config.model.action_horizon,
            config.batch_size,
            config.model,
            config.num_workers,
            max_frames,
        )

        output_path = config.assets_dirs / emb.name
        _compute_and_save_stats(
            data_loader,
            num_batches,
            output_path,
            description=f"Computing stats [{emb.name}]",
        )


def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)

    if isinstance(config.data, _config.MultiEmbodimentDataConfig):
        compute_multi_embodiment_stats(config, max_frames)
    else:
        compute_single_config_stats(config, max_frames)


if __name__ == "__main__":
    tyro.cli(main)
