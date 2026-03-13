"""Multi-embodiment data loader that interleaves samples from multiple robots."""

from __future__ import annotations

import logging
import pathlib
from collections.abc import Iterator, Sequence
from typing import SupportsIndex, TypeVar

import jax
import numpy as np
import torch

import openpi.models.model as _model
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms
from openpi.data.embodiments.config import EmbodimentConfig

T_co = TypeVar("T_co", covariant=True)


def load_embodiment_norm_stats(
    assets_dir: pathlib.Path,
    embodiment: EmbodimentConfig,
) -> dict[str, _transforms.NormStats] | None:
    """Load norm stats for a single embodiment from its hierarchical path.

    Looks for norm_stats.json at: assets_dir / embodiment.name / norm_stats.json
    """
    stats_dir = assets_dir / embodiment.name
    try:
        norm_stats = _normalize.load(stats_dir)
        logging.info(f"Loaded norm stats for embodiment '{embodiment.name}' from {stats_dir}")
        return norm_stats
    except FileNotFoundError:
        logging.warning(f"Norm stats not found for embodiment '{embodiment.name}' at {stats_dir}")
        return None


class MultiEmbodimentDataset(_data_loader.Dataset):
    """Combines multiple embodiment datasets with weighted sampling.

    Each sample is augmented with an ``embodiment_id`` key so that downstream
    model or loss logic can identify the source robot.
    """

    def __init__(
        self,
        datasets: Sequence[_data_loader.Dataset],
        embodiments: Sequence[EmbodimentConfig],
        transforms_per_embodiment: Sequence[Sequence[_transforms.DataTransformFn]],
    ) -> None:
        if len(datasets) != len(embodiments):
            raise ValueError("datasets and embodiments must have the same length")

        self._datasets = list(datasets)
        self._embodiments = list(embodiments)
        self._transforms = [_transforms.compose(t) for t in transforms_per_embodiment]

        # Build cumulative index mapping: global_idx -> (dataset_idx, local_idx)
        self._offsets: list[int] = []
        self._dataset_indices: list[int] = []
        offset = 0
        for i, ds in enumerate(self._datasets):
            self._offsets.append(offset)
            offset += len(ds)
        self._total_len = offset

        # Compute sampling weights (normalised).
        weights = np.array([e.sampling_weight for e in embodiments], dtype=np.float64)
        self._weights = weights / weights.sum()

    def __getitem__(self, index: SupportsIndex) -> dict:
        idx = index.__index__()
        # Map global index to (dataset_index, local_index).
        ds_idx = 0
        for i, offset in enumerate(self._offsets):
            if i + 1 < len(self._offsets) and idx >= self._offsets[i + 1]:
                continue
            ds_idx = i
            break
        local_idx = idx - self._offsets[ds_idx]
        local_idx = local_idx % len(self._datasets[ds_idx])

        sample = self._datasets[ds_idx][local_idx]
        return self._transforms[ds_idx](sample)

    def __len__(self) -> int:
        return self._total_len

    @property
    def weights(self) -> np.ndarray:
        """Per-embodiment sampling weights (normalised to sum to 1)."""
        return self._weights

    @property
    def embodiments(self) -> list[EmbodimentConfig]:
        return self._embodiments


class WeightedMultiEmbodimentSampler(torch.utils.data.Sampler):
    """Weighted random sampler that respects per-embodiment sampling weights.

    Draws samples proportionally to the configured ``sampling_weight`` of each
    embodiment, so that under-represented robots can be up-weighted.
    """

    def __init__(
        self,
        dataset: MultiEmbodimentDataset,
        num_samples: int | None = None,
        seed: int = 0,
    ) -> None:
        self._dataset = dataset
        self._num_samples = num_samples or len(dataset)
        self._generator = torch.Generator().manual_seed(seed)

        # Build per-sample weight vector.
        sample_weights: list[float] = []
        for i, (emb, ds) in enumerate(zip(dataset.embodiments, dataset._datasets)):
            w = dataset.weights[i] / len(ds) if len(ds) > 0 else 0.0
            sample_weights.extend([w] * len(ds))
        self._sample_weights = torch.tensor(sample_weights, dtype=torch.double)

    def __iter__(self) -> Iterator[int]:
        indices = torch.multinomial(
            self._sample_weights,
            self._num_samples,
            replacement=True,
            generator=self._generator,
        )
        yield from indices.tolist()

    def __len__(self) -> int:
        return self._num_samples


def create_multi_embodiment_dataset(
    embodiments: Sequence[EmbodimentConfig],
    assets_dir: pathlib.Path,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    *,
    skip_norm_stats: bool = False,
) -> MultiEmbodimentDataset:
    """Create a MultiEmbodimentDataset from a sequence of EmbodimentConfigs.

    For each embodiment:
    1. Creates a LeRobot dataset.
    2. Loads its per-embodiment norm_stats.
    3. Builds the full transform pipeline (repack -> data -> normalize -> inject id).
    """
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

    datasets: list[_data_loader.Dataset] = []
    all_transforms: list[list[_transforms.DataTransformFn]] = []

    for emb in embodiments:
        # Create the raw dataset.
        if emb.data_path == "fake":
            ds: _data_loader.Dataset = _data_loader.FakeDataset(model_config, num_samples=1024)
        else:
            dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(emb.data_path)
            ds = lerobot_dataset.LeRobotDataset(
                emb.data_path,
                delta_timestamps={
                    key: [t / dataset_meta.fps for t in range(action_horizon)]
                    for key in emb.action_sequence_keys
                },
            )
            if emb.prompt_from_task:
                ds = _data_loader.TransformedDataset(
                    ds, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)]
                )

        # Load per-embodiment norm stats.
        norm_stats: dict[str, _transforms.NormStats] | None = None
        if not skip_norm_stats and emb.data_path != "fake":
            norm_stats = load_embodiment_norm_stats(assets_dir, emb)

        # Build transform pipeline for this embodiment.
        repack = emb.get_repack_transforms()
        transforms_list: list[_transforms.DataTransformFn] = [
            *repack.inputs,
            *emb.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=emb.use_quantile_norm),
            _transforms.InjectEmbodimentId(emb.tag_id),
        ]

        datasets.append(ds)
        all_transforms.append(transforms_list)

    return MultiEmbodimentDataset(datasets, embodiments, all_transforms)
