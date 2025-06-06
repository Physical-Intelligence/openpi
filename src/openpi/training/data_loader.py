from collections.abc import Iterator, Sequence
import multiprocessing
import os
import typing
from typing import Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


class MixtureDataset(Dataset[T_co]):
    """Dataset that samples from multiple datasets with weights similar to TensorFlow's sample_from_datasets.

    For each element, a dataset is selected based on weight[i], and the next unused element
    from that dataset is returned. Sampling is done without replacement.
    """

    def __init__(
        self,
        datasets: Sequence[Dataset],
        weights: Sequence[float] | None = None,
        *,
        stop_on_empty_dataset: bool = False,
        seed: int | None = None,
    ):
        """Create a dataset that samples from multiple datasets with given weights.

        Args:
            datasets: A sequence of datasets to sample from.
            weights: Optional weights for each dataset. If not provided, uniform weights are used.
            stop_on_empty_dataset: If True, stop when any dataset is exhausted.
                If False, continue with the remaining datasets.
            seed: Optional seed for reproducibility.
        """
        if not datasets:
            raise ValueError("At least one dataset must be provided.")

        self._datasets = datasets

        if weights is None:
            weights = [1.0] * len(datasets)

        if len(weights) != len(datasets):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of datasets ({len(datasets)}).")

        self._weights = np.array(weights, dtype=np.float64)
        self._stop_on_empty_dataset = stop_on_empty_dataset
        self._rng = np.random.RandomState(seed)

        self._length = sum(len(dataset) for dataset in datasets)
        self._sampling_order = self._calculate_sampling_order()

    @property
    def datasets(self) -> Sequence[Dataset]:
        """Get the datasets in this mixture."""
        return self._datasets

    @datasets.setter
    def datasets(self, new_datasets: Sequence[Dataset]) -> None:
        """Set new datasets for this mixture.

        Args:
            new_datasets: A sequence of datasets to sample from.

        Raises:
            ValueError: If the number of weights doesn't match the number of datasets.
        """
        if not new_datasets:
            raise ValueError("At least one dataset must be provided.")

        if len(new_datasets) != len(self._weights):
            raise ValueError(
                f"Number of weights ({len(self._weights)}) must match number of datasets ({len(new_datasets)})."
            )

        self._datasets = new_datasets
        self._length = sum(len(dataset) for dataset in new_datasets)
        self._sampling_order = self._calculate_sampling_order()

    def _calculate_sampling_order(self):
        """Calculate the sampling order for this dataset mixture."""
        remaining_indices = [list(range(len(dataset))) for dataset in self._datasets]
        active_datasets = list(range(len(self._datasets)))
        weights = self._weights.copy()

        sampling_order = []

        while active_datasets and len(sampling_order) < self._length:
            active_weights = weights[active_datasets]
            total_weight = np.sum(active_weights)

            if total_weight <= 0:
                active_weights = np.ones(len(active_datasets)) / len(active_datasets)
            else:
                active_weights = active_weights / total_weight

            dataset_idx = self._rng.choice(active_datasets, p=active_weights)

            if remaining_indices[dataset_idx]:
                element_idx = remaining_indices[dataset_idx].pop(0)
                sampling_order.append((dataset_idx, element_idx))
            else:
                active_datasets.remove(dataset_idx)

                if self._stop_on_empty_dataset:
                    break

        return sampling_order

    def __len__(self) -> int:
        return len(self._sampling_order)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        idx = index.__index__()

        if idx >= len(self._sampling_order):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self._sampling_order)}")

        dataset_idx, element_idx = self._sampling_order[idx]

        return self._datasets[dataset_idx][element_idx]


def create_dataset(data_config: _config.DataConfigBase, model_config: _model.BaseModelConfig) -> Dataset:
    """Create a dataset for training."""
    if isinstance(data_config, _config.DataMixtureConfig):
        datasets = []
        for config in data_config.dataset_configs:
            dataset = create_dataset(config, model_config)
            datasets.append(dataset)

        return MixtureDataset(
            datasets,
            weights=data_config.weights,
            stop_on_empty_dataset=data_config.stop_on_empty_dataset,
        )

    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(model_config.action_horizon)]
            for key in data_config.action_sequence_keys
        },
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def transform_dataset(
    dataset: Dataset, data_config: _config.DataConfigBase, *, skip_norm_stats: bool = False
) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    if isinstance(dataset, MixtureDataset):
        dataset.datasets = [
            transform_dataset(d, cfg, skip_norm_stats=skip_norm_stats)
            for d, cfg in zip(dataset.datasets, data_config.dataset_configs, strict=True)
        ]
        return dataset

    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = create_dataset(data_config, config.model)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=config.seed,
    )

    class DataLoaderImpl(DataLoader):
        def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader):
            self._data_config = data_config
            self._data_loader = data_loader

        def data_config(self) -> _config.DataConfig:
            return self._data_config

        def __iter__(self):
            for batch in self._data_loader:
                yield _model.Observation.from_dict(batch), batch["actions"]

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *x: np.stack(np.asarray(x), axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
