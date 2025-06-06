import dataclasses

import jax

from openpi.models import pi0
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


class TaggedDataset(_data_loader.FakeDataset):
    def __init__(self, model_config, num_samples, tag):
        super().__init__(model_config, num_samples)
        self.tag = tag

    def __getitem__(self, index):
        data = super().__getitem__(index)
        data["tag"] = self.tag
        return data


def test_dataset_mixture():
    config = _config.get_config("debug_mixture")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)

    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_dataset_mixture_init():
    """Test basic initialization of WeightedSampleFromDatasets."""
    config = pi0.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy")
    dataset1 = _data_loader.FakeDataset(config, 20)
    dataset2 = _data_loader.FakeDataset(config, 30)

    mixture1 = _data_loader.MixtureDataset([dataset1, dataset2], seed=42, weights=[0.5, 0.5])
    mixture2 = _data_loader.MixtureDataset([dataset1, dataset2], seed=42, weights=[0.5, 0.5])

    assert mixture1._sampling_order == mixture2._sampling_order  # noqa: SLF001

    mixture1 = _data_loader.MixtureDataset([dataset1, dataset2], seed=42, weights=[0.6, 0.4])
    mixture2 = _data_loader.MixtureDataset([dataset1, dataset2], seed=42, weights=[0.4, 0.6])

    assert mixture1._sampling_order != mixture2._sampling_order  # noqa: SLF001


def test_dataset_mixture_item_access():
    """Test that items can be accessed from the dataset and are from the correct source."""
    config = pi0.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy")

    dataset1 = TaggedDataset(config, 20, "dataset1")
    dataset2 = TaggedDataset(config, 30, "dataset2")

    # Create mixture with equal weights
    mixture = _data_loader.MixtureDataset([dataset1, dataset2], weights=[0.5, 0.5], seed=42)

    # Count items from each dataset to verify distribution
    sources = [mixture[i]["tag"] for i in range(len(mixture))]
    dataset1_count = sources.count("dataset1")
    dataset2_count = sources.count("dataset2")

    # Since datasets are 20 and 30 items, and weights are equal,
    # we should get all 20 from dataset1 and 30 from dataset2
    assert dataset1_count == 20
    assert dataset2_count == 30
    assert dataset1_count + dataset2_count == len(mixture)


def test_dataset_mixture_edge_cases():
    """Test edge cases like single dataset, empty dataset, etc."""
    config = pi0.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy")

    # Single dataset
    single_dataset = _data_loader.FakeDataset(config, 20)
    mixture = _data_loader.MixtureDataset([single_dataset])
    assert len(mixture) == 20

    # Empty datasets should be skipped
    empty_dataset = _data_loader.FakeDataset(config, 0)
    normal_dataset = _data_loader.FakeDataset(config, 20)

    mixture = _data_loader.MixtureDataset([empty_dataset, normal_dataset], weights=[0.3, 0.7])
    assert len(mixture) == 20

    # With stop_on_empty_dataset=True and one empty dataset, length should be less than 20 (on average)
    mixture = _data_loader.MixtureDataset(
        [empty_dataset, normal_dataset], weights=[0.5, 0.5], stop_on_empty_dataset=True
    )
    assert len(mixture) < 20


def test_dataset_mixture_distribution():
    """Test that sampling follows the specified weights."""
    config = pi0.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy")

    dataset1 = TaggedDataset(config, 1000, "dataset1")
    dataset2 = TaggedDataset(config, 1000, "dataset2")

    weights = [0.8, 0.2]
    mixture = _data_loader.MixtureDataset(
        [dataset1, dataset2],
        weights=weights,
        stop_on_empty_dataset=True,  # To get exact count
        seed=42,
    )

    samples = [mixture[i]["tag"] for i in range(500)]
    dataset1_count = samples.count("dataset1")
    dataset2_count = samples.count("dataset2")

    # Allow some statistical variance (within ±5%)
    expected_dataset1 = 500 * 0.8
    expected_dataset2 = 500 * 0.2
    assert abs(dataset1_count - expected_dataset1) < 25
    assert abs(dataset2_count - expected_dataset2) < 25
