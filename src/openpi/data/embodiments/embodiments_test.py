"""Tests for the multi-embodiment data pipeline."""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pytest

import openpi.shared.normalize as _normalize
import openpi.transforms as _transforms
from openpi.data.embodiments.config import EmbodimentConfig
from openpi.data.embodiments.data_loader import (
    MultiEmbodimentDataset,
    WeightedMultiEmbodimentSampler,
    load_embodiment_norm_stats,
)
from openpi.data.embodiments.registry import TransformRegistry


# ---------------------------------------------------------------------------
# EmbodimentConfig tests
# ---------------------------------------------------------------------------


class TestEmbodimentConfig:
    def test_basic_creation(self) -> None:
        cfg = EmbodimentConfig(
            name="test_robot",
            tag_id=0,
            action_dim=7,
            data_path="fake",
        )
        assert cfg.name == "test_robot"
        assert cfg.tag_id == 0
        assert cfg.action_dim == 7
        assert cfg.sampling_weight == 1.0

    def test_get_repack_transforms_from_key_mapping(self) -> None:
        cfg = EmbodimentConfig(
            name="test",
            tag_id=0,
            action_dim=7,
            data_path="fake",
            key_mapping={
                "state": "observation.state",
                "actions": "action",
            },
        )
        group = cfg.get_repack_transforms()
        assert len(group.inputs) == 1
        assert isinstance(group.inputs[0], _transforms.RepackTransform)

    def test_get_repack_transforms_explicit(self) -> None:
        explicit = _transforms.Group(
            inputs=[_transforms.RepackTransform({"state": "observation.state"})]
        )
        cfg = EmbodimentConfig(
            name="test",
            tag_id=0,
            action_dim=7,
            data_path="fake",
            repack_transforms=explicit,
        )
        assert cfg.get_repack_transforms() is explicit

    def test_get_repack_transforms_empty(self) -> None:
        cfg = EmbodimentConfig(name="test", tag_id=0, action_dim=7, data_path="fake")
        group = cfg.get_repack_transforms()
        assert len(group.inputs) == 0
        assert len(group.outputs) == 0


# ---------------------------------------------------------------------------
# TransformRegistry tests
# ---------------------------------------------------------------------------


class TestTransformRegistry:
    def test_register_and_get(self) -> None:
        registry = TransformRegistry()
        registry.register(
            "robot_a",
            data_transforms=lambda: _transforms.Group(inputs=[_transforms.InjectEmbodimentId(0)]),
        )
        group = registry.get_data_transforms("robot_a")
        assert len(group.inputs) == 1

    def test_missing_returns_empty(self) -> None:
        registry = TransformRegistry()
        group = registry.get_data_transforms("nonexistent")
        assert len(group.inputs) == 0

    def test_registered_names(self) -> None:
        registry = TransformRegistry()
        registry.register("b", data_transforms=lambda: _transforms.Group())
        registry.register("a", repack_transforms=lambda: _transforms.Group())
        assert registry.registered_names() == ["a", "b"]


# ---------------------------------------------------------------------------
# InjectEmbodimentId transform tests
# ---------------------------------------------------------------------------


class TestInjectEmbodimentId:
    def test_injects_id(self) -> None:
        transform = _transforms.InjectEmbodimentId(embodiment_id=42)
        data: dict = {"state": np.array([1.0, 2.0])}
        result = transform(data)
        assert "embodiment_id" in result
        assert result["embodiment_id"] == 42
        assert result["embodiment_id"].dtype == np.int32

    def test_does_not_remove_existing_keys(self) -> None:
        transform = _transforms.InjectEmbodimentId(embodiment_id=0)
        data: dict = {"state": np.array([1.0]), "actions": np.array([[1.0, 2.0]])}
        result = transform(data)
        assert "state" in result
        assert "actions" in result
        assert "embodiment_id" in result


# ---------------------------------------------------------------------------
# Hierarchical norm stats loading
# ---------------------------------------------------------------------------


class TestLoadEmbodimentNormStats:
    def test_load_existing(self, tmp_path: pathlib.Path) -> None:
        emb = EmbodimentConfig(name="robot_a", tag_id=0, action_dim=7, data_path="fake")
        stats_dir = tmp_path / "robot_a"
        stats_dir.mkdir()
        norm_stats = {
            "state": _normalize.NormStats(mean=np.zeros(7), std=np.ones(7)),
            "actions": _normalize.NormStats(mean=np.zeros(7), std=np.ones(7)),
        }
        _normalize.save(stats_dir, norm_stats)

        loaded = load_embodiment_norm_stats(tmp_path, emb)
        assert loaded is not None
        assert "state" in loaded
        assert "actions" in loaded
        np.testing.assert_array_equal(loaded["state"].mean, np.zeros(7))

    def test_missing_returns_none(self, tmp_path: pathlib.Path) -> None:
        emb = EmbodimentConfig(name="nonexistent", tag_id=0, action_dim=7, data_path="fake")
        result = load_embodiment_norm_stats(tmp_path, emb)
        assert result is None


# ---------------------------------------------------------------------------
# MultiEmbodimentDataset tests
# ---------------------------------------------------------------------------


class _FakeSimpleDataset:
    """Minimal dataset for testing without real data."""

    def __init__(self, size: int, action_dim: int, prefix: str = "") -> None:
        self._size = size
        self._action_dim = action_dim
        self._prefix = prefix

    def __getitem__(self, index) -> dict:
        idx = index.__index__() if hasattr(index, "__index__") else int(index)
        return {
            "state": np.full(self._action_dim, float(idx), dtype=np.float32),
            "actions": np.full((4, self._action_dim), float(idx), dtype=np.float32),
            "prompt": f"{self._prefix}task_{idx}",
        }

    def __len__(self) -> int:
        return self._size


class TestMultiEmbodimentDataset:
    def test_length(self) -> None:
        ds1 = _FakeSimpleDataset(10, 7, "a_")
        ds2 = _FakeSimpleDataset(20, 7, "b_")
        emb1 = EmbodimentConfig(name="a", tag_id=0, action_dim=7, data_path="fake")
        emb2 = EmbodimentConfig(name="b", tag_id=1, action_dim=7, data_path="fake")

        identity = [_transforms.InjectEmbodimentId(0)]
        multi_ds = MultiEmbodimentDataset(
            [ds1, ds2],
            [emb1, emb2],
            [identity, [_transforms.InjectEmbodimentId(1)]],
        )
        assert len(multi_ds) == 30

    def test_samples_contain_embodiment_id(self) -> None:
        ds1 = _FakeSimpleDataset(5, 7)
        emb1 = EmbodimentConfig(name="a", tag_id=42, action_dim=7, data_path="fake")
        transforms_list = [[_transforms.InjectEmbodimentId(42)]]

        multi_ds = MultiEmbodimentDataset([ds1], [emb1], transforms_list)
        sample = multi_ds[0]
        assert "embodiment_id" in sample
        assert sample["embodiment_id"] == 42

    def test_weights_normalised(self) -> None:
        ds1 = _FakeSimpleDataset(5, 7)
        ds2 = _FakeSimpleDataset(5, 7)
        emb1 = EmbodimentConfig(name="a", tag_id=0, action_dim=7, data_path="fake", sampling_weight=3.0)
        emb2 = EmbodimentConfig(name="b", tag_id=1, action_dim=7, data_path="fake", sampling_weight=1.0)

        multi_ds = MultiEmbodimentDataset(
            [ds1, ds2],
            [emb1, emb2],
            [[_transforms.InjectEmbodimentId(0)], [_transforms.InjectEmbodimentId(1)]],
        )
        np.testing.assert_allclose(multi_ds.weights, [0.75, 0.25])


class TestWeightedMultiEmbodimentSampler:
    def test_produces_correct_count(self) -> None:
        ds1 = _FakeSimpleDataset(10, 7)
        ds2 = _FakeSimpleDataset(10, 7)
        emb1 = EmbodimentConfig(name="a", tag_id=0, action_dim=7, data_path="fake", sampling_weight=1.0)
        emb2 = EmbodimentConfig(name="b", tag_id=1, action_dim=7, data_path="fake", sampling_weight=1.0)

        multi_ds = MultiEmbodimentDataset(
            [ds1, ds2],
            [emb1, emb2],
            [[_transforms.InjectEmbodimentId(0)], [_transforms.InjectEmbodimentId(1)]],
        )
        sampler = WeightedMultiEmbodimentSampler(multi_ds, num_samples=50, seed=0)
        indices = list(sampler)
        assert len(indices) == 50
        assert all(0 <= idx < 20 for idx in indices)


# ---------------------------------------------------------------------------
# Config integration tests
# ---------------------------------------------------------------------------


class TestMultiEmbodimentDataConfig:
    def test_config_exists(self) -> None:
        from openpi.training.config import get_config
        config = get_config("pi0_multi_embodiment_example")
        assert config is not None

    def test_config_has_embodiments(self) -> None:
        from openpi.training.config import MultiEmbodimentDataConfig, get_config
        config = get_config("pi0_multi_embodiment_example")
        assert isinstance(config.data, MultiEmbodimentDataConfig)
        assert len(config.data.embodiments) == 2
        assert config.data.embodiments[0].name == "aloha_sim"
        assert config.data.embodiments[1].name == "aloha_sim_insertion"
        assert config.data.embodiments[0].tag_id == 0
        assert config.data.embodiments[1].tag_id == 1
