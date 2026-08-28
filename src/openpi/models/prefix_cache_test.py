"""Tests for prefix_cache module."""

import numpy as np
import pytest

from openpi.models import prefix_cache


class TestFastArrayHash:
    """Tests for _fast_array_hash function."""

    def test_same_array_same_hash(self):
        """Same array content produces same hash."""
        arr = np.random.rand(100, 100)
        hash1 = prefix_cache.fast_array_hash(arr)
        hash2 = prefix_cache.fast_array_hash(arr)
        assert hash1 == hash2

    def test_different_array_different_hash(self):
        """Different array content produces different hash."""
        arr1 = np.zeros((100, 100))
        arr2 = np.ones((100, 100))
        hash1 = prefix_cache.fast_array_hash(arr1)
        hash2 = prefix_cache.fast_array_hash(arr2)
        assert hash1 != hash2

    def test_small_array_full_hash(self):
        """Small arrays use full content for hash."""
        arr = np.array([1, 2, 3])
        hash1 = prefix_cache.fast_array_hash(arr, sample_size=1000)
        hash2 = prefix_cache.fast_array_hash(arr, sample_size=1000)
        assert hash1 == hash2

    def test_large_array_sampled_hash(self):
        """Large arrays use sampling for hash."""
        arr = np.random.rand(10000)
        hash1 = prefix_cache.fast_array_hash(arr, sample_size=100)
        hash2 = prefix_cache.fast_array_hash(arr, sample_size=100)
        assert hash1 == hash2

    def test_different_shapes_different_hash(self):
        """Arrays with same values but different shapes have different hashes."""
        arr1 = np.ones((2, 3))
        arr2 = np.ones((3, 2))
        hash1 = prefix_cache.fast_array_hash(arr1)
        hash2 = prefix_cache.fast_array_hash(arr2)
        assert hash1 != hash2


class TestPrefixCacheKey:
    """Tests for PrefixCacheKey."""

    def test_equality(self):
        """Same content produces equal keys."""
        key1 = prefix_cache.PrefixCacheKey(
            image_hashes=(("img1", "abc123"),),
            prompt_hash="prompt_hash",
            batch_size=1,
        )
        key2 = prefix_cache.PrefixCacheKey(
            image_hashes=(("img1", "abc123"),),
            prompt_hash="prompt_hash",
            batch_size=1,
        )
        assert key1 == key2

    def test_different_image_hash(self):
        """Different image hashes produce different keys."""
        key1 = prefix_cache.PrefixCacheKey(
            image_hashes=(("img1", "abc123"),),
            prompt_hash="prompt_hash",
            batch_size=1,
        )
        key2 = prefix_cache.PrefixCacheKey(
            image_hashes=(("img1", "def456"),),
            prompt_hash="prompt_hash",
            batch_size=1,
        )
        assert key1 != key2

    def test_different_prompt_hash(self):
        """Different prompt hashes produce different keys."""
        key1 = prefix_cache.PrefixCacheKey(
            image_hashes=(("img1", "abc123"),),
            prompt_hash="prompt_hash_1",
            batch_size=1,
        )
        key2 = prefix_cache.PrefixCacheKey(
            image_hashes=(("img1", "abc123"),),
            prompt_hash="prompt_hash_2",
            batch_size=1,
        )
        assert key1 != key2

    def test_different_batch_size(self):
        """Different batch sizes produce different keys."""
        key1 = prefix_cache.PrefixCacheKey(
            image_hashes=(("img1", "abc123"),),
            prompt_hash="prompt_hash",
            batch_size=1,
        )
        key2 = prefix_cache.PrefixCacheKey(
            image_hashes=(("img1", "abc123"),),
            prompt_hash="prompt_hash",
            batch_size=2,
        )
        assert key1 != key2

    def test_hashable(self):
        """Keys are hashable and can be used in sets/dicts."""
        key = prefix_cache.PrefixCacheKey(
            image_hashes=(("img1", "abc123"),),
            prompt_hash="prompt_hash",
            batch_size=1,
        )
        key_set = {key}
        assert key in key_set


class TestPrefixCacheKeyFromObservation:
    """Tests for PrefixCacheKey.from_observation."""

    @pytest.fixture
    def mock_observation(self):
        """Create a mock observation for testing."""
        from dataclasses import dataclass

        @dataclass
        class MockObservation:
            images: dict
            image_masks: dict
            state: np.ndarray
            tokenized_prompt: np.ndarray | None = None
            tokenized_prompt_mask: np.ndarray | None = None

        return MockObservation

    def test_same_observation_same_key(self, mock_observation):
        """Same observation produces same key."""
        obs = mock_observation(
            images={"img1": np.zeros((1, 224, 224, 3))},
            image_masks={"img1": np.array([True])},
            state=np.zeros((1, 14)),
            tokenized_prompt=np.array([[1, 2, 3]]),
            tokenized_prompt_mask=np.array([[True, True, True]]),
        )
        key1 = prefix_cache.PrefixCacheKey.from_observation(obs)
        key2 = prefix_cache.PrefixCacheKey.from_observation(obs)
        assert key1 == key2

    def test_different_image_different_key(self, mock_observation):
        """Different image produces different key."""
        obs1 = mock_observation(
            images={"img1": np.zeros((1, 224, 224, 3))},
            image_masks={"img1": np.array([True])},
            state=np.zeros((1, 14)),
        )
        obs2 = mock_observation(
            images={"img1": np.ones((1, 224, 224, 3))},
            image_masks={"img1": np.array([True])},
            state=np.zeros((1, 14)),
        )
        key1 = prefix_cache.PrefixCacheKey.from_observation(obs1)
        key2 = prefix_cache.PrefixCacheKey.from_observation(obs2)
        assert key1 != key2

    def test_no_prompt_key(self, mock_observation):
        """Observation without prompt has None prompt_hash."""
        obs = mock_observation(
            images={"img1": np.zeros((1, 224, 224, 3))},
            image_masks={"img1": np.array([True])},
            state=np.zeros((1, 14)),
            tokenized_prompt=None,
        )
        key = prefix_cache.PrefixCacheKey.from_observation(obs)
        assert key.prompt_hash is None

    def test_multiple_images_sorted(self, mock_observation):
        """Multiple images are sorted by name for consistent ordering."""
        obs1 = mock_observation(
            images={
                "img_b": np.zeros((1, 224, 224, 3)),
                "img_a": np.ones((1, 224, 224, 3)),
            },
            image_masks={
                "img_b": np.array([True]),
                "img_a": np.array([True]),
            },
            state=np.zeros((1, 14)),
        )
        obs2 = mock_observation(
            images={
                "img_a": np.ones((1, 224, 224, 3)),
                "img_b": np.zeros((1, 224, 224, 3)),
            },
            image_masks={
                "img_a": np.array([True]),
                "img_b": np.array([True]),
            },
            state=np.zeros((1, 14)),
        )
        key1 = prefix_cache.PrefixCacheKey.from_observation(obs1)
        key2 = prefix_cache.PrefixCacheKey.from_observation(obs2)
        assert key1 == key2


class TestPrefixCache:
    """Tests for PrefixCache."""

    @pytest.fixture
    def mock_observation(self):
        """Create a mock observation for testing."""
        from dataclasses import dataclass

        @dataclass
        class MockObservation:
            images: dict
            image_masks: dict
            state: np.ndarray
            tokenized_prompt: np.ndarray | None = None
            tokenized_prompt_mask: np.ndarray | None = None

        return MockObservation

    def test_is_valid_for_same_observation(self, mock_observation):
        """Cache is valid for same observation."""
        obs = mock_observation(
            images={"img1": np.zeros((1, 224, 224, 3))},
            image_masks={"img1": np.array([True])},
            state=np.zeros((1, 14)),
        )
        key = prefix_cache.PrefixCacheKey.from_observation(obs)
        cache = prefix_cache.PrefixCache(
            key=key,
            prefix_tokens=np.zeros((1, 100, 256)),
            prefix_mask=np.ones((1, 100), dtype=bool),
            prefix_ar_mask=np.zeros((100,), dtype=bool),
            kv_cache=(np.zeros((18, 1, 100, 1, 256)), np.zeros((18, 1, 100, 1, 256))),
        )
        assert cache.is_valid_for(obs)

    def test_is_valid_for_different_observation(self, mock_observation):
        """Cache is invalid for different observation."""
        obs1 = mock_observation(
            images={"img1": np.zeros((1, 224, 224, 3))},
            image_masks={"img1": np.array([True])},
            state=np.zeros((1, 14)),
        )
        obs2 = mock_observation(
            images={"img1": np.ones((1, 224, 224, 3))},
            image_masks={"img1": np.array([True])},
            state=np.zeros((1, 14)),
        )
        key = prefix_cache.PrefixCacheKey.from_observation(obs1)
        cache = prefix_cache.PrefixCache(
            key=key,
            prefix_tokens=np.zeros((1, 100, 256)),
            prefix_mask=np.ones((1, 100), dtype=bool),
            prefix_ar_mask=np.zeros((100,), dtype=bool),
            kv_cache=(np.zeros((18, 1, 100, 1, 256)), np.zeros((18, 1, 100, 1, 256))),
        )
        assert not cache.is_valid_for(obs2)

    def test_get_prefix_len(self):
        """get_prefix_len returns correct length."""
        key = prefix_cache.PrefixCacheKey(
            image_hashes=(),
            prompt_hash=None,
            batch_size=1,
        )
        cache = prefix_cache.PrefixCache(
            key=key,
            prefix_tokens=np.zeros((1, 150, 256)),
            prefix_mask=np.ones((1, 150), dtype=bool),
            prefix_ar_mask=np.zeros((150,), dtype=bool),
            kv_cache=(np.zeros((18, 1, 150, 1, 256)), np.zeros((18, 1, 150, 1, 256))),
        )
        assert cache.get_prefix_len() == 150
