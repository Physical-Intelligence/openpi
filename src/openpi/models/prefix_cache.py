"""Prefix cache for KV cache reuse during continuous inference.

This module provides caching infrastructure to avoid redundant computation
of prefix embeddings (images + language) when the observation hasn't changed
between consecutive inference calls.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import jax

    from openpi.models import model as _model

import numpy as np


def fast_array_hash(arr: np.ndarray | jax.Array, sample_size: int = 1000) -> str:
    """Compute a fast hash of an array using sampling for large arrays.

    For small arrays, computes the hash of the entire content plus metadata.
    For large arrays, samples uniformly distributed points to balance
    speed and collision resistance.

    Args:
        arr: The array to hash.
        sample_size: Number of samples for large arrays.

    Returns:
        MD5 hex digest of the array content.
    """
    arr = np.asarray(arr)
    flat = arr.ravel()

    if flat.size <= sample_size:
        data = flat.tobytes() + str(arr.shape).encode() + str(arr.dtype).encode()
    else:
        indices = np.linspace(0, flat.size - 1, sample_size, dtype=np.int64)
        sampled = flat[indices]
        data = sampled.tobytes() + str(arr.shape).encode() + str(arr.dtype).encode()

    return hashlib.md5(data).hexdigest()


@dataclasses.dataclass(frozen=True)
class PrefixCacheKey:
    """Cache key based on image and prompt content hashes.

    This is used to determine whether the cached prefix can be reused.
    The key is immutable and hashable.
    """

    image_hashes: tuple[tuple[str, str], ...]
    prompt_hash: str | None
    batch_size: int

    @classmethod
    def from_observation(cls, observation: _model.Observation) -> PrefixCacheKey:
        """Compute cache key from an Observation.

        Args:
            observation: The observation to compute the key from.

        Returns:
            A PrefixCacheKey that uniquely identifies this observation's prefix.
        """
        image_hashes = []
        for name in sorted(observation.images.keys()):
            image = observation.images[name]
            h = fast_array_hash(image)
            image_hashes.append((name, h))

        prompt_hash = None
        if observation.tokenized_prompt is not None:
            prompt_hash = fast_array_hash(observation.tokenized_prompt)

        batch_size = int(np.asarray(observation.state).shape[0])

        return cls(
            image_hashes=tuple(image_hashes),
            prompt_hash=prompt_hash,
            batch_size=batch_size,
        )


@dataclasses.dataclass
class PrefixCache:
    """Stores computed prefix results for reuse.

    Contains the cache key and all intermediate results needed to skip
    prefix computation on subsequent inference calls.
    """

    key: PrefixCacheKey

    prefix_tokens: jax.Array
    prefix_mask: jax.Array
    prefix_ar_mask: jax.Array
    kv_cache: tuple[jax.Array, jax.Array]

    def is_valid_for(self, observation: _model.Observation) -> bool:
        """Check if this cache is valid for the given observation.

        Args:
            observation: The observation to check against.

        Returns:
            True if the cache can be reused, False otherwise.
        """
        new_key = PrefixCacheKey.from_observation(observation)
        return self.key == new_key

    def get_prefix_len(self) -> int:
        """Get the length of the cached prefix sequence."""
        return int(self.prefix_tokens.shape[1])
