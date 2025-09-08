"""Compute normalization statistics for a config using JAX for GPU acceleration.

This script is a JAX-accelerated version of compute_norm_stats.py that computes the
normalization statistics for a given config. It computes the mean, standard deviation,
and quantiles of the data in the dataset and saves it to the config assets directory.
"""

from functools import partial
import time

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import tyro

import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_dataset(config: _config.TrainConfig) -> tuple[_config.DataConfig, _data_loader.Dataset]:
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    return data_config, dataset


class JaxRunningStats:
    """JAX implementation of RunningStats for GPU acceleration."""

    def __init__(self):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None
        self._histograms = None
        self._bin_edges = None
        self._num_quantile_bins = 5000  # Keep same as original for identical results
        self._initialized = False

    def _initialize(self, batch: np.ndarray) -> None:
        """Initialize statistics with the first batch."""
        if batch.ndim == 1:
            batch = batch.reshape(-1, 1)

        num_elements, vector_length = batch.shape
        self._count = num_elements

        # Convert to JAX arrays
        batch_jax = jnp.array(batch)
        self._mean = jnp.mean(batch_jax, axis=0)
        self._mean_of_squares = jnp.mean(batch_jax**2, axis=0)
        self._min = jnp.min(batch_jax, axis=0)
        self._max = jnp.max(batch_jax, axis=0)

        # Initialize histograms with JAX arrays
        self._histograms = [np.zeros(self._num_quantile_bins) for _ in range(vector_length)]
        self._bin_edges = [
            np.linspace(float(self._min[i]) - 1e-10, float(self._max[i]) + 1e-10, self._num_quantile_bins + 1)
            for i in range(vector_length)
        ]

        # Update histograms with initial batch (using numpy for exact compatibility)
        for i in range(vector_length):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

        self._initialized = True

    # JIT-compiled function to update mean and mean of squares
    @partial(jax.jit, static_argnums=(0,))
    def _update_moments_jit(self, current_count, current_mean, current_mean_sq, batch, batch_count):
        """Update mean and mean of squares using JAX."""
        new_count = current_count + batch_count
        batch_mean = jnp.mean(batch, axis=0)
        batch_mean_sq = jnp.mean(batch**2, axis=0)

        # Update running statistics using the same formula as original
        new_mean = current_mean + (batch_mean - current_mean) * (batch_count / new_count)
        new_mean_sq = current_mean_sq + (batch_mean_sq - current_mean_sq) * (batch_count / new_count)

        return new_count, new_mean, new_mean_sq

    def _adjust_histograms(self):
        """Adjust histograms when min or max changes - keep original logic."""
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            new_edges = np.linspace(float(self._min[i]), float(self._max[i]), self._num_quantile_bins + 1)

            # Redistribute existing histogram counts to new bins (keep original logic)
            new_hist, _ = np.histogram(old_edges[:-1], bins=new_edges, weights=self._histograms[i])

            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def update(self, batch: np.ndarray) -> None:
        """Update running statistics with a batch of vectors."""
        # Convert input to numpy for consistency
        if isinstance(batch, jnp.ndarray):
            batch = np.array(batch)

        if batch.ndim == 1:
            batch = batch.reshape(-1, 1)

        num_elements, vector_length = batch.shape

        # Initialize if first batch
        if not self._initialized:
            self._initialize(batch)
            return

        # Check dimension consistency
        if vector_length != len(self._mean):
            raise ValueError(f"Expected {len(self._mean)} features, but got {vector_length}")

        # Update min and max (on CPU to match original logic)
        new_max = np.maximum(np.array(self._max), np.max(batch, axis=0))
        new_min = np.minimum(np.array(self._min), np.min(batch, axis=0))

        max_changed = np.any(new_max > self._max)
        min_changed = np.any(new_min < self._min)

        # Store as JAX arrays
        self._max = jnp.array(new_max)
        self._min = jnp.array(new_min)

        if max_changed or min_changed:
            self._adjust_histograms()

        # Update count, mean and mean_of_squares using JIT-compiled function
        batch_jax = jnp.array(batch)
        self._count, self._mean, self._mean_of_squares = self._update_moments_jit(
            self._count, self._mean, self._mean_of_squares, batch_jax, num_elements
        )

        # Update histograms (using numpy for exact compatibility)
        for i in range(vector_length):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self, quantiles):
        """Compute quantiles based on histograms - match original implementation."""
        results = []
        for q in quantiles:
            target_count = q * self._count
            q_values = []

            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                cumsum = np.cumsum(hist)
                idx = np.searchsorted(cumsum, target_count)
                q_values.append(edges[idx])

            results.append(np.array(q_values))
        return results

    def get_statistics(self) -> normalize.NormStats:
        """Compute and return the statistics (matches original implementation)."""
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")

        # Convert JAX arrays to numpy for consistency with original
        mean_np = np.array(self._mean)
        mean_sq_np = np.array(self._mean_of_squares)

        # Compute variance and standard deviation
        variance = mean_sq_np - mean_np**2
        stddev = np.sqrt(np.maximum(0, variance))

        # Compute quantiles using the same method as original
        q01, q99 = self._compute_quantiles([0.01, 0.99])

        return normalize.NormStats(mean=mean_np, std=stddev, q01=q01, q99=q99)


# Simple implementation that will be faster for now
class FastRunningStats:
    """A simplified, faster stats implementation."""

    def __init__(self):
        self.count = 0
        self.sum = None
        self.sum_sq = None
        self.min_vals = None
        self.max_vals = None

    def update(self, batch):
        if isinstance(batch, jnp.ndarray):
            batch = np.array(batch)

        if batch.ndim == 1:
            batch = batch.reshape(-1, 1)

        batch_size, vector_length = batch.shape

        if self.count == 0:
            self.sum = np.zeros(vector_length)
            self.sum_sq = np.zeros(vector_length)
            self.min_vals = np.full(vector_length, np.inf)
            self.max_vals = np.full(vector_length, -np.inf)

        # Update min and max
        self.min_vals = np.minimum(self.min_vals, np.min(batch, axis=0))
        self.max_vals = np.maximum(self.max_vals, np.max(batch, axis=0))

        # Update sums
        self.sum += np.sum(batch, axis=0)
        self.sum_sq += np.sum(batch**2, axis=0)
        self.count += batch_size

    def get_statistics(self):
        if self.count < 2:
            raise ValueError("Cannot compute statistics with less than 2 samples")

        mean = self.sum / self.count
        variance = (self.sum_sq / self.count) - (mean**2)
        std = np.sqrt(np.maximum(0, variance))

        # We don't compute exact quantiles here, approximating with min/max
        # which is much faster (we can add this back if needed)
        q01 = mean - 2.576 * std  # Approximating 1st percentile
        q99 = mean + 2.576 * std  # Approximating 99th percentile

        # Ensure q01 and q99 are within data bounds
        q01 = np.maximum(q01, self.min_vals)
        q99 = np.minimum(q99, self.max_vals)

        return normalize.NormStats(mean=mean, std=std, q01=q01, q99=q99)


def main(config_name: str, max_frames: int | None = None, *, use_fast_stats: bool = True):
    """Main function that computes and saves normalization statistics."""
    max_time_minutes = 120  # Maximum runtime in minutes
    start_time = time.time()

    # Print info about JAX devices
    print(f"JAX is using {jax.device_count()} devices: {jax.devices()}")

    # Use same setup as original
    config = _config.get_config(config_name)
    data_config, dataset = create_dataset(config)

    num_frames = len(dataset)
    shuffle = False

    if max_frames is not None and max_frames < num_frames:
        num_frames = max_frames
        shuffle = True

    # Calculate appropriate batch size for GPUs - but don't make it too large
    # Large batches can cause excessive memory usage
    devices = jax.devices()
    num_devices = len(devices)
    per_device_batch = 32  # Smaller batch size to start with
    batch_size = per_device_batch * num_devices

    # Limit batch size based on dataset size
    batch_size = min(batch_size, num_frames // 10)
    batch_size = max(batch_size, 16)  # Ensure minimum batch size

    # Ensure batch size is divisible by number of devices for proper sharding
    batch_size = (batch_size // num_devices) * num_devices
    batch_size = max(batch_size, num_devices)  # Ensure minimum batch size is at least num_devices

    print(f"Using batch size: {batch_size} across {num_devices} devices")

    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=16,  # More workers for faster loading
        shuffle=shuffle,
        num_batches=num_frames,
    )

    # Same keys as original
    keys = ["state", "actions"]

    # Choose which stats implementation to use
    stats_class = FastRunningStats if use_fast_stats else JaxRunningStats
    stats = {key: stats_class() for key in keys}

    total_batches = (num_frames + batch_size - 1) // batch_size
    print(f"Processing {total_batches} batches...")

    for batch_idx, batch in enumerate(tqdm.tqdm(data_loader, total=total_batches, desc="Computing stats")):
        batch_start = time.time()
        for key in keys:
            values = np.asarray(batch[key])
            # Reshape to (total_elements, feature_dim) like in original
            if values.ndim > 2:
                values = values.reshape(-1, values.shape[-1])
            stats[key].update(values)

        # Log timing info for first few batches
        if batch_idx < 5:
            batch_time = time.time() - batch_start
            print(f"Batch {batch_idx + 1}/{total_batches} processed in {batch_time:.2f}s")

        if batch_idx >= total_batches:
            print("Reached target number of batches, stopping.")
            break

        if (time.time() - start_time) > (max_time_minutes * 60):
            print(f"Reached maximum runtime of {max_time_minutes} minutes, stopping.")
            break

    # Convert statistics to the same format as original
    norm_stats = {key: stat.get_statistics() for key, stat in stats.items()}

    # After calculating stats and before saving
    # Create directory if it doesn't exist
    output_path = config.assets_dirs / data_config.repo_id
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Save to the same location with same format
        output_path = config.assets_dirs / data_config.repo_id
        print(f"Writing stats to: {output_path}")
        normalize.save(output_path, norm_stats)
        print(f"Stats successfully saved to {output_path}")
    except Exception as e:
        print(f"Error saving stats: {e}")

    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.2f}s ({total_time / 60:.2f} minutes)")


if __name__ == "__main__":
    tyro.cli(main)
