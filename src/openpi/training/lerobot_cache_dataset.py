"""
Custom LeRobot dataset loader that works with ONLY metadata + HF cache.

This allows training without the original Parquet files, saving 152GB of storage.

Required structure:
- meta/ directory (92MB) - Contains info.json, episodes.jsonl, etc.
- HF cache (152GB) - Arrow files created by HuggingFace datasets

NOT required:
- data/ directory (152GB) - Original Parquet files
"""

import os
from pathlib import Path

import datasets
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset


class CachedLeRobotDataset(lerobot_dataset.LeRobotDataset):
    """
    Modified LeRobotDataset that loads directly from HuggingFace cache.

    This bypasses the requirement for original Parquet files in data/ directory.
    """

    def __init__(self, repo_id: str, **kwargs):
        # Get cache directory from environment
        cache_dir = os.environ.get("HF_DATASETS_CACHE",
                                   os.path.join(os.environ.get("HF_HOME", "~/.cache/huggingface"), "datasets"))
        self.cache_dir = Path(cache_dir).expanduser().resolve()

        # Store kwargs for later use
        self._init_kwargs = kwargs

        # Call parent __init__ but it will fail when trying to load dataset
        # We catch all possible exceptions including HuggingFace errors
        try:
            super().__init__(repo_id=repo_id, **kwargs)
        except Exception as e:
            # Expected to fail because data/ directory doesn't exist
            # The parent __init__ has already set up all attributes up to line 498
            # including self.meta, self.root, self.episodes, self.delta_timestamps, etc.
            print(f"Standard initialization failed at dataset loading (expected): {type(e).__name__}")
            print("Falling back to cache-only initialization")

            # Verify metadata was loaded (should have been done by parent before failure)
            if not hasattr(self, 'meta'):
                raise RuntimeError(
                    "Parent initialization failed before loading metadata. "
                    "This should not happen - check that meta/ directory exists."
                )

            # Load dataset from cache instead of Parquet files
            self.hf_dataset = self.load_hf_dataset_from_cache()

            # Set up episode data index (parent would do this after loading dataset)
            self.episode_data_index = lerobot_dataset.get_episode_data_index(
                self.meta.episodes,
                self.episodes
            )

            # Check timestamps (parent does this at lines 507-511)
            import torch
            timestamps = torch.stack(self.hf_dataset["timestamp"]).numpy()
            episode_indices = torch.stack(self.hf_dataset["episode_index"]).numpy()
            ep_data_index_np = {k: t.numpy() for k, t in self.episode_data_index.items()}
            lerobot_dataset.check_timestamps_sync(
                timestamps, episode_indices, ep_data_index_np, self.fps, self.tolerance_s
            )

            # Setup delta_indices if needed (parent does this at lines 514-516)
            if self.delta_timestamps is not None:
                lerobot_dataset.check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
                self.delta_indices = lerobot_dataset.get_delta_indices(
                    self.delta_timestamps,
                    self.fps
                )

    def load_hf_dataset_from_cache(self) -> datasets.Dataset:
        """
        Load dataset directly from HuggingFace cache without requiring Parquet files.

        We directly load the cached Arrow dataset instead of going through load_dataset().

        Returns:
            datasets.Dataset: The cached dataset in Arrow format
        """
        # Search for cached arrow files
        cache_base = self.cache_dir / "parquet"

        if not cache_base.exists():
            raise FileNotFoundError(
                f"HuggingFace cache not found at {self.cache_dir}. "
                f"Please run training once with original dataset to create cache, "
                f"or set HF_DATASETS_CACHE environment variable correctly."
            )

        # Find the cache directory (it has a hash-based structure)
        arrow_files = list(cache_base.glob("**/parquet-train-*.arrow"))

        if not arrow_files:
            raise FileNotFoundError(
                f"No cached Arrow files found in {cache_base}. "
                f"The cache may not have been created yet. Run training once with "
                f"the original Parquet dataset to generate the cache."
            )

        cache_dataset_dir = arrow_files[0].parent
        print(f"Found {len(arrow_files)} cached Arrow files in {cache_dataset_dir}")
        print(f"Loading dataset directly from HuggingFace cache directory")

        # The cache directory is a complete HuggingFace dataset cache
        # Use load_from_disk() to load it
        try:
            hf_dataset = datasets.load_from_disk(str(cache_dataset_dir))
        except Exception as e:
            print(f"Error loading from disk: {e}")
            print("Attempting alternative method: loading arrow files directly")

            # Fallback: try to construct dataset from arrow files
            # These are memory-mapped arrow files
            arrow_file_paths = sorted([str(f) for f in arrow_files])

            # Use datasets' built-in method to load memory-mapped files
            hf_dataset = datasets.Dataset.from_file(arrow_file_paths[0])

            # If there are multiple shards, concatenate them
            if len(arrow_file_paths) > 1:
                all_datasets = [datasets.Dataset.from_file(f) for f in arrow_file_paths]
                hf_dataset = datasets.concatenate_datasets(all_datasets)

        # Set transform to convert to torch
        hf_dataset.set_transform(lerobot_dataset.hf_transform_to_torch)

        return hf_dataset

    def load_hf_dataset(self) -> datasets.Dataset:
        """
        Override parent method to load from cache instead of Parquet files.
        """
        return self.load_hf_dataset_from_cache()


def create_cached_torch_dataset(
    data_config,
    action_horizon: int,
    model_config
):
    """
    Drop-in replacement for create_torch_dataset that uses cache-only loading.

    Usage:
        # In data_loader.py, replace:
        # dataset = create_torch_dataset(data_config, action_horizon, model_config)
        # with:
        from openpi.training.lerobot_cache_dataset import create_cached_torch_dataset
        dataset = create_cached_torch_dataset(data_config, action_horizon, model_config)
    """
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        # Use fake dataset
        from openpi.training.data_loader import FakeDataset
        return FakeDataset(model_config, num_samples=1024)

    # Use cached dataset loader
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = CachedLeRobotDataset(
        repo_id=data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)]
            for key in data_config.action_sequence_keys
        },
    )

    if data_config.prompt_from_task:
        from openpi.training.data_loader import TransformedDataset
        import openpi.transforms as _transforms
        dataset = TransformedDataset(
            dataset,
            [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)]
        )

    return dataset
