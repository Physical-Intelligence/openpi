"""Compare transformed training samples with configured normalization statistics."""

import compute_norm_stats
import numpy as np
import tqdm
import tyro

import openpi.training.config as _config
from openpi.training.norm_stats_report import analyze_norm_stats
from openpi.training.norm_stats_report import format_norm_stats_report


def main(config_name: str, max_frames: int = 5000) -> None:
    """Print a read-only compatibility report for a training configuration."""
    if max_frames < 1:
        raise ValueError("max_frames must be positive")

    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.norm_stats is None:
        raise ValueError(
            "The selected config does not load normalization statistics. "
            "Run compute_norm_stats.py first or configure checkpoint assets to compare against."
        )

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = compute_norm_stats.create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = compute_norm_stats.create_torch_dataloader(
            data_config,
            config.model.action_horizon,
            config.batch_size,
            config.model,
            config.num_workers,
            max_frames,
        )

    keys = ("state", "actions")
    samples: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Checking normalization stats"):
        for key in keys:
            if key not in batch:
                raise KeyError(f"Transformed data does not contain required key '{key}'.")
            samples[key].append(np.asarray(batch[key]))

    for key, batches in samples.items():
        if key not in data_config.norm_stats:
            print(f"{key}\n  result: WARNING (missing configured normalization statistics)\n")
            continue
        if not batches:
            raise ValueError("The data loader produced no samples. Increase max_frames or check the dataset config.")
        report = analyze_norm_stats(key, np.concatenate(batches), data_config.norm_stats[key])
        print(format_norm_stats_report(report))
        print()


if __name__ == "__main__":
    tyro.cli(main)
