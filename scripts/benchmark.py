"""
uv run python scripts/benchmark.py
"""

import dataclasses
import time

import numpy as np
import rich.console
import rich.table
import torch
import tqdm
import tyro

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    """Command line arguments for benchmarking."""

    # Number of warmup steps before timing.
    num_warmup: int = 2
    # Number of steps to benchmark.
    num_steps: int = 20
    # Whether to benchmark JAX model
    benchmark_jax: bool = True
    # Whether to benchmark PyTorch model
    benchmark_pytorch: bool = True
    # Whether to benchmark CUDA model
    benchmark_cuda: bool = True


class TimingRecorder:
    """Records timing measurements for different keys."""

    def __init__(self) -> None:
        self._timings: dict[str, list[float]] = {}

    def record(self, key: str, time_ms: float) -> None:
        """Record a timing measurement for the given key."""
        if key not in self._timings:
            self._timings[key] = []
        self._timings[key].append(time_ms)

    def get_stats(self, key: str) -> dict[str, float]:
        """Get statistics for the given key."""
        times = self._timings[key]
        return {
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "p25": float(np.quantile(times, 0.25)),
            "p50": float(np.quantile(times, 0.50)),
            "p75": float(np.quantile(times, 0.75)),
            "p90": float(np.quantile(times, 0.90)),
            "p95": float(np.quantile(times, 0.95)),
            "p99": float(np.quantile(times, 0.99)),
        }

    def print_all_stats(self) -> None:
        """Print statistics for all keys in a concise format."""

        table = rich.table.Table(
            title="[bold blue]Timing Statistics[/bold blue]",
            show_header=True,
            header_style="bold white",
            border_style="blue",
            title_justify="center",
        )

        # Add metric column with custom styling
        table.add_column("Metric", style="cyan", justify="left", no_wrap=True)

        # Add statistical columns with consistent styling
        stat_columns = [
            ("Mean", "yellow", "mean"),
            ("Std", "yellow", "std"),
            ("P25", "magenta", "p25"),
            ("P50", "magenta", "p50"),
            ("P75", "magenta", "p75"),
            ("P90", "magenta", "p90"),
            ("P95", "magenta", "p95"),
            ("P99", "magenta", "p99"),
        ]

        for name, style, _ in stat_columns:
            table.add_column(name, justify="right", style=style, no_wrap=True)

        # Add rows for each metric with formatted values
        for key in sorted(self._timings.keys()):
            stats = self.get_stats(key)
            values = [f"{stats[key]:.1f}" for _, _, key in stat_columns]
            table.add_row(key, *values)

        # Print with custom console settings
        console = rich.console.Console(width=None, highlight=True)
        console.print(table)


def random_observation_droid() -> dict:
    """Generate a random DROID observation."""
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }


def benchmark_model(
    model_name: str, checkpoint_dir: str, args: Args, observations: list[dict], fixed_noise: np.ndarray
) -> list[dict]:
    """Benchmark a single model and return its outputs.

    Args:
        model_name: Name of the model being benchmarked
        checkpoint_dir: Path to the model checkpoint
        args: Benchmark arguments
        observations: List of observations to use for inference
        fixed_noise: Optional fixed noise for deterministic inference

    Returns:
        List of action outputs from the model
    """
    # Load DROID policy
    config = _config.get_config("pi05_droid")
    policy = _policy_config.create_trained_policy(
        config,
        checkpoint_dir,
        pytorch_device="cuda",
        use_cuda_policy=model_name == "CUDA",
    )

    # Warmup runs
    for _ in range(args.num_warmup):
        policy.infer(random_observation_droid(), noise=fixed_noise)

    # Benchmark runs
    timing_recorder = TimingRecorder()
    actions = []

    for obs in tqdm.tqdm(observations, desc=f"Benchmarking {model_name}"):
        inference_start = time.time()
        action = policy.infer(obs, noise=fixed_noise)
        inference_time_ms = 1000 * (time.time() - inference_start)
        timing_recorder.record("inference_ms", inference_time_ms)

        # Record policy internal timings if available
        if hasattr(action, "get") and isinstance(action, dict):
            for key, value in action.get("policy_timing", {}).items():
                timing_recorder.record(f"policy_{key}", value)

        actions.append(action)

    print(f"\n{model_name} Results:")
    timing_recorder.print_all_stats()

    del policy  # Free up GPU memory
    torch.cuda.empty_cache()

    return actions


def compare_outputs(results: dict[str, list[dict]]) -> None:
    """Compare outputs from different model implementations.

    Args:
        results: Dictionary mapping model names to their action outputs
    """
    if len(results) < 2:
        return

    model_names = list(results.keys())
    print("\n" + "=" * 60)
    print("Comparing Model Outputs")
    print("=" * 60)

    # Use first model as reference
    reference_name = model_names[0]
    reference_actions = results[reference_name]

    for model_name in model_names[1:]:
        print(f"\nComparing {reference_name} vs {model_name}:")

        model_actions = results[model_name]
        differences = []

        for ref_action, model_action in zip(reference_actions, model_actions):  # noqa: B905
            # Extract action arrays
            ref_act = ref_action.get("actions", ref_action.get("action"))
            mod_act = model_action.get("actions", model_action.get("action"))

            if ref_act is not None and mod_act is not None:
                diff = np.abs(ref_act - mod_act)
                differences.append(diff)

        if differences:
            all_diffs = np.concatenate([d.flatten() for d in differences])

            # Get action value ranges for context
            ref_actions_flat = np.concatenate(
                [
                    (a.get("actions", a.get("action"))).flatten()
                    for a in reference_actions
                    if a.get("actions") is not None or a.get("action") is not None
                ]
            )

            # Calculate percentage within thresholds
            thresholds = [0.001, 0.01, 0.1, 1.0]
            percentages = {thresh: 100 * np.mean(all_diffs < thresh) for thresh in thresholds}

            table = rich.table.Table(
                title=f"[bold blue]{reference_name} vs {model_name}[/bold blue]",
                show_header=True,
                header_style="bold white",
                border_style="blue",
            )

            table.add_column("Metric", style="cyan", justify="left")
            table.add_column("Value", style="yellow", justify="right")

            # Action value ranges
            table.add_row("Action Min", f"{np.min(ref_actions_flat):.6f}")
            table.add_row("Action Max", f"{np.max(ref_actions_flat):.6f}")
            table.add_row("Action Mean", f"{np.mean(ref_actions_flat):.6f}")
            table.add_row("", "")  # Separator

            # Differences
            table.add_row("Mean Absolute Diff", f"{np.mean(all_diffs):.6f}")
            table.add_row("Max Absolute Diff", f"{np.max(all_diffs):.6f}")
            table.add_row("Median Absolute Diff", f"{np.median(all_diffs):.6f}")
            table.add_row("", "")  # Separator
            for thresh in thresholds:
                table.add_row(f"% within {thresh}", f"{percentages[thresh]:.2f}%")

            console = rich.console.Console()
            console.print(table)
        else:
            print("  Could not extract action arrays for comparison")


def main(args: Args) -> None:
    # Generate shared observations for fair comparison
    print(f"Generating {args.num_steps} random observations...")
    observations = [random_observation_droid() for _ in range(args.num_steps)]

    # Generate fixed noise for deterministic diffusion (action_horizon=15, action_dim=32 for DROID)
    np.random.seed(42)
    fixed_noise = np.random.randn(1, 15, 32).astype(np.float32)

    # Store results from each model
    results: dict[str, list[dict]] = {}

    if args.benchmark_jax:
        results["JAX"] = benchmark_model(
            model_name="JAX",
            checkpoint_dir="gs://openpi-assets/checkpoints/pi05_droid",
            args=args,
            observations=observations,
            fixed_noise=fixed_noise,
        )

    if args.benchmark_pytorch:
        results["PyTorch"] = benchmark_model(
            model_name="PyTorch",
            checkpoint_dir="gs://openpi-assets/checkpoints/pi05_droid_pytorch",
            args=args,
            observations=observations,
            fixed_noise=fixed_noise,
        )

    if args.benchmark_cuda:
        results["CUDA"] = benchmark_model(
            model_name="CUDA",
            checkpoint_dir="gs://openpi-assets/checkpoints/pi05_droid_pytorch",
            args=args,
            observations=observations,
            fixed_noise=fixed_noise,
        )

    # Compare outputs across implementations
    compare_outputs(results)


if __name__ == "__main__":
    main(tyro.cli(Args))
