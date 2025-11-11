#!/usr/bin/env python3
"""Download specific tasks from BEHAVIOR-1K dataset using git sparse-checkout.

This script downloads only the specified tasks instead of the entire 1.5TB dataset.
Uses git sparse-checkout to avoid Hugging Face API rate limits.

Usage:
    python scripts/download_behavior_tasks.py --tasks turning_on_radio picking_up_trash
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


# Task name to task folder mapping (from BEHAVIOR-1K website)
TASK_MAPPING = {
    "turning_on_radio": "task-0000",
    "picking_up_trash": "task-0001",
    "putting_away_halloween_decorations": "task-0002",
    "cleaning_up_plates_and_food": "task-0003",
    "can_meat": "task-0004",
    "setting_mousetraps": "task-0005",
    "hiding_easter_eggs": "task-0006",
    "picking_up_toys": "task-0007",
    "rearranging_kitchen_furniture": "task-0008",
    "putting_up_christmas_decorations_inside": "task-0009",
    # Add more as needed...
}


def download_specific_tasks_git(tasks: list[str], local_dir: str | None = None):
    """Download only specific tasks using git sparse-checkout.

    This method avoids Hugging Face API rate limits by using git directly.

    Args:
        tasks: List of task names (e.g., ["turning_on_radio", "picking_up_trash"])
        local_dir: Local directory to clone into (default: ~/.cache/huggingface/datasets/behavior-1k)
    """
    repo_url = "https://huggingface.co/datasets/behavior-1k/2025-challenge-demos"

    # Convert task names to folder names
    task_folders = []
    for task in tasks:
        if task not in TASK_MAPPING:
            raise ValueError(
                f"Unknown task: {task}. Available tasks: {list(TASK_MAPPING.keys())}"
            )
        task_folders.append(TASK_MAPPING[task])

    # Default local directory
    if local_dir is None:
        local_dir = Path.home() / ".cache" / "huggingface" / "datasets" / "behavior-1k" / "2025-challenge-demos"
    else:
        local_dir = Path(local_dir)

    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading tasks: {tasks}")
    print(f"Task folders: {task_folders}")
    print(f"Local directory: {local_dir}")

    # Check if already cloned
    if (local_dir / ".git").exists():
        print("\nRepository already exists, updating sparse-checkout...")

        # Update sparse-checkout patterns
        sparse_checkout_file = local_dir / ".git" / "info" / "sparse-checkout"

        # Read existing patterns
        existing_patterns = set()
        if sparse_checkout_file.exists():
            with open(sparse_checkout_file, "r") as f:
                existing_patterns = set(line.strip() for line in f if line.strip())

        # Add new patterns
        new_patterns = existing_patterns.copy()
        for task_folder in task_folders:
            new_patterns.add(f"data/{task_folder}/")
            new_patterns.add(f"meta/episodes/{task_folder}/")
            new_patterns.add(f"videos/{task_folder}/")
            new_patterns.add(f"annotations/{task_folder}/")

        # Write updated patterns
        with open(sparse_checkout_file, "w") as f:
            f.write("# Essential metadata (root level only)\n")
            f.write("/*.json\n")
            f.write("/*.yaml\n")
            f.write("/*.md\n")
            f.write("\n# Meta directory files\n")
            f.write("/meta/*.jsonl\n")
            f.write("/meta/*.json\n")
            f.write("\n# Task-specific data\n")
            for pattern in sorted(new_patterns):
                if pattern not in ["*.json", "*.yaml"]:
                    f.write(f"{pattern}\n")

        # Update the repository
        print("Fetching updates...")
        subprocess.run(["git", "fetch", "origin", "main"], cwd=local_dir, check=True)
        subprocess.run(["git", "checkout", "main"], cwd=local_dir, check=True)
        subprocess.run(["git", "reset", "--hard", "origin/main"], cwd=local_dir, check=True)

        # Pull LFS files only for selected tasks
        print("Downloading LFS files for selected tasks...")
        for task_folder in task_folders:
            print(f"  Pulling LFS files for {task_folder}...")
            # Pull specific directories
            subprocess.run(
                ["git", "lfs", "pull", "--include", f"data/{task_folder}/**"],
                cwd=local_dir,
                check=True
            )
            subprocess.run(
                ["git", "lfs", "pull", "--include", f"meta/episodes/{task_folder}/**"],
                cwd=local_dir,
                check=True
            )
            subprocess.run(
                ["git", "lfs", "pull", "--include", f"videos/{task_folder}/**"],
                cwd=local_dir,
                check=True
            )
            subprocess.run(
                ["git", "lfs", "pull", "--include", f"annotations/{task_folder}/**"],
                cwd=local_dir,
                check=True
            )

        # Pull meta files
        print("  Pulling meta files...")
        subprocess.run(
            ["git", "lfs", "pull", "--include", "meta/*.jsonl,meta/*.json"],
            cwd=local_dir,
            check=True
        )

    else:
        print("\nCloning repository with sparse-checkout...")

        # Initialize repo
        subprocess.run(["git", "init"], cwd=local_dir, check=True)
        subprocess.run(["git", "remote", "add", "origin", repo_url], cwd=local_dir, check=True)

        # Enable sparse-checkout
        subprocess.run(["git", "config", "core.sparseCheckout", "true"], cwd=local_dir, check=True)

        # Set sparse-checkout patterns
        sparse_checkout_file = local_dir / ".git" / "info" / "sparse-checkout"
        sparse_checkout_file.parent.mkdir(parents=True, exist_ok=True)

        with open(sparse_checkout_file, "w") as f:
            f.write("# Essential metadata (root level only)\n")
            f.write("/*.json\n")
            f.write("/*.yaml\n")
            f.write("/*.md\n")
            f.write("\n# Meta directory files\n")
            f.write("/meta/*.jsonl\n")
            f.write("/meta/*.json\n")
            f.write("\n# Task-specific data and metadata\n")
            for task_folder in task_folders:
                f.write(f"data/{task_folder}/\n")
                f.write(f"meta/episodes/{task_folder}/\n")
                f.write(f"videos/{task_folder}/\n")
                f.write(f"annotations/{task_folder}/\n")

        # Pull only specified files
        print("Downloading specified tasks (this may take a while)...")
        subprocess.run(["git", "pull", "origin", "main"], cwd=local_dir, check=True)

        # Pull LFS files only for selected tasks
        print("Downloading LFS files for selected tasks...")
        for task_folder in task_folders:
            print(f"  Pulling LFS files for {task_folder}...")
            # Pull specific directories
            subprocess.run(
                ["git", "lfs", "pull", "--include", f"data/{task_folder}/**"],
                cwd=local_dir,
                check=True
            )
            subprocess.run(
                ["git", "lfs", "pull", "--include", f"meta/episodes/{task_folder}/**"],
                cwd=local_dir,
                check=True
            )
            subprocess.run(
                ["git", "lfs", "pull", "--include", f"videos/{task_folder}/**"],
                cwd=local_dir,
                check=True
            )
            subprocess.run(
                ["git", "lfs", "pull", "--include", f"annotations/{task_folder}/**"],
                cwd=local_dir,
                check=True
            )

        # Pull meta files
        print("  Pulling meta files...")
        subprocess.run(
            ["git", "lfs", "pull", "--include", "meta/*.jsonl,meta/*.json"],
            cwd=local_dir,
            check=True
        )

    print(f"\n[v] Download complete!")
    print(f"Local directory: {local_dir}")

    return local_dir


def main():
    parser = argparse.ArgumentParser(
        description="Download specific BEHAVIOR-1K tasks using git sparse-checkout"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["turning_on_radio", "picking_up_trash"],
        help="Task names to download",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="Local directory (default: ~/.cache/huggingface/datasets/behavior-1k/2025-challenge-demos)",
    )

    args = parser.parse_args()

    try:
        local_dir = download_specific_tasks_git(args.tasks, args.local_dir)

        # Print summary
        print("\n" + "="*60)
        print("DOWNLOAD SUMMARY")
        print("="*60)

        local_path = Path(local_dir)

        # Count parquet files
        parquet_files = list(local_path.rglob("*.parquet"))
        print(f"Tasks downloaded: {args.tasks}")
        print(f"Parquet files: {len(parquet_files)}")

        if parquet_files:
            total_size = sum(f.stat().st_size for f in parquet_files)
            print(f"Total size: {total_size / 1e9:.2f} GB")

        # Show directory structure
        print(f"\nDirectory structure:")
        for task_folder in sorted([d.name for d in (local_path / "data").iterdir() if d.is_dir()]):
            task_files = list((local_path / "data" / task_folder).glob("*.parquet"))
            print(f"  data/{task_folder}: {len(task_files)} files")

    except subprocess.CalledProcessError as e:
        print(f"\n [x] Git error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure git is installed: git --version")
        print("2. Check Hugging Face authentication if needed")
        sys.exit(1)
    except Exception as e:
        print(f"\n [x] Error: {e}")
        raise


if __name__ == "__main__":
    main()
