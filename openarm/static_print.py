#!/usr/bin/env python3
"""Summarize static EDR/cosine/final-loss statistics from metadata outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print per-layer static EDR/cosine stats.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory produced by static_inference.py (contains metadata.json).",
    )
    parser.add_argument(
        "--num-demos",
        type=int,
        default=50,
        help="Number of demos to include (uses the first N entries, default: 50).",
    )
    return parser.parse_args()


def build_metadata_from_dirs(root: Path) -> list[dict]:
    entries: list[dict] = []
    subdirs = sorted([path for path in root.iterdir() if path.is_dir()])
    if not subdirs:
        artifacts: Dict[str, str] = {}
        for path in sorted(root.glob("static_*.npy")):
            artifacts[path.stem] = path.relative_to(root).as_posix()
        if artifacts:
            entries.append({"trajectory_rel_dir": ".", "artifacts": artifacts})
        return entries

    for traj_dir in subdirs:
        artifacts: Dict[str, str] = {}
        candidate_dir = traj_dir / "npy-metadata"
        search_dirs = [candidate_dir] if candidate_dir.is_dir() else [traj_dir]
        for search_dir in search_dirs:
            for path in sorted(search_dir.glob("static_*.npy")):
                artifacts[path.stem] = path.relative_to(root).as_posix()
        if artifacts:
            entries.append({"trajectory_rel_dir": traj_dir.name, "artifacts": artifacts})
    return entries


def load_metadata(root: Path) -> list[dict]:
    meta_path = root / "metadata.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return build_metadata_from_dirs(root)


def select_metadata(metadata: list[dict], num_demos: int) -> list[dict]:
    if num_demos < 0:
        raise ValueError("num_demos must be >= 0")
    return metadata[: min(num_demos, len(metadata))]


def available_layers(metadata: list[dict]) -> Iterable[int]:
    layers: set[int] = set()
    for entry in metadata:
        for key in entry.get("artifacts", {}):
            if key.startswith("static_edr_layer_"):
                suffix = key.split("_")[-1]
                layers.add(int(suffix))
    return sorted(layers)


def collect_values(root: Path, metadata: list[dict], name_pattern: str) -> np.ndarray | None:
    arrays: list[np.ndarray] = []
    for entry in metadata:
        rel_path = entry.get("artifacts", {}).get(name_pattern)
        if not rel_path:
            continue
        path = root / rel_path
        if not path.exists():
            continue
        arrays.append(np.load(path).astype(np.float64, copy=False).reshape(-1))
    if not arrays:
        return None
    return np.concatenate(arrays, axis=0)


def summarize_layer(root: Path, metadata: list[dict], layer_idx: int) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    edr = collect_values(root, metadata, f"static_edr_layer_{layer_idx:02d}")
    cosine = collect_values(root, metadata, f"static_cosine_layer_{layer_idx:02d}")
    if edr is not None:
        stats["edr"] = {"mean": float(np.nanmean(edr)), "var": float(np.nanvar(edr)), "std": float(np.nanstd(edr))}
    if cosine is not None:
        stats["cosine"] = {
            "mean": float(np.nanmean(cosine)),
            "var": float(np.nanvar(cosine)),
            "std": float(np.nanstd(cosine)),
        }
    return stats


def summarize_final_loss(root: Path, metadata: list[dict]) -> Dict[str, float] | None:
    values = collect_values(root, metadata, "static_final_loss")
    if values is None:
        return None
    return {"mean": float(np.nanmean(values)), "var": float(np.nanvar(values)), "std": float(np.nanstd(values))}


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    metadata = select_metadata(load_metadata(data_dir), args.num_demos)

    layers = available_layers(metadata)
    if not layers:
        print("No layer artifacts found.")
        return

    print("Per-layer statistics:")
    for layer_idx in layers:
        stats = summarize_layer(data_dir, metadata, layer_idx)
        edr = stats.get("edr")
        cosine = stats.get("cosine")
        edr_str = (
            f"EDR mean={edr['mean']:.6f} var={edr['var']:.6f} std={edr['std']:.6f}"
            if edr
            else "EDR N/A"
        )
        cosine_str = (
            f"Cosine mean={cosine['mean']:.6f} var={cosine['var']:.6f} std={cosine['std']:.6f}"
            if cosine
            else "Cosine N/A"
        )
        print(f"  Layer {layer_idx:02d}: {edr_str} | {cosine_str}")

    loss_stats = summarize_final_loss(data_dir, metadata)
    if loss_stats:
        print(
            f"\nFinal layer loss: mean={loss_stats['mean']:.6f} "
            f"var={loss_stats['var']:.6f} std={loss_stats['std']:.6f}"
        )
    else:
        print("\nFinal layer loss: N/A")


if __name__ == "__main__":
    main()
