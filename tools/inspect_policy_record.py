#!/usr/bin/env python3
"""Inspect policy_records saved by PolicyRecorder.

Usage:
  python tools/inspect_policy_record.py policy_records/step_0.npy

This prints a nested summary of inputs and outputs, including array shapes and small samples.
"""
from __future__ import annotations

import sys
import numpy as np
from textwrap import shorten
from typing import Any


def load_record(path: str) -> Any:
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.shape == ():
        return arr.item()
    return arr


def unflatten_if_needed(flat: dict) -> dict:
    nested: dict = {}
    for key, val in flat.items():
        if isinstance(key, tuple):
            parts = list(key)
        elif isinstance(key, str) and "/" in key:
            parts = key.split("/")
        else:
            nested[key] = val
            continue

        d = nested
        for p in parts[:-1]:
            if p not in d or not isinstance(d[p], dict):
                d[p] = {}
            d = d[p]
        d[parts[-1]] = val
    return nested


def sample_repr(x: Any, max_len: int = 240) -> str:
    try:
        import numpy as _np

        if isinstance(x, _np.ndarray):
            s = f"ndarray shape={x.shape} dtype={x.dtype}"
            flat = x.ravel()
            if flat.size > 0:
                sample = flat[:10].tolist()
                s += f" sample={sample}"
            return s
    except Exception:
        pass

    try:
        if hasattr(x, "shape"):
            try:
                return f"{type(x).__name__} shape={x.shape}"
            except Exception:
                pass
        return shorten(repr(x), width=max_len)
    except Exception:
        return "<unrepresentable>"


def print_summary(nested: dict) -> None:
    for top in ("inputs", "outputs"):
        if top not in nested:
            continue
        print(f"--- {top} ---")
        section = nested[top]
        if isinstance(section, dict):
            for k, v in section.items():
                print(f"{k}: {sample_repr(v)}")
        else:
            print(sample_repr(section))
        print()


def main(path: str) -> None:
    data = load_record(path)
    if not isinstance(data, dict):
        print("Loaded data is not a dict, type:", type(data))
        print("Raw loaded object:", data)
        return
    nested = unflatten_if_needed(data)
    print("Top-level keys:", list(nested.keys()))
    print_summary(nested)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: inspect_policy_record.py <path-to-step_*.npy>")
        sys.exit(1)
    main(sys.argv[1])
