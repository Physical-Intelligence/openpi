"""Thin, lazy-import bridge to the validated legacy hardware implementation."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
from typing import Any


def load_legacy(root: str | Path, module: str, symbol: str) -> Any:
    """Import a legacy symbol without copying its environment or OpenPI files."""
    root = Path(root).resolve()
    source = root / "src"
    if not source.is_dir():
        raise FileNotFoundError(f"legacy source directory not found: {source}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    loaded = importlib.import_module(module)
    return getattr(loaded, symbol)


class LegacyHardwareBridge:
    """Construction is explicit so importing this module never opens hardware."""

    def __init__(self, root: str | Path, factory_module: str, factory_symbol: str, config: dict):
        self._root, self._module, self._symbol, self._config = root, factory_module, factory_symbol, config
        self._delegate = None

    def connect(self) -> None:
        factory = load_legacy(self._root, self._module, self._symbol)
        self._delegate = factory(self._config)
        connect = getattr(self._delegate, "connect", None)
        if callable(connect):
            connect()

    def __getattr__(self, name: str):
        if self._delegate is None:
            raise RuntimeError("legacy hardware is not connected")
        return getattr(self._delegate, name)
