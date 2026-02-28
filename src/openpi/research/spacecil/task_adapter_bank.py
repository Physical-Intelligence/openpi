"""Per-task PEFT module registry using openpi's LoRA infrastructure.

Each task adapter = separate LoRA parameter set (nnx.State filtered to .*lora.*).
Bank = dictionary {task_id: lora_pure_dict}.
Swap via nnx.update(model, lora_state) before forward passes — MUST be called outside JIT.
"""

from __future__ import annotations

import json
from pathlib import Path

import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np

from openpi.shared import nnx_utils

# Matches any parameter path containing "lora" (e.g. lora_a, lora_b).
LORA_FILTER = nnx_utils.PathRegex(".*lora.*")


def _pure_dict_to_numpy(d: dict) -> dict:
    """Recursively convert jax/numpy arrays in a pure dict to numpy arrays."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _pure_dict_to_numpy(v)
        else:
            # Handle jax arrays, numpy arrays, and nnx.Variable wrappers
            arr = v
            if isinstance(arr, nnx.Variable):
                arr = arr.value
            out[k] = np.asarray(arr)
    return out


def _numpy_dict_to_jax(d: dict) -> dict:
    """Recursively convert numpy arrays in a dict to jax arrays."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _numpy_dict_to_jax(v)
        else:
            out[k] = jnp.asarray(v)
    return out


def _flatten_dict(d: dict, prefix: str = "") -> dict[str, np.ndarray]:
    """Flatten a nested dict into flat keys with '/' separator for npz storage."""
    items: dict[str, np.ndarray] = {}
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, key))
        else:
            items[key] = np.asarray(v)
    return items


def _unflatten_dict(flat: dict[str, np.ndarray]) -> dict:
    """Unflatten flat '/' separated keys back into a nested dict."""
    out: dict = {}
    for key, val in flat.items():
        parts = key.split("/")
        d = out
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = val
    return out


class TaskAdapterBank:
    """Versioned registry mapping task IDs to LoRA parameter states.

    Stores adapter weights as pure dicts (via nnx.State.to_pure_dict()) to avoid
    serialization issues. Adapters can be frozen to prevent re-registration.

    Usage:
        bank = TaskAdapterBank()
        bank.register_adapter("task_0", model)  # extract LoRA from live model
        bank.freeze_adapter("task_0")            # immutable snapshot
        bank.merge_into_model(model, "task_0")   # inject LoRA weights (outside JIT!)
        bank.save("/tmp/bank")
        bank = TaskAdapterBank.load("/tmp/bank")
    """

    def __init__(self) -> None:
        self._adapters: dict[str, dict] = {}  # task_id -> pure dict
        self._frozen: set[str] = set()
        self._registration_order: list[str] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_adapter(self, task_id: str, model: nnx.Module) -> None:
        """Extract LoRA state from a live model and store it.

        Args:
            task_id: Unique identifier for this task adapter.
            model: An nnx.Module whose LoRA parameters will be snapshotted.

        Raises:
            ValueError: If task_id is already frozen.
        """
        if task_id in self._frozen:
            raise ValueError(f"Adapter '{task_id}' is frozen and cannot be re-registered.")
        model_state = nnx.state(model)
        lora_state = model_state.filter(LORA_FILTER)
        pure = lora_state.to_pure_dict()
        # Convert to numpy for consistent storage
        self._store(task_id, _pure_dict_to_numpy(pure))

    def register_adapter_from_state(self, task_id: str, lora_state: nnx.State) -> None:
        """Store adapter from an existing nnx.State.

        Args:
            task_id: Unique identifier for this task adapter.
            lora_state: An nnx.State containing LoRA parameters.

        Raises:
            ValueError: If task_id is already frozen.
        """
        if task_id in self._frozen:
            raise ValueError(f"Adapter '{task_id}' is frozen and cannot be re-registered.")
        pure = lora_state.to_pure_dict()
        self._store(task_id, _pure_dict_to_numpy(pure))

    def _store(self, task_id: str, pure_dict: dict) -> None:
        """Internal: store a pure dict and track registration order."""
        is_new = task_id not in self._adapters
        self._adapters[task_id] = pure_dict
        if is_new:
            self._registration_order.append(task_id)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_adapter(self, task_id: str) -> dict:
        """Return the stored pure dict for a task adapter.

        Raises:
            KeyError: If task_id is not registered.
        """
        if task_id not in self._adapters:
            raise KeyError(f"No adapter registered for task '{task_id}'.")
        return self._adapters[task_id]

    def merge_into_model(self, model: nnx.Module, task_id: str) -> None:
        """Apply stored LoRA weights to a live model via nnx.update.

        MUST be called outside JIT boundaries to avoid recompilation.

        Args:
            model: The live nnx.Module to update in-place.
            task_id: Which adapter's weights to inject.

        Raises:
            KeyError: If task_id is not registered.
        """
        pure = self.get_adapter(task_id)
        # Convert numpy arrays back to jax for the model
        jax_dict = _numpy_dict_to_jax(pure)
        # Build an nnx.State from the pure dict and update the model
        model_state = nnx.state(model)
        lora_state = model_state.filter(LORA_FILTER)
        lora_state.replace_by_pure_dict(jax_dict)
        nnx.update(model, lora_state)

    # ------------------------------------------------------------------
    # Freezing
    # ------------------------------------------------------------------

    def freeze_adapter(self, task_id: str) -> None:
        """Mark an adapter as immutable (prevent re-registration).

        Raises:
            KeyError: If task_id is not registered.
        """
        if task_id not in self._adapters:
            raise KeyError(f"No adapter registered for task '{task_id}'.")
        self._frozen.add(task_id)

    def is_frozen(self, task_id: str) -> bool:
        """Check whether an adapter is frozen."""
        return task_id in self._frozen

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_adapters(self) -> int:
        """Number of registered adapters."""
        return len(self._adapters)

    @property
    def registered_tasks(self) -> list[str]:
        """Task IDs in registration order."""
        return list(self._registration_order)

    # ------------------------------------------------------------------
    # Persistence (JSON metadata + npz per adapter)
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save adapter bank to disk.

        Creates directory structure:
            path/metadata.json         — frozen set + registration order
            path/{task_id}/adapter.npz — flattened LoRA arrays per task
        """
        base = Path(path)
        base.mkdir(parents=True, exist_ok=True)

        # Metadata
        meta = {
            "frozen": sorted(self._frozen),
            "registration_order": list(self._registration_order),
        }
        with open(base / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Per-adapter arrays
        for task_id, pure_dict in self._adapters.items():
            task_dir = base / task_id
            task_dir.mkdir(parents=True, exist_ok=True)
            flat = _flatten_dict(pure_dict)
            np.savez(task_dir / "adapter.npz", **flat)

    @classmethod
    def load(cls, path: str | Path) -> TaskAdapterBank:
        """Load adapter bank from disk.

        Args:
            path: Directory previously written by save().

        Returns:
            A new TaskAdapterBank with all adapters and frozen state restored.
        """
        base = Path(path)

        with open(base / "metadata.json") as f:
            meta = json.load(f)

        bank = cls()
        bank._frozen = set(meta["frozen"])
        bank._registration_order = list(meta["registration_order"])

        for task_id in bank._registration_order:
            npz_path = base / task_id / "adapter.npz"
            with np.load(str(npz_path)) as data:
                flat = {k: data[k] for k in data.files}
            bank._adapters[task_id] = _unflatten_dict(flat)

        return bank

    def __repr__(self) -> str:
        frozen_info = f", frozen={sorted(self._frozen)}" if self._frozen else ""
        return f"TaskAdapterBank(tasks={self._registration_order}{frozen_info})"
