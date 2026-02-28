"""Visual-path environment adapters for LunarCompose.

Each environment adapter = separate LoRA parameter set targeting vision encoder layers
(e.g. SigLIP). Bank = dictionary {env_id: lora_pure_dict}.
Swap via nnx.update(model, lora_state) before forward passes — MUST be called outside JIT.

Preferred path: visual-path LoRA targeting vision encoder layers.
Fallback: environment-prefix conditioning (set fallback_prefix_mode=True).

Composition rule: effective_weights = base + task_lora_delta + env_lora_delta
Both deltas applied via sequential nnx.update calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import flax.nnx as nnx
import numpy as np

from openpi.research.spacecil.task_adapter_bank import _flatten_dict
from openpi.research.spacecil.task_adapter_bank import _numpy_dict_to_jax
from openpi.research.spacecil.task_adapter_bank import _pure_dict_to_numpy
from openpi.research.spacecil.task_adapter_bank import _unflatten_dict
from openpi.shared import nnx_utils


class EnvAdapterBank:
    """Versioned registry mapping environment IDs to LoRA parameter states.

    Unlike TaskAdapterBank (which uses a fixed ``.*lora.*`` filter), EnvAdapterBank
    accepts a configurable ``lora_target`` regex that defaults to ``".*siglip.*"``
    so that environment adapters target vision encoder (SigLIP) layers specifically.

    If ``fallback_prefix_mode=True``, the bank instead stores environment-prefix
    conditioning strings rather than LoRA weight states.

    Usage (LoRA mode):
        bank = EnvAdapterBank()
        bank.register_env("nominal", model)        # extract matching params
        bank.freeze_env("nominal")                  # immutable snapshot
        bank.merge_into_model(model, "nominal")     # inject weights (outside JIT!)
        bank.save("/tmp/env_bank")
        bank = EnvAdapterBank.load("/tmp/env_bank")

    Usage (prefix fallback mode):
        bank = EnvAdapterBank(fallback_prefix_mode=True)
        bank.register_env("nominal", prefix="lunar nominal lighting")
        bank.get_prefix("nominal")  # → "lunar nominal lighting"
    """

    def __init__(
        self,
        lora_target: str = ".*siglip.*",
        fallback_prefix_mode: bool = False,
    ) -> None:
        self._lora_target = lora_target
        self._lora_filter = nnx_utils.PathRegex(self._lora_target)
        self._fallback_prefix_mode = fallback_prefix_mode

        self._adapters: dict[str, dict] = {}  # env_id -> pure dict
        self._prefixes: dict[str, str] = {}  # env_id -> prefix string (fallback mode)
        self._frozen: set[str] = set()
        self._registration_order: list[str] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def lora_target(self) -> str:
        """The regex pattern used to filter LoRA parameters."""
        return self._lora_target

    @property
    def fallback_prefix_mode(self) -> bool:
        """Whether the bank operates in prefix fallback mode."""
        return self._fallback_prefix_mode

    @property
    def num_adapters(self) -> int:
        """Number of registered environment adapters."""
        return len(self._adapters) if not self._fallback_prefix_mode else len(self._prefixes)

    @property
    def registered_envs(self) -> list[str]:
        """Environment IDs in registration order."""
        return list(self._registration_order)

    def list_envs(self) -> list[str]:
        """Return environment IDs in registration order (alias for registered_envs)."""
        return self.registered_envs

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_env(
        self,
        env_id: str,
        model: nnx.Module | None = None,
        *,
        prefix: str | None = None,
    ) -> None:
        """Register an environment adapter.

        In LoRA mode (default): extract matching params from ``model`` and store them.
        In prefix fallback mode: store the ``prefix`` string for env conditioning.

        Args:
            env_id: Unique identifier for this environment adapter.
            model: An nnx.Module whose matching parameters will be snapshotted.
                   Required in LoRA mode, ignored in prefix fallback mode.
            prefix: Environment conditioning prefix string.
                    Required in prefix fallback mode, ignored in LoRA mode.

        Raises:
            ValueError: If env_id is already frozen.
            ValueError: If model is None in LoRA mode or prefix is None in prefix mode.
        """
        if env_id in self._frozen:
            raise ValueError(f"Adapter '{env_id}' is frozen and cannot be re-registered.")

        if self._fallback_prefix_mode:
            if prefix is None:
                raise ValueError("prefix is required when fallback_prefix_mode=True.")
            self._store_prefix(env_id, prefix)
        else:
            if model is None:
                raise ValueError("model is required when fallback_prefix_mode=False.")
            model_state = nnx.state(model)
            filtered_state = model_state.filter(self._lora_filter)
            pure = filtered_state.to_pure_dict()
            self._store(env_id, _pure_dict_to_numpy(pure))

    def register_env_from_state(self, env_id: str, lora_state: nnx.State) -> None:
        """Store adapter from an existing nnx.State.

        Args:
            env_id: Unique identifier for this environment adapter.
            lora_state: An nnx.State containing matching LoRA parameters.

        Raises:
            ValueError: If env_id is already frozen.
        """
        if env_id in self._frozen:
            raise ValueError(f"Adapter '{env_id}' is frozen and cannot be re-registered.")
        pure = lora_state.to_pure_dict()
        self._store(env_id, _pure_dict_to_numpy(pure))

    def _store(self, env_id: str, pure_dict: dict) -> None:
        """Internal: store a pure dict and track registration order."""
        is_new = env_id not in self._adapters
        self._adapters[env_id] = pure_dict
        if is_new and env_id not in self._registration_order:
            self._registration_order.append(env_id)

    def _store_prefix(self, env_id: str, prefix: str) -> None:
        """Internal: store a prefix string and track registration order."""
        is_new = env_id not in self._prefixes
        self._prefixes[env_id] = prefix
        if is_new and env_id not in self._registration_order:
            self._registration_order.append(env_id)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_env(self, env_id: str) -> dict:
        """Return the stored pure dict for an environment adapter.

        Raises:
            KeyError: If env_id is not registered.
            RuntimeError: If bank is in prefix fallback mode.
        """
        if self._fallback_prefix_mode:
            raise RuntimeError("get_env() is not available in prefix fallback mode. Use get_prefix() instead.")
        if env_id not in self._adapters:
            raise KeyError(f"No adapter registered for env '{env_id}'.")
        return self._adapters[env_id]

    def get_prefix(self, env_id: str) -> str:
        """Return the stored prefix string for an environment.

        Only available when ``fallback_prefix_mode=True``.

        Raises:
            KeyError: If env_id is not registered.
            RuntimeError: If bank is not in prefix fallback mode.
        """
        if not self._fallback_prefix_mode:
            raise RuntimeError("get_prefix() is only available in prefix fallback mode.")
        if env_id not in self._prefixes:
            raise KeyError(f"No prefix registered for env '{env_id}'.")
        return self._prefixes[env_id]

    def merge_into_model(self, model: nnx.Module, env_id: str) -> None:
        """Apply stored LoRA weights to a live model via nnx.update.

        MUST be called outside JIT boundaries to avoid recompilation.

        Args:
            model: The live nnx.Module to update in-place.
            env_id: Which adapter's weights to inject.

        Raises:
            KeyError: If env_id is not registered.
            RuntimeError: If bank is in prefix fallback mode.
        """
        pure = self.get_env(env_id)
        # Convert numpy arrays back to jax for the model
        jax_dict = _numpy_dict_to_jax(pure)
        # Build an nnx.State from the pure dict and update the model
        model_state = nnx.state(model)
        filtered_state = model_state.filter(self._lora_filter)
        filtered_state.replace_by_pure_dict(jax_dict)
        nnx.update(model, filtered_state)

    # ------------------------------------------------------------------
    # Freezing
    # ------------------------------------------------------------------

    def freeze_env(self, env_id: str) -> None:
        """Mark an adapter as immutable (prevent re-registration).

        Raises:
            KeyError: If env_id is not registered.
        """
        storage = self._prefixes if self._fallback_prefix_mode else self._adapters
        if env_id not in storage:
            raise KeyError(f"No adapter registered for env '{env_id}'.")
        self._frozen.add(env_id)

    def is_frozen(self, env_id: str) -> bool:
        """Check whether an adapter is frozen."""
        return env_id in self._frozen

    # ------------------------------------------------------------------
    # Persistence (JSON metadata + npz per adapter)
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save adapter bank to disk.

        Creates directory structure:
            path/metadata.json          — config, frozen set, registration order
            path/{env_id}/adapter.npz   — flattened LoRA arrays per env (LoRA mode)
            path/{env_id}/prefix.txt    — prefix string per env (prefix mode)
        """
        base = Path(path)
        base.mkdir(parents=True, exist_ok=True)

        # Metadata
        meta: dict = {
            "lora_target": self._lora_target,
            "fallback_prefix_mode": self._fallback_prefix_mode,
            "frozen": sorted(self._frozen),
            "registration_order": list(self._registration_order),
        }

        if self._fallback_prefix_mode:
            meta["prefixes"] = dict(self._prefixes)
        else:
            # Per-adapter arrays
            for env_id, pure_dict in self._adapters.items():
                env_dir = base / env_id
                env_dir.mkdir(parents=True, exist_ok=True)
                flat = _flatten_dict(pure_dict)
                np.savez(env_dir / "adapter.npz", **flat)

        with open(base / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> EnvAdapterBank:
        """Load adapter bank from disk.

        Args:
            path: Directory previously written by save().

        Returns:
            A new EnvAdapterBank with all adapters and frozen state restored.
        """
        base = Path(path)

        with open(base / "metadata.json") as f:
            meta = json.load(f)

        bank = cls(
            lora_target=meta["lora_target"],
            fallback_prefix_mode=meta["fallback_prefix_mode"],
        )
        bank._frozen = set(meta["frozen"])
        bank._registration_order = list(meta["registration_order"])

        if bank._fallback_prefix_mode:
            bank._prefixes = dict(meta.get("prefixes", {}))
        else:
            for env_id in bank._registration_order:
                npz_path = base / env_id / "adapter.npz"
                with np.load(str(npz_path)) as data:
                    flat = {k: data[k] for k in data.files}
                bank._adapters[env_id] = _unflatten_dict(flat)

        return bank

    def __repr__(self) -> str:
        mode = "prefix" if self._fallback_prefix_mode else "lora"
        frozen_info = f", frozen={sorted(self._frozen)}" if self._frozen else ""
        return (
            f"EnvAdapterBank(envs={self._registration_order}, mode={mode}, target={self._lora_target!r}{frozen_info})"
        )
