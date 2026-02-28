"""Tests for env_adapter_bank module."""

from __future__ import annotations

from pathlib import Path

import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.research.lunarcompose.env_adapter_bank import EnvAdapterBank
from openpi.research.spacecil.task_adapter_bank import TaskAdapterBank

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class DummySigLIPModel(nnx.Module):
    """Model with siglip-named LoRA params for env adapter testing.

    PathRegex(".*siglip.*") matches `siglip_lora_a` and `siglip_lora_b`.
    PathRegex(".*lora.*") matches `siglip_lora_a`, `siglip_lora_b`, AND `lora_c`, `lora_d`.
    `backbone_w` matches neither filter.
    """

    def __init__(self, rngs: nnx.Rngs) -> None:
        self.backbone_w = nnx.Param(jnp.ones((4, 4)))
        # SigLIP-path params (matched by env adapter's .*siglip.* filter)
        self.siglip_lora_a = nnx.Param(jnp.zeros((4, 2)))
        self.siglip_lora_b = nnx.Param(jnp.zeros((2, 4)))
        # Action/LM-path params (matched by task adapter's .*lora.* filter, but NOT .*siglip.*)
        self.lora_c = nnx.Param(jnp.zeros((4, 2)))
        self.lora_d = nnx.Param(jnp.zeros((2, 4)))


def _make_siglip_model(seed: int = 0) -> DummySigLIPModel:
    return DummySigLIPModel(rngs=nnx.Rngs(seed))


def _make_siglip_model_with_values(
    siglip_a: float = 0.0,
    siglip_b: float = 0.0,
    lora_c: float = 0.0,
    lora_d: float = 0.0,
) -> DummySigLIPModel:
    """Create a model with specific param values for deterministic testing."""
    model = _make_siglip_model()
    model.siglip_lora_a = nnx.Param(jnp.full((4, 2), siglip_a))
    model.siglip_lora_b = nnx.Param(jnp.full((2, 4), siglip_b))
    model.lora_c = nnx.Param(jnp.full((4, 2), lora_c))
    model.lora_d = nnx.Param(jnp.full((2, 4), lora_d))
    return model


# ---------------------------------------------------------------------------
# Existing import smoke test (preserved)
# ---------------------------------------------------------------------------


def test_env_adapter_bank_module_imports():
    """Verify the env_adapter_bank module can be imported."""
    from openpi.research.lunarcompose import env_adapter_bank  # noqa: F401


# ---------------------------------------------------------------------------
# Test a) Register and retrieve
# ---------------------------------------------------------------------------


def test_register_and_retrieve():
    """Register env adapter from model, retrieve, verify non-empty numpy arrays."""
    model = _make_siglip_model_with_values(siglip_a=3.0, siglip_b=5.0)
    bank = EnvAdapterBank()
    bank.register_env("nominal", model)

    pure = bank.get_env("nominal")
    assert len(pure) > 0, "Retrieved dict should be non-empty"

    # Verify values are numpy arrays with expected values
    assert "siglip_lora_a" in pure
    assert "siglip_lora_b" in pure
    assert isinstance(pure["siglip_lora_a"], np.ndarray)
    np.testing.assert_allclose(pure["siglip_lora_a"], 3.0)
    np.testing.assert_allclose(pure["siglip_lora_b"], 5.0)


# ---------------------------------------------------------------------------
# Test b) Merge into model
# ---------------------------------------------------------------------------


def test_merge_into_model():
    """merge_into_model injects stored env weights into a live model."""
    bank = EnvAdapterBank()
    source = _make_siglip_model_with_values(siglip_a=7.0, siglip_b=8.0)
    bank.register_env("nominal", source)

    # Target model starts with zeros for siglip params
    target = _make_siglip_model_with_values(siglip_a=0.0, siglip_b=0.0)
    np.testing.assert_allclose(np.asarray(target.siglip_lora_a.value), 0.0)

    bank.merge_into_model(target, "nominal")

    # Verify siglip params changed
    np.testing.assert_allclose(np.asarray(target.siglip_lora_a.value), 7.0)
    np.testing.assert_allclose(np.asarray(target.siglip_lora_b.value), 8.0)
    # Backbone should be untouched
    np.testing.assert_allclose(np.asarray(target.backbone_w.value), 1.0)


# ---------------------------------------------------------------------------
# Test c) Composition with task adapter
# ---------------------------------------------------------------------------


def test_composition_with_task_adapter():
    """Apply both task adapter (lora) and env adapter (siglip) to same model.

    Demonstrates the composition rule:
        effective = base + task_lora_delta + env_lora_delta
    """
    model = _make_siglip_model_with_values(siglip_a=0.0, siglip_b=0.0, lora_c=0.0, lora_d=0.0)

    # --- Task adapter bank (targets .*lora.* — matches all lora-named params) ---
    task_bank = TaskAdapterBank()
    task_source = _make_siglip_model_with_values(siglip_a=1.0, siglip_b=1.0, lora_c=10.0, lora_d=20.0)
    task_bank.register_adapter("pick", task_source)

    # --- Env adapter bank (targets .*siglip.* — matches only siglip-named params) ---
    env_bank = EnvAdapterBank()
    env_source = _make_siglip_model_with_values(siglip_a=100.0, siglip_b=200.0, lora_c=0.0, lora_d=0.0)
    env_bank.register_env("dark", env_source)

    # Apply task adapter first — updates ALL lora-named params
    task_bank.merge_into_model(model, "pick")
    np.testing.assert_allclose(np.asarray(model.lora_c.value), 10.0)
    np.testing.assert_allclose(np.asarray(model.lora_d.value), 20.0)
    # Task adapter also touched siglip params (they match .*lora.*)
    np.testing.assert_allclose(np.asarray(model.siglip_lora_a.value), 1.0)

    # Apply env adapter — overwrites only siglip-named params
    env_bank.merge_into_model(model, "dark")
    np.testing.assert_allclose(np.asarray(model.siglip_lora_a.value), 100.0)
    np.testing.assert_allclose(np.asarray(model.siglip_lora_b.value), 200.0)
    # Task adapter's non-siglip lora params should be unchanged
    np.testing.assert_allclose(np.asarray(model.lora_c.value), 10.0)
    np.testing.assert_allclose(np.asarray(model.lora_d.value), 20.0)


# ---------------------------------------------------------------------------
# Test d) Checkpoint roundtrip
# ---------------------------------------------------------------------------


def test_checkpoint_roundtrip(tmp_path: Path):
    """Save 2 env adapters, load back, verify states match exactly."""
    bank = EnvAdapterBank()
    bank.register_env("nominal", _make_siglip_model_with_values(siglip_a=1.0, siglip_b=2.0))
    bank.register_env("dark", _make_siglip_model_with_values(siglip_a=3.0, siglip_b=4.0))
    bank.freeze_env("nominal")

    save_path = tmp_path / "env_bank"
    bank.save(save_path)

    loaded = EnvAdapterBank.load(save_path)

    # Metadata
    assert loaded.num_adapters == 2
    assert loaded.registered_envs == ["nominal", "dark"]
    assert loaded.is_frozen("nominal")
    assert not loaded.is_frozen("dark")
    assert loaded.lora_target == ".*siglip.*"
    assert not loaded.fallback_prefix_mode

    # Values
    pure_nom = loaded.get_env("nominal")
    np.testing.assert_array_equal(pure_nom["siglip_lora_a"], np.full((4, 2), 1.0))
    np.testing.assert_array_equal(pure_nom["siglip_lora_b"], np.full((2, 4), 2.0))

    pure_dark = loaded.get_env("dark")
    np.testing.assert_array_equal(pure_dark["siglip_lora_a"], np.full((4, 2), 3.0))
    np.testing.assert_array_equal(pure_dark["siglip_lora_b"], np.full((2, 4), 4.0))


# ---------------------------------------------------------------------------
# Test e) Visual path targeting
# ---------------------------------------------------------------------------


def test_visual_path_targeting():
    """Env adapter only captures siglip-matching params, not other lora params."""
    model = _make_siglip_model_with_values(siglip_a=5.0, siglip_b=6.0, lora_c=99.0, lora_d=99.0)

    bank = EnvAdapterBank()  # default filter: .*siglip.*
    bank.register_env("test_env", model)

    pure = bank.get_env("test_env")

    # Should contain siglip params
    assert "siglip_lora_a" in pure
    assert "siglip_lora_b" in pure
    np.testing.assert_allclose(pure["siglip_lora_a"], 5.0)
    np.testing.assert_allclose(pure["siglip_lora_b"], 6.0)

    # Should NOT contain non-siglip lora params or backbone
    assert "lora_c" not in pure
    assert "lora_d" not in pure
    assert "backbone_w" not in pure


# ---------------------------------------------------------------------------
# Test f) Fallback prefix mode
# ---------------------------------------------------------------------------


def test_fallback_prefix_mode():
    """Prefix fallback mode stores and retrieves env conditioning strings."""
    bank = EnvAdapterBank(fallback_prefix_mode=True)

    bank.register_env("nominal", prefix="lunar nominal lighting")
    bank.register_env("dark", prefix="lunar dark shadow environment")

    assert bank.get_prefix("nominal") == "lunar nominal lighting"
    assert bank.get_prefix("dark") == "lunar dark shadow environment"
    assert bank.num_adapters == 2
    assert bank.registered_envs == ["nominal", "dark"]


def test_fallback_prefix_mode_roundtrip(tmp_path: Path):
    """Prefix mode survives save/load cycle."""
    bank = EnvAdapterBank(fallback_prefix_mode=True)
    bank.register_env("nominal", prefix="lunar nominal lighting")
    bank.register_env("dark", prefix="lunar dark shadow")
    bank.freeze_env("nominal")

    save_path = tmp_path / "prefix_bank"
    bank.save(save_path)

    loaded = EnvAdapterBank.load(save_path)

    assert loaded.fallback_prefix_mode is True
    assert loaded.get_prefix("nominal") == "lunar nominal lighting"
    assert loaded.get_prefix("dark") == "lunar dark shadow"
    assert loaded.is_frozen("nominal")
    assert not loaded.is_frozen("dark")


# ---------------------------------------------------------------------------
# Additional edge case tests
# ---------------------------------------------------------------------------


def test_freeze_rejects_reregistration():
    """Frozen env adapter cannot be re-registered."""
    bank = EnvAdapterBank()
    bank.register_env("nominal", _make_siglip_model())
    bank.freeze_env("nominal")

    assert bank.is_frozen("nominal")

    with pytest.raises(ValueError, match="frozen"):
        bank.register_env("nominal", _make_siglip_model())


def test_get_env_nonexistent_raises_keyerror():
    """get_env on a non-existent env raises KeyError."""
    bank = EnvAdapterBank()
    with pytest.raises(KeyError, match="No adapter registered"):
        bank.get_env("nonexistent")


def test_list_envs():
    """list_envs returns registration order."""
    bank = EnvAdapterBank()
    for name in ["alpha", "beta", "gamma"]:
        bank.register_env(name, _make_siglip_model())
    assert bank.list_envs() == ["alpha", "beta", "gamma"]


def test_repr():
    """Verify __repr__ includes env list, mode, and target."""
    bank = EnvAdapterBank()
    bank.register_env("nominal", _make_siglip_model())
    bank.register_env("dark", _make_siglip_model())
    bank.freeze_env("nominal")

    r = repr(bank)
    assert "nominal" in r
    assert "dark" in r
    assert "lora" in r
    assert "siglip" in r
    assert "frozen" in r
