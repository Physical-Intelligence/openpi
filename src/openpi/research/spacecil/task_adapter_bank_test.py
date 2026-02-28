"""Tests for task_adapter_bank module."""

from __future__ import annotations

from pathlib import Path
import tempfile

import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.research.spacecil.task_adapter_bank import LORA_FILTER
from openpi.research.spacecil.task_adapter_bank import TaskAdapterBank
from openpi.research.spacecil.task_adapter_bank import _flatten_dict
from openpi.research.spacecil.task_adapter_bank import _unflatten_dict

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class DummyLoRAModel(nnx.Module):
    """Minimal model with LoRA-like params for testing.

    PathRegex(".*lora.*") will match `lora_a` and `lora_b` but NOT `backbone_w`.
    """

    def __init__(self, rngs: nnx.Rngs) -> None:
        self.backbone_w = nnx.Param(jnp.ones((4, 4)))
        self.lora_a = nnx.Param(jnp.zeros((4, 2)))
        self.lora_b = nnx.Param(jnp.zeros((2, 4)))


def _make_model(seed: int = 0) -> DummyLoRAModel:
    return DummyLoRAModel(rngs=nnx.Rngs(seed))


def _make_model_with_values(lora_a_val: float, lora_b_val: float) -> DummyLoRAModel:
    """Create a model with specific LoRA values for deterministic testing."""
    model = _make_model()
    model.lora_a = nnx.Param(jnp.full((4, 2), lora_a_val))
    model.lora_b = nnx.Param(jnp.full((2, 4), lora_b_val))
    return model


# ---------------------------------------------------------------------------
# Basic module import
# ---------------------------------------------------------------------------


def test_task_adapter_bank_module_imports():
    """Verify the task_adapter_bank module can be imported."""
    from openpi.research.spacecil import task_adapter_bank  # noqa: F401


# ---------------------------------------------------------------------------
# Registration and retrieval
# ---------------------------------------------------------------------------


def test_register_and_get_adapter():
    """Register adapter from model, retrieve, verify values match."""
    model = _make_model_with_values(1.0, 2.0)
    bank = TaskAdapterBank()
    bank.register_adapter("task_0", model)

    pure = bank.get_adapter("task_0")
    # The pure dict should contain lora_a and lora_b at top level
    assert "lora_a" in pure
    assert "lora_b" in pure
    np.testing.assert_allclose(np.asarray(pure["lora_a"]), 1.0)
    np.testing.assert_allclose(np.asarray(pure["lora_b"]), 2.0)


def test_register_two_adapters():
    """Register two adapters, verify both are retrievable with correct values."""
    bank = TaskAdapterBank()

    model_a = _make_model_with_values(1.0, 2.0)
    bank.register_adapter("task_a", model_a)

    model_b = _make_model_with_values(3.0, 4.0)
    bank.register_adapter("task_b", model_b)

    pure_a = bank.get_adapter("task_a")
    pure_b = bank.get_adapter("task_b")
    np.testing.assert_allclose(np.asarray(pure_a["lora_a"]), 1.0)
    np.testing.assert_allclose(np.asarray(pure_b["lora_a"]), 3.0)


def test_register_adapter_from_state():
    """Register from an nnx.State object rather than a model."""
    model = _make_model_with_values(5.0, 6.0)
    lora_state = nnx.state(model).filter(LORA_FILTER)

    bank = TaskAdapterBank()
    bank.register_adapter_from_state("task_s", lora_state)

    pure = bank.get_adapter("task_s")
    np.testing.assert_allclose(np.asarray(pure["lora_a"]), 5.0)
    np.testing.assert_allclose(np.asarray(pure["lora_b"]), 6.0)


def test_overwrite_unfrozen_adapter():
    """Re-registering an unfrozen adapter overwrites it."""
    bank = TaskAdapterBank()
    bank.register_adapter("task_0", _make_model_with_values(1.0, 1.0))
    bank.register_adapter("task_0", _make_model_with_values(9.0, 9.0))

    pure = bank.get_adapter("task_0")
    np.testing.assert_allclose(np.asarray(pure["lora_a"]), 9.0)


# ---------------------------------------------------------------------------
# Freezing
# ---------------------------------------------------------------------------


def test_freeze_and_reject_reregister():
    """Frozen adapter cannot be re-registered."""
    bank = TaskAdapterBank()
    bank.register_adapter("task_0", _make_model_with_values(1.0, 1.0))
    bank.freeze_adapter("task_0")

    assert bank.is_frozen("task_0")

    with pytest.raises(ValueError, match="frozen"):
        bank.register_adapter("task_0", _make_model_with_values(2.0, 2.0))


def test_freeze_rejects_from_state_too():
    """Frozen adapter also rejects register_adapter_from_state."""
    bank = TaskAdapterBank()
    model = _make_model_with_values(1.0, 1.0)
    bank.register_adapter("task_0", model)
    bank.freeze_adapter("task_0")

    lora_state = nnx.state(model).filter(LORA_FILTER)
    with pytest.raises(ValueError, match="frozen"):
        bank.register_adapter_from_state("task_0", lora_state)


def test_freeze_nonexistent_raises_keyerror():
    """Freezing a non-existent task raises KeyError."""
    bank = TaskAdapterBank()
    with pytest.raises(KeyError, match="No adapter registered"):
        bank.freeze_adapter("nonexistent")


def test_is_frozen_returns_false_for_unknown():
    """is_frozen returns False for tasks that don't exist."""
    bank = TaskAdapterBank()
    assert not bank.is_frozen("unknown_task")


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_num_adapters():
    """num_adapters reflects registration count."""
    bank = TaskAdapterBank()
    assert bank.num_adapters == 0

    bank.register_adapter("t0", _make_model())
    assert bank.num_adapters == 1

    bank.register_adapter("t1", _make_model())
    assert bank.num_adapters == 2

    # Re-registration doesn't increase count
    bank.register_adapter("t0", _make_model())
    assert bank.num_adapters == 2


def test_registered_tasks_order():
    """registered_tasks preserves insertion order."""
    bank = TaskAdapterBank()
    for name in ["alpha", "beta", "gamma"]:
        bank.register_adapter(name, _make_model())

    assert bank.registered_tasks == ["alpha", "beta", "gamma"]


# ---------------------------------------------------------------------------
# KeyError on missing adapter
# ---------------------------------------------------------------------------


def test_get_adapter_nonexistent_raises_keyerror():
    """get_adapter on a non-existent task raises KeyError."""
    bank = TaskAdapterBank()
    with pytest.raises(KeyError, match="No adapter registered"):
        bank.get_adapter("nonexistent")


# ---------------------------------------------------------------------------
# merge_into_model
# ---------------------------------------------------------------------------


def test_merge_into_model_changes_params():
    """merge_into_model injects stored LoRA weights into the live model."""
    # Register adapter with specific values
    bank = TaskAdapterBank()
    source = _make_model_with_values(7.0, 8.0)
    bank.register_adapter("task_src", source)

    # Target model starts with zeros
    target = _make_model_with_values(0.0, 0.0)
    np.testing.assert_allclose(np.asarray(target.lora_a.value), 0.0)

    # Merge adapter into target
    bank.merge_into_model(target, "task_src")

    # Verify LoRA params changed
    np.testing.assert_allclose(np.asarray(target.lora_a.value), 7.0)
    np.testing.assert_allclose(np.asarray(target.lora_b.value), 8.0)
    # Backbone should be untouched
    np.testing.assert_allclose(np.asarray(target.backbone_w.value), 1.0)


def test_merge_into_model_nonexistent_raises_keyerror():
    """merge_into_model with non-existent task raises KeyError."""
    bank = TaskAdapterBank()
    model = _make_model()
    with pytest.raises(KeyError, match="No adapter registered"):
        bank.merge_into_model(model, "missing")


# ---------------------------------------------------------------------------
# Save / Load round-trip
# ---------------------------------------------------------------------------


def test_save_load_roundtrip():
    """Save bank to disk, load it back, verify all adapters are equal."""
    bank = TaskAdapterBank()
    bank.register_adapter("task_a", _make_model_with_values(1.0, 2.0))
    bank.register_adapter("task_b", _make_model_with_values(3.0, 4.0))
    bank.freeze_adapter("task_a")

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "bank"
        bank.save(save_path)

        loaded = TaskAdapterBank.load(save_path)

    # Check metadata
    assert loaded.num_adapters == 2
    assert loaded.registered_tasks == ["task_a", "task_b"]
    assert loaded.is_frozen("task_a")
    assert not loaded.is_frozen("task_b")

    # Check adapter values
    pure_a = loaded.get_adapter("task_a")
    np.testing.assert_allclose(np.asarray(pure_a["lora_a"]), 1.0)
    np.testing.assert_allclose(np.asarray(pure_a["lora_b"]), 2.0)

    pure_b = loaded.get_adapter("task_b")
    np.testing.assert_allclose(np.asarray(pure_b["lora_a"]), 3.0)
    np.testing.assert_allclose(np.asarray(pure_b["lora_b"]), 4.0)


def test_registration_order_preserved_after_roundtrip():
    """Registration order survives save/load cycle."""
    bank = TaskAdapterBank()
    order = ["zulu", "alpha", "mike", "bravo"]
    for name in order:
        bank.register_adapter(name, _make_model())

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "bank"
        bank.save(save_path)
        loaded = TaskAdapterBank.load(save_path)

    assert loaded.registered_tasks == order


def test_loaded_bank_merge_into_model():
    """Adapters loaded from disk can be merged into a model."""
    bank = TaskAdapterBank()
    bank.register_adapter("task_x", _make_model_with_values(11.0, 22.0))

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "bank"
        bank.save(save_path)
        loaded = TaskAdapterBank.load(save_path)

    target = _make_model_with_values(0.0, 0.0)
    loaded.merge_into_model(target, "task_x")
    np.testing.assert_allclose(np.asarray(target.lora_a.value), 11.0)
    np.testing.assert_allclose(np.asarray(target.lora_b.value), 22.0)


def test_save_creates_expected_files():
    """Verify the directory structure created by save()."""
    bank = TaskAdapterBank()
    bank.register_adapter("t0", _make_model())

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "bank"
        bank.save(save_path)

        assert (save_path / "metadata.json").is_file()
        assert (save_path / "t0" / "adapter.npz").is_file()


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------


def test_flatten_unflatten_roundtrip():
    """_flatten_dict and _unflatten_dict are inverses."""
    nested = {
        "a": {"b": np.array([1, 2]), "c": np.array([3])},
        "d": np.array([4, 5, 6]),
    }
    flat = _flatten_dict(nested)
    assert set(flat.keys()) == {"a/b", "a/c", "d"}

    restored = _unflatten_dict(flat)
    np.testing.assert_array_equal(restored["a"]["b"], nested["a"]["b"])
    np.testing.assert_array_equal(restored["a"]["c"], nested["a"]["c"])
    np.testing.assert_array_equal(restored["d"], nested["d"])


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


def test_repr():
    """Verify __repr__ includes task list and frozen info."""
    bank = TaskAdapterBank()
    bank.register_adapter("t0", _make_model())
    bank.register_adapter("t1", _make_model())
    bank.freeze_adapter("t0")

    r = repr(bank)
    assert "t0" in r
    assert "t1" in r
    assert "frozen" in r
