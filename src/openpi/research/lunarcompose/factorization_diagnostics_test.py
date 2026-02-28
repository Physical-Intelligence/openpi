"""Tests for factorization_diagnostics module."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.research.lunarcompose import factorization_diagnostics


# ---------------------------------------------------------------------------
# Synthetic MissingCornerResult (lightweight stand-in)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _SyntheticMissingCornerResult:
    """Minimal stand-in for MissingCornerResult used by diagnostics."""

    seen_mean: float
    unseen_mean: float
    per_cell_success: dict[tuple[str, str], float]
    per_task_breakdown: dict[str, float]
    per_env_breakdown: dict[str, float]


# ---------------------------------------------------------------------------
# Import smoke test (PRESERVED from original)
# ---------------------------------------------------------------------------


def test_factorization_diagnostics_module_imports():
    """Verify the factorization_diagnostics module can be imported."""
    from openpi.research.lunarcompose import factorization_diagnostics  # noqa: F401


# ---------------------------------------------------------------------------
# 1. seen_unseen_gap
# ---------------------------------------------------------------------------


def test_seen_unseen_gap_known_values():
    """Gap should equal seen_mean - unseen_mean exactly."""
    result = _SyntheticMissingCornerResult(
        seen_mean=0.8,
        unseen_mean=0.4,
        per_cell_success={},
        per_task_breakdown={},
        per_env_breakdown={},
    )
    gap = factorization_diagnostics.seen_unseen_gap(result)
    assert gap == pytest.approx(0.4)


def test_seen_unseen_gap_negative():
    """Gap can be negative if unseen outperforms seen."""
    result = _SyntheticMissingCornerResult(
        seen_mean=0.3,
        unseen_mean=0.6,
        per_cell_success={},
        per_task_breakdown={},
        per_env_breakdown={},
    )
    assert factorization_diagnostics.seen_unseen_gap(result) == pytest.approx(-0.3)


# ---------------------------------------------------------------------------
# 2. cross_condition_breakdown
# ---------------------------------------------------------------------------


def test_cross_condition_breakdown_shape():
    """Returned dict must have 'per_task' and 'per_env' with correct sub-keys."""
    result = _SyntheticMissingCornerResult(
        seen_mean=0.7,
        unseen_mean=0.5,
        per_cell_success={},
        per_task_breakdown={"grasp": 0.9, "place": 0.6, "pour": 0.4},
        per_env_breakdown={"lab": 0.8, "field": 0.5},
    )
    breakdown = factorization_diagnostics.cross_condition_breakdown(result)

    assert set(breakdown.keys()) == {"per_task", "per_env"}
    assert set(breakdown["per_task"].keys()) == {"grasp", "place", "pour"}
    assert set(breakdown["per_env"].keys()) == {"lab", "field"}
    assert breakdown["per_task"]["grasp"] == pytest.approx(0.9)
    assert breakdown["per_env"]["field"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 3. routing_interaction_analysis
# ---------------------------------------------------------------------------


def test_routing_interaction_scalar():
    """MI should be a single non-negative float for random softmax distributions."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    task_probs = jax.nn.softmax(jax.random.normal(k1, (10, 4)), axis=-1)
    env_probs = jax.nn.softmax(jax.random.normal(k2, (10, 3)), axis=-1)

    mi = factorization_diagnostics.routing_interaction_analysis(task_probs, env_probs)

    assert isinstance(mi, float)
    assert mi >= 0.0


def test_routing_interaction_independent():
    """MI should be near zero for perfectly independent uniform distributions."""
    # Uniform distributions → independent routing → MI ≈ 0
    task_probs = jnp.ones((50, 4)) / 4.0
    env_probs = jnp.ones((50, 3)) / 3.0

    mi = factorization_diagnostics.routing_interaction_analysis(task_probs, env_probs)

    assert mi == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# 4. task_env_entanglement
# ---------------------------------------------------------------------------


def test_entanglement_zero_for_orthogonal():
    """Orthogonal parameter vectors should give near-zero cosine similarity."""
    # Construct two orthogonal vectors
    v1 = np.array([1.0, 0.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0, 0.0])

    task_params = {"layer1": {"weight": v1}}
    env_params = {"layer1": {"weight": v2}}

    entanglement = factorization_diagnostics.task_env_entanglement(task_params, env_params)
    assert abs(entanglement) < 0.1


def test_entanglement_high_for_identical():
    """Identical parameter vectors should give cosine similarity near 1.0."""
    rng = np.random.RandomState(123)
    v = rng.randn(64)

    task_params = {"block": {"w": v, "b": rng.randn(8)}}
    env_params = {"block": {"w": v.copy(), "b": rng.randn(8).copy()}}

    # Make env_params identical
    env_params["block"]["b"] = task_params["block"]["b"].copy()

    entanglement = factorization_diagnostics.task_env_entanglement(task_params, env_params)
    assert entanglement > 0.9


def test_entanglement_different_sizes():
    """Entanglement should handle parameter dicts of different total sizes."""
    task_params = {"a": np.ones(10)}
    env_params = {"a": np.ones(5)}  # shorter

    # Should not raise — shorter is zero-padded
    entanglement = factorization_diagnostics.task_env_entanglement(task_params, env_params)
    assert -1.0 <= entanglement <= 1.0


# ---------------------------------------------------------------------------
# 5. counterfactual_swap_test
# ---------------------------------------------------------------------------


def test_counterfactual_swap_structure():
    """Returned dict must have the four expected keys with correct semantics."""
    original = {
        ("grasp", "lab"): 0.9,
        ("grasp", "field"): 0.7,
        ("place", "lab"): 0.8,
        ("place", "field"): 0.6,
    }
    swapped = {
        ("grasp", "lab"): 0.85,
        ("grasp", "field"): 0.65,
        ("place", "lab"): 0.75,
        ("place", "field"): 0.55,
    }

    result = factorization_diagnostics.counterfactual_swap_test(original, swapped)

    assert set(result.keys()) == {"mean_delta", "max_delta", "min_delta", "num_cells"}
    assert result["num_cells"] == pytest.approx(4.0)
    # All deltas are -0.05, so mean = -0.05, max(|d|) = 0.05, min(|d|) = 0.05
    assert result["mean_delta"] == pytest.approx(-0.05)
    assert result["max_delta"] == pytest.approx(0.05)
    assert result["min_delta"] == pytest.approx(0.05)


def test_counterfactual_swap_empty():
    """Empty input should return zero deltas and num_cells=0."""
    result = factorization_diagnostics.counterfactual_swap_test({}, {})
    assert result["num_cells"] == pytest.approx(0.0)
    assert result["mean_delta"] == pytest.approx(0.0)


def test_counterfactual_swap_partial_overlap():
    """Only common keys should be compared."""
    original = {("a", "x"): 0.5, ("b", "y"): 0.6}
    swapped = {("a", "x"): 0.7, ("c", "z"): 0.9}

    result = factorization_diagnostics.counterfactual_swap_test(original, swapped)
    assert result["num_cells"] == pytest.approx(1.0)
    assert result["mean_delta"] == pytest.approx(0.2)
