"""Task-environment factorization diagnostics for LunarCompose.

Provides five diagnostic functions that assess whether the task-environment
factorization assumption holds. Results from these diagnostics feed directly
into Paper B claims.

All functions are pure (no side effects). Functions use numpy where possible;
``routing_interaction_analysis`` uses JAX for softmax-derived probability arrays.

Interpretation guide:
    - seen_unseen_gap > 0.3  → strong evidence *against* factorization
    - seen_unseen_gap < 0.1  → consistent with successful factorization
    - routing MI ≈ 0         → task and env routing are independent (ideal)
    - entanglement ≈ 0       → adapters are orthogonal (ideal)
    - counterfactual delta ≈ 0 → swapping one factor doesn't break the other
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# 1. Seen-unseen gap
# ---------------------------------------------------------------------------


def seen_unseen_gap(results: Any) -> float:
    """Primary factorization metric: gap between seen and unseen success rates.

    Args:
        results: Object with ``seen_mean`` (float) and ``unseen_mean`` (float)
            attributes, typically a ``MissingCornerResult``.

    Returns:
        ``results.seen_mean - results.unseen_mean``.  A positive value means
        unseen (compositional) combinations are harder.

    Interpretation:
        - gap > 0.3: strong evidence *against* factorization
        - gap < 0.1: consistent with successful factorization
    """
    return float(results.seen_mean - results.unseen_mean)


# ---------------------------------------------------------------------------
# 2. Cross-condition breakdown
# ---------------------------------------------------------------------------


def cross_condition_breakdown(results: Any) -> dict[str, dict[str, float]]:
    """Decompose results by task and environment to find bottlenecks.

    Args:
        results: Object with ``per_task_breakdown`` (``dict[str, float]``) and
            ``per_env_breakdown`` (``dict[str, float]``) attributes.

    Returns:
        ``{"per_task": results.per_task_breakdown, "per_env": results.per_env_breakdown}``
    """
    return {
        "per_task": dict(results.per_task_breakdown),
        "per_env": dict(results.per_env_breakdown),
    }


# ---------------------------------------------------------------------------
# 3. Routing interaction analysis (mutual information)
# ---------------------------------------------------------------------------


def routing_interaction_analysis(
    task_probs: jax.Array,
    env_probs: jax.Array,
) -> float:
    """Estimate mutual information between task and env routing distributions.

    Operates on raw probability arrays so the function is testable without
    constructing a full ``DualHeadRouter``.

    Args:
        task_probs: Softmax routing probabilities, shape ``(B, num_tasks)``.
        env_probs:  Softmax routing probabilities, shape ``(B, num_envs)``.

    Returns:
        Non-negative float scalar estimating MI(task, env).  Zero means the
        two routing heads are fully independent (ideal for factorization).
    """
    # Joint distribution: average over batch of outer products
    joint = jnp.mean(task_probs[:, :, None] * env_probs[:, None, :], axis=0)  # (T, E)

    # Marginals
    p_task = jnp.mean(task_probs, axis=0)  # (T,)
    p_env = jnp.mean(env_probs, axis=0)  # (E,)

    # Product of marginals
    outer = p_task[:, None] * p_env[None, :]  # (T, E)

    # MI = sum_{t,e} P(t,e) * log(P(t,e) / (P(t)*P(e)))
    mi = jnp.sum(joint * jnp.log(joint / (outer + 1e-10) + 1e-10))

    # MI is non-negative by definition; clamp numerical noise
    return float(jnp.maximum(mi, 0.0))


# ---------------------------------------------------------------------------
# 4. Task-env entanglement (cosine similarity of adapter params)
# ---------------------------------------------------------------------------


def task_env_entanglement(
    task_params: dict,
    env_params: dict,
) -> float:
    """Cosine similarity between flattened task and environment adapter params.

    A value near zero means the two parameter sets are orthogonal (ideal for
    factorization); high absolute value indicates entanglement.

    Args:
        task_params: Nested dict of numpy-compatible arrays (e.g. from
            ``task_bank.get_adapter(task_id)``).
        env_params:  Nested dict of numpy-compatible arrays (e.g. from
            ``env_bank.get_env(env_id)``).

    Returns:
        Cosine similarity in ``[-1, 1]``.
    """
    a = _flatten_params(task_params)
    b = _flatten_params(env_params)

    # Align lengths: zero-pad the shorter vector
    if len(a) != len(b):
        max_len = max(len(a), len(b))
        a = np.pad(a, (0, max_len - len(a)))
        b = np.pad(b, (0, max_len - len(b)))

    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    return dot / (norm_a * norm_b + 1e-10)


def _flatten_params(params: dict) -> np.ndarray:
    """Recursively flatten a nested param dict into a 1-D numpy array."""
    leaves: list[np.ndarray] = []
    _collect_leaves(params, leaves)
    if not leaves:
        return np.zeros(1, dtype=np.float64)
    return np.concatenate([np.asarray(v, dtype=np.float64).ravel() for v in leaves])


def _collect_leaves(node: Any, acc: list[np.ndarray]) -> None:
    """DFS into nested dicts/lists to collect array leaves."""
    if isinstance(node, dict):
        for key in sorted(node.keys()):
            _collect_leaves(node[key], acc)
    elif isinstance(node, (list, tuple)):
        for item in node:
            _collect_leaves(item, acc)
    else:
        # Treat as array-like leaf
        acc.append(np.asarray(node, dtype=np.float64))


# ---------------------------------------------------------------------------
# 5. Counterfactual swap test
# ---------------------------------------------------------------------------


def counterfactual_swap_test(
    original_scores: dict[tuple[str, str], float],
    swapped_scores: dict[tuple[str, str], float],
) -> dict[str, float]:
    """Compare per-cell scores before and after swapping one adaptation factor.

    If factorization holds, swapping (e.g.) the env adapter while keeping the
    task adapter should not degrade performance.

    Args:
        original_scores: ``{(task_id, env_id): success_rate, ...}``
        swapped_scores:  ``{(task_id, env_id): success_rate, ...}`` after one
            factor was replaced with a different adapter.

    Returns:
        Dict with keys ``"mean_delta"``, ``"max_delta"``, ``"min_delta"``,
        ``"num_cells"``.  Deltas are ``swapped - original``; ``max_delta``
        and ``min_delta`` are over *absolute* deltas.
    """
    common_keys = set(original_scores.keys()) & set(swapped_scores.keys())
    deltas = np.array(
        [swapped_scores[k] - original_scores[k] for k in sorted(common_keys)],
        dtype=np.float64,
    )

    if len(deltas) == 0:
        return {
            "mean_delta": 0.0,
            "max_delta": 0.0,
            "min_delta": 0.0,
            "num_cells": 0.0,
        }

    abs_deltas = np.abs(deltas)
    return {
        "mean_delta": float(np.mean(deltas)),
        "max_delta": float(np.max(abs_deltas)),
        "min_delta": float(np.min(abs_deltas)),
        "num_cells": float(len(deltas)),
    }
