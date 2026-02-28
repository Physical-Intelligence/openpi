"""Mission-aware forgetting metrics for SpaceCIL.

Provides:
- Standard CL metrics: average success, backward transfer, forgetting
- Mission-aware metrics: operationally weighted forgetting
- Diagnostic metrics: routing entropy, routing accuracy

All functions are pure (no side effects), deterministic, and operate on numpy arrays.
The result_matrix convention: R[i][j] = success rate on task j evaluated after training
up to task i. Shape is (T, T) where T is the number of tasks.
"""

from __future__ import annotations

import numpy as np


def average_success(result_matrix: np.ndarray) -> float:
    """Mean success rate across all tasks after training on the final task.

    Args:
        result_matrix: Shape (T, T). R[i][j] = success rate on task j after training task i.

    Returns:
        mean(R[-1, :]) — scalar.
    """
    return float(np.mean(result_matrix[-1, :]))


def backward_transfer(result_matrix: np.ndarray) -> float:
    """Average change from just-trained performance to final performance.

    Positive means improvement (positive transfer), negative means degradation.

    For each task j in [0, T-2], computes R[j][j] - R[-1][j] (performance right after
    training task j minus final performance on task j).

    Args:
        result_matrix: Shape (T, T).

    Returns:
        mean(R[j][j] - R[-1][j] for j in range(T-1)) — scalar.
        Returns 0.0 for a single-task matrix (T=1).
    """
    t = result_matrix.shape[0]
    if t <= 1:
        return 0.0
    diffs = np.array([result_matrix[j, j] - result_matrix[-1, j] for j in range(t - 1)])
    return float(np.mean(diffs))


def forgetting(result_matrix: np.ndarray) -> float:
    """Average drop from peak performance to final performance.

    For each task j in [0, T-2], computes max_k(R[k][j]) - R[-1][j].
    Always non-negative since max_k includes the final row.

    Args:
        result_matrix: Shape (T, T).

    Returns:
        mean(max_k(R[k][j]) - R[-1][j] for j in range(T-1)) — scalar.
        Returns 0.0 for a single-task matrix (T=1).
    """
    t = result_matrix.shape[0]
    if t <= 1:
        return 0.0
    peaks = np.max(result_matrix[:, : t - 1], axis=0)  # shape (T-1,)
    final = result_matrix[-1, : t - 1]  # shape (T-1,)
    drops = peaks - final
    return float(np.mean(drops))


def operational_forgetting(result_matrix: np.ndarray, weights: np.ndarray) -> float:
    """Weighted forgetting with per-task operational importance.

    sum_j(w[j] * (max_k(R[k][j]) - R[-1][j])) / sum_j(w[j]) for j in range(T-1).

    Args:
        result_matrix: Shape (T, T).
        weights: Shape (T,) — per-task operational importance. Only the first T-1
            entries are used (the last task has no forgetting by convention).

    Returns:
        Weighted forgetting scalar. Returns 0.0 if sum(weights[:T-1]) == 0 or T <= 1.
    """
    t = result_matrix.shape[0]
    if t <= 1:
        return 0.0
    w = weights[: t - 1]
    w_sum = float(np.sum(w))
    if w_sum == 0.0:
        return 0.0
    peaks = np.max(result_matrix[:, : t - 1], axis=0)  # shape (T-1,)
    final = result_matrix[-1, : t - 1]  # shape (T-1,)
    drops = peaks - final
    return float(np.sum(w * drops) / w_sum)


def routing_entropy(routing_probs: np.ndarray) -> np.ndarray:
    """Shannon entropy of routing probability distributions.

    Args:
        routing_probs: Shape (B, K) or (K,) where K = number of tasks.
            Each row should be a valid probability distribution (sums to ~1).

    Returns:
        Entropy per sample: shape (B,) or scalar (0-d array).
        Uses eps=1e-10 for numerical stability.
    """
    eps = 1e-10
    p = np.asarray(routing_probs, dtype=np.float64)
    return -np.sum(p * np.log(p + eps), axis=-1)


def routing_accuracy(predicted_tasks: np.ndarray, true_tasks: np.ndarray) -> float:
    """Fraction of correctly predicted task assignments.

    Args:
        predicted_tasks: Integer array of predicted task IDs.
        true_tasks: Integer array of ground-truth task IDs (same shape).

    Returns:
        mean(predicted == true) — scalar in [0, 1].
    """
    return float(np.mean(np.asarray(predicted_tasks) == np.asarray(true_tasks)))
