"""Base scorer interface and per-task scorer implementations.

Scorer design principle (from masterplan):
The scorer must not become the hidden confounder of the paper.

Every scorer must be validated against manual labels on a pilot subset.
"""

import abc
import dataclasses
from typing import Any

import numpy as np

from openpi.research.shared.episode_schema import Episode


@dataclasses.dataclass(frozen=True)
class ScorerResult:
    """Result from scoring a single episode."""

    success: bool
    confidence: float
    fail_type: str | None = None
    details: dict[str, Any] = dataclasses.field(default_factory=dict)


class Scorer(abc.ABC):
    """Abstract base class for task-specific scorers."""

    @abc.abstractmethod
    def score(self, episode: Episode) -> ScorerResult:
        """Score a single episode for task success.

        Args:
            episode: Complete episode to evaluate.

        Returns:
            ScorerResult with success determination and confidence.
        """


class PayloadTransferScorer(Scorer):
    """Checks if payload reached goal region.

    Uses gripper state and joint displacement as heuristic proxies:
    - Gripper closed → payload was grasped
    - Sufficient joint displacement → arm moved to goal region
    """

    def __init__(self, goal_region_threshold: float = 0.1) -> None:
        self.goal_region_threshold = goal_region_threshold

    def score(self, episode: Episode) -> ScorerResult:
        if len(episode.steps) == 0:
            return ScorerResult(success=False, confidence=0.0, fail_type="no_data")

        initial_joint_pos = episode.steps[0].observation.joint_position
        final_joint_pos = episode.steps[-1].observation.joint_position
        final_gripper = episode.steps[-1].observation.gripper_position

        gripper_closed = float(final_gripper[0]) < 0.5
        displacement = float(np.linalg.norm(final_joint_pos - initial_joint_pos))
        moved = displacement > self.goal_region_threshold

        if gripper_closed and moved:
            return ScorerResult(success=True, confidence=0.8)
        elif not gripper_closed and moved:
            return ScorerResult(success=False, confidence=0.7, fail_type="drop")
        elif gripper_closed and not moved:
            return ScorerResult(success=False, confidence=0.6, fail_type="timeout")
        else:
            return ScorerResult(success=False, confidence=0.5, fail_type="other")


class LatchActuationScorer(Scorer):
    """Checks if latch reached actuated state.

    Uses maximum single-joint displacement as a proxy for latch actuation.
    """

    def __init__(self, actuation_threshold: float = 0.5) -> None:
        self.actuation_threshold = actuation_threshold

    def score(self, episode: Episode) -> ScorerResult:
        if len(episode.steps) == 0:
            return ScorerResult(success=False, confidence=0.0, fail_type="no_data")

        initial_joint_pos = episode.steps[0].observation.joint_position
        final_joint_pos = episode.steps[-1].observation.joint_position
        max_displacement = float(np.max(np.abs(final_joint_pos - initial_joint_pos)))

        if max_displacement > self.actuation_threshold:
            return ScorerResult(success=True, confidence=0.85)
        else:
            return ScorerResult(success=False, confidence=0.7, fail_type="timeout")


class SurfaceCleaningScorer(Scorer):
    """Checks if surface was cleaned.

    Estimates workspace coverage by discretising the joint trajectory into bins
    and measuring what fraction of the bin-space was visited.
    """

    def __init__(self, coverage_threshold: float = 0.5) -> None:
        self.coverage_threshold = coverage_threshold

    def score(self, episode: Episode) -> ScorerResult:
        if len(episode.steps) == 0:
            return ScorerResult(success=False, confidence=0.0, fail_type="no_data")

        n_bins = 10
        joint_positions = np.array([s.observation.joint_position for s in episode.steps])  # (T, 7)

        # Discretise each joint to bins over the observed range
        total_bins_visited = 0
        total_bins_possible = 0
        for j in range(joint_positions.shape[1]):
            col = joint_positions[:, j]
            col_min, col_max = float(np.min(col)), float(np.max(col))
            if col_max - col_min < 1e-8:
                # No range → 1 bin visited out of n_bins
                total_bins_visited += 1
                total_bins_possible += n_bins
            else:
                # Bin indices for each timestep
                normalised = (col - col_min) / (col_max - col_min)
                bin_indices = np.clip((normalised * n_bins).astype(int), 0, n_bins - 1)
                unique_bins = len(set(bin_indices.tolist()))
                total_bins_visited += unique_bins
                total_bins_possible += n_bins

        coverage_fraction = total_bins_visited / total_bins_possible if total_bins_possible > 0 else 0.0

        if coverage_fraction > self.coverage_threshold:
            return ScorerResult(success=True, confidence=0.7, details={"coverage": coverage_fraction})
        else:
            return ScorerResult(
                success=False,
                confidence=0.6,
                fail_type="timeout",
                details={"coverage": coverage_fraction},
            )


class ConnectorMatingScorer(Scorer):
    """Checks if connector was mated.

    Uses gripper closure and final-step stability (low joint velocity) as
    heuristic proxies for successful mating.
    """

    def __init__(self, stability_window: int = 5, stability_threshold: float = 0.02) -> None:
        self.stability_window = stability_window
        self.stability_threshold = stability_threshold

    def score(self, episode: Episode) -> ScorerResult:
        if len(episode.steps) == 0:
            return ScorerResult(success=False, confidence=0.0, fail_type="no_data")

        # Check last stability_window steps (or all if fewer)
        window_steps = episode.steps[-self.stability_window :]
        velocities = np.array([s.observation.joint_velocity for s in window_steps])  # (W, 7)
        mean_abs_velocity = float(np.mean(np.abs(velocities)))

        final_gripper = episode.steps[-1].observation.gripper_position
        gripper_closed = float(final_gripper[0]) < 0.5

        if gripper_closed and mean_abs_velocity < self.stability_threshold:
            return ScorerResult(success=True, confidence=0.85)
        elif gripper_closed and mean_abs_velocity >= self.stability_threshold:
            return ScorerResult(success=False, confidence=0.6, fail_type="contact")
        else:
            return ScorerResult(success=False, confidence=0.5, fail_type="other")
