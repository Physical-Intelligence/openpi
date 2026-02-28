"""Missing-corner compositional evaluation harness.

Train on subset of task×env combinations, evaluate on unseen combinations.
Supports multiple rotated splits for statistical robustness.

DESIGN: This is the scientific centerpiece of Paper B (LunarCompose).
The harness manages a 2D grid of (task_id, env_id) cells and partitions
them into train (seen) and test (unseen) subsets. Three canonical split
rotations are hardcoded from the paper to ensure reproducibility.

The harness evaluates per-cell success rates using pre-collected episodes
and task-specific scorers, then computes aggregate metrics including the
seen-unseen gap — the primary indicator of compositional generalization.
"""

from __future__ import annotations

import dataclasses
import datetime
import logging

from openpi.research.shared.episode_schema import Episode
from openpi.research.shared.scorer_base import Scorer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical split definitions (hardcoded from the paper)
# ---------------------------------------------------------------------------

_TASK_IDS = ["payload", "latch", "clean", "connector"]
_ENV_IDS = ["nominal", "shadow", "contamination"]

# Each rotation maps to its TEST cells; train = all_cells - test_cells.
_CANONICAL_SPLITS: dict[int, set[tuple[str, str]]] = {
    # Rotation 0 (canonical A): 4 held-out test cells
    0: {
        ("payload", "contamination"),
        ("latch", "shadow"),
        ("clean", "nominal"),
        ("connector", "shadow"),
    },
    # Rotation 1 (B): 4 held-out test cells
    1: {
        ("payload", "nominal"),
        ("latch", "contamination"),
        ("clean", "shadow"),
        ("connector", "contamination"),
    },
    # Rotation 2 (C): 4 held-out test cells
    2: {
        ("payload", "shadow"),
        ("latch", "nominal"),
        ("clean", "contamination"),
        ("connector", "nominal"),
    },
}


# ---------------------------------------------------------------------------
# MissingCornerResult — pure data container
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class MissingCornerResult:
    """Result of missing-corner compositional evaluation.

    Attributes:
        per_cell_success: (task_id, env_id) -> success rate [0, 1].
        seen_mean: Mean success over train (seen) cells.
        unseen_mean: Mean success over test (unseen) cells.
        seen_unseen_gap: seen_mean - unseen_mean.
        per_task_breakdown: task_id -> mean success across all envs.
        per_env_breakdown: env_id -> mean success across all tasks.
        rotation_id: Which split rotation (0, 1, 2).
        timestamp: ISO format timestamp of evaluation.
    """

    per_cell_success: dict[tuple[str, str], float]
    seen_mean: float
    unseen_mean: float
    seen_unseen_gap: float
    per_task_breakdown: dict[str, float]
    per_env_breakdown: dict[str, float]
    rotation_id: int
    timestamp: str


# ---------------------------------------------------------------------------
# MissingCornerHarness — orchestrator
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MissingCornerHarness:
    """Manages train/test splits over a task × environment grid.

    The harness owns the grid definition, canonical split rotations,
    scorers, and evaluation episodes. It delegates actual model training
    to external scripts and focuses on evaluation and split management.

    Attributes:
        task_ids: Ordered list of task IDs (e.g. ["payload", "latch", ...]).
        env_ids: Ordered list of environment IDs (e.g. ["nominal", "shadow", ...]).
        scorers: Mapping {task_id: Scorer} for evaluation.
        eval_episodes: Mapping {(task_id, env_id): [Episode, ...]} for evaluation.
        train_cells: Set of (task_id, env_id) cells used for training (seen).
        test_cells: Set of (task_id, env_id) cells held out for evaluation (unseen).
    """

    task_ids: list[str]
    env_ids: list[str]
    scorers: dict[str, Scorer]
    eval_episodes: dict[tuple[str, str], list[Episode]]

    # Computed after generate_split
    train_cells: set[tuple[str, str]] = dataclasses.field(default_factory=set)
    test_cells: set[tuple[str, str]] = dataclasses.field(default_factory=set)
    _rotation_id: int = dataclasses.field(default=-1, init=False)

    @property
    def all_cells(self) -> set[tuple[str, str]]:
        """All possible (task_id, env_id) cells in the grid."""
        return {(t, e) for t in self.task_ids for e in self.env_ids}

    # -----------------------------------------------------------------
    # Split generation
    # -----------------------------------------------------------------

    def generate_split(self, rotation: int = 0) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
        """Generate train/test split for the given rotation.

        Args:
            rotation: Split rotation index (0, 1, or 2).

        Returns:
            Tuple of (train_cells, test_cells).

        Raises:
            ValueError: If rotation not in {0, 1, 2} or if the generated
                split fails validation.
        """
        if rotation not in _CANONICAL_SPLITS:
            raise ValueError(f"Invalid rotation {rotation}. Must be one of {sorted(_CANONICAL_SPLITS.keys())}.")

        all_cells = self.all_cells
        test_cells = _CANONICAL_SPLITS[rotation] & all_cells
        train_cells = all_cells - test_cells

        self.train_cells = train_cells
        self.test_cells = test_cells
        self._rotation_id = rotation

        logger.info(
            "Generated split rotation %d: %d train cells, %d test cells",
            rotation,
            len(train_cells),
            len(test_cells),
        )

        # Validate the split — raises if invalid.
        self.validate_split()

        return (train_cells, test_cells)

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    def validate_split(self) -> bool:
        """Validate the current train/test split.

        Checks four constraints:
            1. Every task_id appears in at least one training cell.
            2. Every env_id appears in at least one training cell.
            3. No overlap between train_cells and test_cells.
            4. train_cells ∪ test_cells covers all cells (complete partition).

        Returns:
            True if all constraints pass.

        Raises:
            ValueError: With specific message if any constraint fails.
        """
        # Constraint 1: every task in at least one train cell
        train_tasks = {t for t, _e in self.train_cells}
        missing_tasks = set(self.task_ids) - train_tasks
        if missing_tasks:
            raise ValueError(
                f"Constraint violation: tasks {sorted(missing_tasks)} have no "
                f"training cell. Every task must appear in at least one training cell."
            )

        # Constraint 2: every env in at least one train cell
        train_envs = {e for _t, e in self.train_cells}
        missing_envs = set(self.env_ids) - train_envs
        if missing_envs:
            raise ValueError(
                f"Constraint violation: environments {sorted(missing_envs)} have no "
                f"training cell. Every env must appear in at least one training cell."
            )

        # Constraint 3: no overlap
        overlap = self.train_cells & self.test_cells
        if overlap:
            raise ValueError(
                f"Constraint violation: {len(overlap)} cell(s) appear in both train and test sets: {sorted(overlap)}."
            )

        # Constraint 4: complete partition
        all_cells = self.all_cells
        union = self.train_cells | self.test_cells
        if union != all_cells:
            missing = all_cells - union
            extra = union - all_cells
            parts = []
            if missing:
                parts.append(f"missing cells: {sorted(missing)}")
            if extra:
                parts.append(f"extra cells: {sorted(extra)}")
            raise ValueError(f"Constraint violation: train ∪ test does not cover all cells. " + "; ".join(parts))

        return True

    # -----------------------------------------------------------------
    # Convenience aliases
    # -----------------------------------------------------------------

    def seen_cells(self) -> set[tuple[str, str]]:
        """Alias for train_cells (seen during training)."""
        return set(self.train_cells)

    def unseen_cells(self) -> set[tuple[str, str]]:
        """Alias for test_cells (unseen during training)."""
        return set(self.test_cells)

    # -----------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------

    def evaluate_all_cells(self) -> MissingCornerResult:
        """Evaluate current model on all cells in the grid.

        For each cell (task_id, env_id) in both train and test cells:
            1. Get episodes from self.eval_episodes[(task_id, env_id)].
            2. Get scorer from self.scorers[task_id].
            3. Compute mean success rate over episodes.

        If no episodes exist for a cell, success rate defaults to 0.0.

        Returns:
            MissingCornerResult with per-cell scores and aggregate metrics.

        Raises:
            RuntimeError: If no split has been generated yet.
        """
        if self._rotation_id < 0:
            raise RuntimeError("No split has been generated. Call generate_split() first.")

        all_eval_cells = self.train_cells | self.test_cells
        per_cell_success: dict[tuple[str, str], float] = {}

        for task_id, env_id in all_eval_cells:
            episodes = self.eval_episodes.get((task_id, env_id), [])
            if not episodes:
                per_cell_success[(task_id, env_id)] = 0.0
                continue

            scorer = self.scorers.get(task_id)
            if scorer is None:
                logger.warning("No scorer for task '%s', defaulting to 0.0", task_id)
                per_cell_success[(task_id, env_id)] = 0.0
                continue

            successes = sum(1 for ep in episodes if scorer.score(ep).success)
            per_cell_success[(task_id, env_id)] = successes / len(episodes)

        # Aggregate metrics: seen / unseen means
        seen_scores = [per_cell_success[c] for c in self.train_cells if c in per_cell_success]
        unseen_scores = [per_cell_success[c] for c in self.test_cells if c in per_cell_success]

        seen_mean = sum(seen_scores) / len(seen_scores) if seen_scores else 0.0
        unseen_mean = sum(unseen_scores) / len(unseen_scores) if unseen_scores else 0.0
        seen_unseen_gap = seen_mean - unseen_mean

        # Per-task breakdown: mean across all envs for each task
        per_task_breakdown: dict[str, float] = {}
        for task_id in self.task_ids:
            task_scores = [
                per_cell_success[(task_id, env_id)] for env_id in self.env_ids if (task_id, env_id) in per_cell_success
            ]
            per_task_breakdown[task_id] = sum(task_scores) / len(task_scores) if task_scores else 0.0

        # Per-env breakdown: mean across all tasks for each env
        per_env_breakdown: dict[str, float] = {}
        for env_id in self.env_ids:
            env_scores = [
                per_cell_success[(task_id, env_id)]
                for task_id in self.task_ids
                if (task_id, env_id) in per_cell_success
            ]
            per_env_breakdown[env_id] = sum(env_scores) / len(env_scores) if env_scores else 0.0

        timestamp = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()

        return MissingCornerResult(
            per_cell_success=per_cell_success,
            seen_mean=seen_mean,
            unseen_mean=unseen_mean,
            seen_unseen_gap=seen_unseen_gap,
            per_task_breakdown=per_task_breakdown,
            per_env_breakdown=per_env_breakdown,
            rotation_id=self._rotation_id,
            timestamp=timestamp,
        )
