"""Sequential task training loop and backward transfer evaluation.

Orchestrates the full continual learning pipeline:
- Task sequencing
- Adapter training per task
- Optional behavior distillation
- Backward transfer evaluation after each task

DESIGN: The harness does NOT import openpi's train_step or init_train_state.
Instead, it takes a ``train_fn: Callable`` that the caller (e.g.
``scripts/train_spacecil.py``) provides. This makes the harness fully testable
with mock functions that don't require GPU or real JAX training.
"""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
import logging
from typing import Any

import numpy as np

from openpi.research.shared.episode_schema import Episode
from openpi.research.shared.scorer_base import Scorer
from openpi.research.spacecil.behavior_distillation import BehaviorDistillation
from openpi.research.spacecil.task_adapter_bank import TaskAdapterBank

logger = logging.getLogger(__name__)

# Type alias: train_fn(task_id) -> (trained_model, training_info_list)
# The model must be compatible with TaskAdapterBank.register_adapter (i.e. nnx.Module).
TrainFn = Callable[[str], tuple[Any, list[dict]]]


# ---------------------------------------------------------------------------
# ContinualResult — pure data container
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ContinualResult:
    """Immutable result container for a completed continual learning sequence.

    Attributes:
        task_sequence: Ordered list of task IDs as trained.
        result_matrix: Shape ``(T, T)`` numpy array where ``R[i][j]`` is the
            success rate on task *j* evaluated after training task *i*.
            Upper-triangle entries (``j > i``) are ``NaN`` for tasks not yet seen.
        per_step_info: Per-task training info dicts (loss curves, etc.).
    """

    task_sequence: list[str]
    result_matrix: np.ndarray  # shape [num_tasks, num_tasks]
    per_step_info: list[dict]  # one dict per task trained


# ---------------------------------------------------------------------------
# ContinualHarness — orchestrator
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ContinualHarness:
    """Testable orchestrator for sequential task training and evaluation.

    The harness owns the task sequence, adapter bank, optional distillation,
    scorers, and evaluation data. It delegates actual model training to a
    caller-provided ``train_fn`` to stay decoupled from JAX/openpi internals.

    Attributes:
        task_sequence: Ordered list of task IDs to train.
        adapter_bank: Registry for per-task LoRA adapters.
        distillation: Optional behavior distillation module (can be ``None``).
        scorers: Mapping ``{task_id: Scorer}`` for evaluation.
        eval_episodes: Mapping ``{task_id: list[Episode]}`` for evaluation.
        distillation_alpha: Blending weight for distillation (unused in harness
            logic, but stored for downstream scripts).
    """

    task_sequence: list[str]
    adapter_bank: TaskAdapterBank
    distillation: BehaviorDistillation | None
    scorers: dict[str, Scorer]
    eval_episodes: dict[str, list[Episode]]

    # Configuration
    distillation_alpha: float = 0.5

    # Internal tracking of which tasks have been trained so far.
    _trained_tasks: list[str] = dataclasses.field(default_factory=list, init=False)

    @property
    def trained_tasks(self) -> list[str]:
        """Task IDs that have been trained, in order."""
        return list(self._trained_tasks)

    # -----------------------------------------------------------------
    # Single-task training
    # -----------------------------------------------------------------

    def train_task(
        self,
        task_id: str,
        train_fn: TrainFn,
    ) -> list[dict]:
        """Train a single task using the caller-provided training function.

        Args:
            task_id: Which task to train.
            train_fn: ``callable(task_id) -> (trained_model, training_info_list)``.
                The caller provides the actual training loop (openpi's
                ``train_step``, etc.). This keeps the harness testable without
                real JAX training.

        Returns:
            Training info dicts from the training loop.

        Side effects:
            - Calls ``train_fn(task_id)`` to obtain trained model and info.
            - Registers the adapter in ``adapter_bank``.
            - Freezes the adapter.
            - Updates teacher snapshot if distillation is configured.
            - Appends ``task_id`` to internal ``_trained_tasks`` list.
        """
        logger.info("Training task: %s", task_id)

        trained_model, train_info = train_fn(task_id)

        # Register and freeze the adapter.
        self.adapter_bank.register_adapter(task_id, trained_model)
        self.adapter_bank.freeze_adapter(task_id)
        logger.info("Adapter registered and frozen for task: %s", task_id)

        # Update teacher snapshot if distillation is configured.
        if self.distillation is not None:
            self.distillation.update_teacher(trained_model)
            logger.info("Teacher snapshot updated after task: %s", task_id)

        # Track this task as trained.
        if task_id not in self._trained_tasks:
            self._trained_tasks.append(task_id)

        return train_info

    # -----------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------

    def evaluate_all_tasks(self) -> dict[str, float]:
        """Evaluate current model on all tasks seen so far.

        For each task in ``_trained_tasks``:
            1. Gets eval episodes from ``self.eval_episodes[task_id]``.
            2. Runs scorer on each episode.
            3. Computes mean success rate.

        Returns:
            ``{task_id: mean_success_rate}`` for all trained tasks.

        Note:
            This is a MOCK evaluation — it uses scorers on pre-collected
            episodes, not live rollouts. Real rollouts require a robot/sim
            environment which is out of scope.
        """
        results: dict[str, float] = {}
        for task_id in self._trained_tasks:
            episodes = self.eval_episodes.get(task_id, [])
            if not episodes:
                results[task_id] = 0.0
                continue

            successes = 0
            for ep in episodes:
                scorer = self.scorers.get(task_id)
                if scorer is None:
                    continue
                result = scorer.score(ep)
                if result.success:
                    successes += 1

            results[task_id] = successes / len(episodes) if episodes else 0.0

        return results

    # -----------------------------------------------------------------
    # Full sequence
    # -----------------------------------------------------------------

    def run_sequence(self, train_fn: TrainFn) -> ContinualResult:
        """Run the full continual learning sequence.

        For each task in ``task_sequence``:
            1. ``train_task(task_id, train_fn)``
            2. ``evaluate_all_tasks()``
            3. Record results in ``result_matrix``

        Args:
            train_fn: Training function passed through to ``train_task``.

        Returns:
            ``ContinualResult`` with the full result matrix and per-step info.
        """
        n = len(self.task_sequence)
        if n == 0:
            return ContinualResult(
                task_sequence=[],
                result_matrix=np.empty((0, 0), dtype=np.float64),
                per_step_info=[],
            )

        result_matrix = np.full((n, n), np.nan, dtype=np.float64)
        per_step_info: list[dict] = []

        for i, task_id in enumerate(self.task_sequence):
            # Train the task.
            train_info = self.train_task(task_id, train_fn)
            per_step_info.append({"task_id": task_id, "train_info": train_info})

            # Evaluate all tasks seen so far.
            eval_scores = self.evaluate_all_tasks()

            # Fill row i of the result matrix.
            for j, t_id in enumerate(self.task_sequence):
                if t_id in eval_scores:
                    result_matrix[i, j] = eval_scores[t_id]

        return ContinualResult(
            task_sequence=list(self.task_sequence),
            result_matrix=result_matrix,
            per_step_info=per_step_info,
        )
