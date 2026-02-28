"""Anti-forgetting via teacher snapshot and calibration memory.

Teacher = previous policy snapshot (frozen).
Memory = small calibration set from earlier tasks.
Loss = action-space or latent-space imitation term.
"""

from __future__ import annotations

import dataclasses

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# CalibrationMemory
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CalibrationMemory:
    """Buffer storing calibration episodes from previous tasks.

    Each task contributes up to ``max_episodes_per_task`` episodes.
    ``sample_batch`` draws uniformly across all stored tasks.
    """

    max_episodes_per_task: int = 100
    _episodes: dict[str, list[dict]] = dataclasses.field(default_factory=dict)

    def add_episodes(self, task_id: str, episodes: list[dict]) -> None:
        """Add episodes for a task, truncating to ``max_episodes_per_task``."""
        existing = self._episodes.get(task_id, [])
        existing.extend(episodes)
        self._episodes[task_id] = existing[: self.max_episodes_per_task]

    def sample_batch(
        self,
        batch_size: int,
        rng: np.random.Generator | None = None,
    ) -> list[dict]:
        """Sample ``batch_size`` episodes uniformly across all stored tasks.

        Returns an empty list when no episodes are stored.
        """
        all_episodes: list[dict] = []
        for eps in self._episodes.values():
            all_episodes.extend(eps)
        if not all_episodes:
            return []
        if rng is None:
            rng = np.random.default_rng()
        indices = rng.choice(len(all_episodes), size=min(batch_size, len(all_episodes)), replace=False)
        return [all_episodes[i] for i in indices]

    def num_episodes(self, task_id: str | None = None) -> int:
        """Count episodes for a specific task or total across all tasks."""
        if task_id is not None:
            return len(self._episodes.get(task_id, []))
        return sum(len(eps) for eps in self._episodes.values())

    @property
    def task_ids(self) -> list[str]:
        """Return list of task IDs with stored episodes."""
        return list(self._episodes.keys())


# ---------------------------------------------------------------------------
# TeacherSnapshot
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TeacherSnapshot:
    """Frozen copy of model parameters for distillation.

    Stores a pure dict snapshot of the full model state (backbone + LoRA).
    The teacher is always frozen — no gradient flow through it.
    """

    params: dict | None = None

    def snapshot(self, model: nnx.Module) -> None:
        """Extract and deep-copy full model state from an nnx.Module."""
        pure_dict = nnx.state(model).to_pure_dict()
        self.params = jax.tree.map(
            lambda x: x.copy() if hasattr(x, "copy") else x,
            pure_dict,
        )

    @property
    def has_snapshot(self) -> bool:
        """Whether a snapshot has been taken."""
        return self.params is not None

    def get_params(self) -> dict:
        """Return stored params. Raises RuntimeError if no snapshot exists."""
        if self.params is None:
            raise RuntimeError("No teacher snapshot available. Call snapshot() first.")
        return self.params


# ---------------------------------------------------------------------------
# distillation_loss
# ---------------------------------------------------------------------------


def distillation_loss(
    student_actions: jax.Array,
    teacher_actions: jax.Array,
    loss_type: str = "mse",
) -> jax.Array:
    """Compute distillation loss between student and teacher velocity predictions.

    Args:
        student_actions: shape ``(B, action_horizon, action_dim)`` — student predictions.
        teacher_actions: shape ``(B, action_horizon, action_dim)`` — teacher predictions
            (stop_gradient applied to prevent gradient flow).
        loss_type: ``"mse"`` for mean-squared-error (default),
            ``"l1"`` for mean absolute error (ablation).

    Returns:
        Scalar loss value.

    Raises:
        ValueError: If ``loss_type`` is not ``"mse"`` or ``"l1"``.
    """
    teacher_sg = jax.lax.stop_gradient(teacher_actions)
    if loss_type == "mse":
        return jnp.mean(jnp.square(student_actions - teacher_sg))
    if loss_type == "l1":
        return jnp.mean(jnp.abs(student_actions - teacher_sg))
    raise ValueError(f"Unknown loss_type: {loss_type!r}. Expected 'mse' or 'l1'.")


# ---------------------------------------------------------------------------
# BehaviorDistillation orchestrator
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class BehaviorDistillation:
    """Orchestrator combining calibration memory, teacher snapshot, and distillation loss.

    Usage::

        bd = BehaviorDistillation(
            memory=CalibrationMemory(),
            teacher=TeacherSnapshot(),
            distillation_weight=0.5,
        )
        # After training task 0, snapshot the teacher
        bd.update_teacher(model)
        bd.add_calibration_episodes("task_0", episodes)

        # During task 1 training
        total_loss, metrics = bd.compute_total_loss(task_loss, student_actions, teacher_actions)
    """

    memory: CalibrationMemory
    teacher: TeacherSnapshot
    distillation_weight: float = 0.5
    loss_type: str = "mse"

    def compute_total_loss(
        self,
        task_loss: jax.Array,
        student_actions: jax.Array,
        teacher_actions: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Compute combined task + distillation loss.

        Args:
            task_loss: Scalar task-specific loss (e.g. flow matching loss).
            student_actions: ``(B, action_horizon, action_dim)`` student predictions.
            teacher_actions: ``(B, action_horizon, action_dim)`` teacher predictions.

        Returns:
            Tuple of ``(total_loss, metrics_dict)`` where ``metrics_dict`` contains
            ``"task_loss"``, ``"distill_loss"``, and ``"total_loss"`` scalars.
        """
        if self.distillation_weight == 0.0:
            return task_loss, {
                "task_loss": task_loss,
                "distill_loss": jnp.zeros_like(task_loss),
                "total_loss": task_loss,
            }

        distill = distillation_loss(student_actions, teacher_actions, loss_type=self.loss_type)
        total = task_loss + self.distillation_weight * distill
        return total, {
            "task_loss": task_loss,
            "distill_loss": distill,
            "total_loss": total,
        }

    def update_teacher(self, model: nnx.Module) -> None:
        """Snapshot current model as the teacher."""
        self.teacher.snapshot(model)

    def add_calibration_episodes(self, task_id: str, episodes: list[dict]) -> None:
        """Add calibration episodes for a task to the replay memory."""
        self.memory.add_episodes(task_id, episodes)
