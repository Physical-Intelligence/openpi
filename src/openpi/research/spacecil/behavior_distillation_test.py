"""Tests for behavior_distillation module."""

from __future__ import annotations

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.research.spacecil.behavior_distillation import BehaviorDistillation
from openpi.research.spacecil.behavior_distillation import CalibrationMemory
from openpi.research.spacecil.behavior_distillation import TeacherSnapshot
from openpi.research.spacecil.behavior_distillation import distillation_loss

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TinyModel(nnx.Module):
    """Minimal nnx.Module for snapshot tests."""

    def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(4, 2, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.linear(x)


def _make_episodes(n: int, prefix: str = "ep") -> list[dict]:
    """Create *n* dummy episode dicts."""
    return [{"id": f"{prefix}_{i}", "data": np.random.randn(3).tolist()} for i in range(n)]


# ===========================================================================
# CalibrationMemory tests
# ===========================================================================


class TestCalibrationMemory:
    def test_add_and_sample_two_tasks(self):
        """Add episodes for 2 tasks; sample_batch returns from both."""
        mem = CalibrationMemory(max_episodes_per_task=50)
        eps_a = _make_episodes(10, prefix="a")
        eps_b = _make_episodes(10, prefix="b")
        mem.add_episodes("task_a", eps_a)
        mem.add_episodes("task_b", eps_b)

        rng = np.random.default_rng(42)
        batch = mem.sample_batch(20, rng=rng)
        assert len(batch) == 20
        # Verify episodes from both tasks appear
        ids = {ep["id"] for ep in batch}
        has_a = any(i.startswith("a_") for i in ids)
        has_b = any(i.startswith("b_") for i in ids)
        assert has_a, f"Expected episodes from task_a, got ids: {ids}"
        assert has_b, f"Expected episodes from task_b, got ids: {ids}"

    def test_max_episodes_per_task_truncation(self):
        """Adding more than max_episodes_per_task keeps only the first max."""
        mem = CalibrationMemory(max_episodes_per_task=5)
        mem.add_episodes("task_x", _make_episodes(10, prefix="x"))
        assert mem.num_episodes("task_x") == 5

    def test_empty_memory_returns_empty_list(self):
        """sample_batch on empty memory returns []."""
        mem = CalibrationMemory()
        assert mem.sample_batch(10) == []

    def test_num_episodes_per_task_and_total(self):
        """num_episodes returns correct per-task and total counts."""
        mem = CalibrationMemory(max_episodes_per_task=100)
        mem.add_episodes("a", _make_episodes(3))
        mem.add_episodes("b", _make_episodes(7))
        assert mem.num_episodes("a") == 3
        assert mem.num_episodes("b") == 7
        assert mem.num_episodes() == 10
        assert mem.num_episodes("nonexistent") == 0

    def test_task_ids_property(self):
        """task_ids returns stored task IDs."""
        mem = CalibrationMemory()
        mem.add_episodes("alpha", _make_episodes(1))
        mem.add_episodes("beta", _make_episodes(1))
        assert sorted(mem.task_ids) == ["alpha", "beta"]

    def test_sample_batch_size_capped(self):
        """Requesting more than available returns all without error."""
        mem = CalibrationMemory()
        mem.add_episodes("t", _make_episodes(3))
        batch = mem.sample_batch(100)
        assert len(batch) == 3


# ===========================================================================
# TeacherSnapshot tests
# ===========================================================================


class TestTeacherSnapshot:
    def test_snapshot_stores_params(self):
        """snapshot() stores params and has_snapshot returns True."""
        teacher = TeacherSnapshot()
        assert not teacher.has_snapshot

        model = _TinyModel(rngs=nnx.Rngs(0))
        teacher.snapshot(model)
        assert teacher.has_snapshot
        params = teacher.get_params()
        assert isinstance(params, dict)
        assert len(params) > 0

    def test_get_params_before_snapshot_raises(self):
        """get_params() before snapshot raises RuntimeError."""
        teacher = TeacherSnapshot()
        with pytest.raises(RuntimeError, match="No teacher snapshot available"):
            teacher.get_params()

    def test_teacher_params_independent_of_model(self):
        """Teacher params don't change when model is modified after snapshot."""
        model = _TinyModel(rngs=nnx.Rngs(0))
        teacher = TeacherSnapshot()
        teacher.snapshot(model)

        # Capture teacher params before modification
        params_before = teacher.get_params()
        kernel_before = params_before["linear"]["kernel"].copy()

        # Mutate the model weights
        model.linear.kernel = nnx.Param(model.linear.kernel.value + 999.0)

        # Teacher params should be unchanged
        params_after = teacher.get_params()
        kernel_after = params_after["linear"]["kernel"]
        np.testing.assert_array_equal(kernel_before, kernel_after)


# ===========================================================================
# distillation_loss tests
# ===========================================================================


class TestDistillationLoss:
    def test_mse_known_values(self):
        """MSE with known inputs produces expected output."""
        # student = [[1, 2]], teacher = [[3, 4]]
        # diff = [[-2, -2]], sq = [[4, 4]], mean = 4.0
        student = jnp.array([[[1.0, 2.0]]])  # (1, 1, 2)
        teacher = jnp.array([[[3.0, 4.0]]])  # (1, 1, 2)
        loss = distillation_loss(student, teacher, loss_type="mse")
        np.testing.assert_allclose(float(loss), 4.0, atol=1e-6)

    def test_mse_identical_inputs_zero(self):
        """MSE of identical arrays is 0."""
        x = jnp.ones((2, 5, 8))
        loss = distillation_loss(x, x, loss_type="mse")
        np.testing.assert_allclose(float(loss), 0.0, atol=1e-7)

    def test_l1_known_values(self):
        """L1 with known inputs produces expected output."""
        # student = [[1, 2]], teacher = [[4, 6]]
        # abs diff = [[3, 4]], mean = 3.5
        student = jnp.array([[[1.0, 2.0]]])
        teacher = jnp.array([[[4.0, 6.0]]])
        loss = distillation_loss(student, teacher, loss_type="l1")
        np.testing.assert_allclose(float(loss), 3.5, atol=1e-6)

    def test_invalid_loss_type_raises(self):
        """Invalid loss_type raises ValueError."""
        x = jnp.ones((1, 1, 1))
        with pytest.raises(ValueError, match="Unknown loss_type"):
            distillation_loss(x, x, loss_type="kl")

    def test_stop_gradient_applied(self):
        """Verify stop_gradient is applied on teacher_actions."""
        student = jnp.array([[[1.0, 2.0]]])
        teacher = jnp.array([[[3.0, 4.0]]])

        # Grad w.r.t. teacher should be zero because of stop_gradient
        grad_fn = jax.grad(lambda t: float(distillation_loss(student, t, loss_type="mse")))
        grad_teacher = grad_fn(teacher)
        np.testing.assert_allclose(grad_teacher, 0.0, atol=1e-7)


# ===========================================================================
# BehaviorDistillation tests
# ===========================================================================


class TestBehaviorDistillation:
    def test_lambda_zero_total_equals_task_loss(self):
        """When distillation_weight=0, total_loss equals task_loss exactly."""
        bd = BehaviorDistillation(
            memory=CalibrationMemory(),
            teacher=TeacherSnapshot(),
            distillation_weight=0.0,
        )
        task_loss = jnp.array(2.5)
        student = jnp.ones((2, 4, 8))
        teacher = jnp.zeros((2, 4, 8))  # different from student

        total, metrics = bd.compute_total_loss(task_loss, student, teacher)
        np.testing.assert_allclose(float(total), 2.5, atol=1e-7)
        np.testing.assert_allclose(float(metrics["task_loss"]), 2.5, atol=1e-7)
        np.testing.assert_allclose(float(metrics["distill_loss"]), 0.0, atol=1e-7)
        np.testing.assert_allclose(float(metrics["total_loss"]), 2.5, atol=1e-7)

    def test_lambda_one_correct_weighted_sum(self):
        """When distillation_weight=1, total = task_loss + distill_loss."""
        bd = BehaviorDistillation(
            memory=CalibrationMemory(),
            teacher=TeacherSnapshot(),
            distillation_weight=1.0,
        )
        task_loss = jnp.array(1.0)
        # student=2, teacher=0 → MSE = mean(4) = 4.0
        student = jnp.full((1, 1, 1), 2.0)
        teacher = jnp.full((1, 1, 1), 0.0)

        total, metrics = bd.compute_total_loss(task_loss, student, teacher)
        expected_distill = 4.0
        expected_total = 1.0 + 1.0 * 4.0
        np.testing.assert_allclose(float(metrics["distill_loss"]), expected_distill, atol=1e-6)
        np.testing.assert_allclose(float(total), expected_total, atol=1e-6)
        np.testing.assert_allclose(float(metrics["total_loss"]), expected_total, atol=1e-6)

    def test_lambda_half_correct_weighted_sum(self):
        """When distillation_weight=0.5, total = task_loss + 0.5 * distill_loss."""
        bd = BehaviorDistillation(
            memory=CalibrationMemory(),
            teacher=TeacherSnapshot(),
            distillation_weight=0.5,
        )
        task_loss = jnp.array(3.0)
        student = jnp.full((1, 1, 1), 2.0)
        teacher = jnp.full((1, 1, 1), 0.0)

        total, metrics = bd.compute_total_loss(task_loss, student, teacher)
        expected_total = 3.0 + 0.5 * 4.0
        np.testing.assert_allclose(float(total), expected_total, atol=1e-6)

    def test_update_teacher_creates_snapshot(self):
        """update_teacher creates a teacher snapshot."""
        bd = BehaviorDistillation(
            memory=CalibrationMemory(),
            teacher=TeacherSnapshot(),
        )
        assert not bd.teacher.has_snapshot
        model = _TinyModel(rngs=nnx.Rngs(0))
        bd.update_teacher(model)
        assert bd.teacher.has_snapshot

    def test_add_calibration_episodes_delegates(self):
        """add_calibration_episodes stores episodes in memory."""
        bd = BehaviorDistillation(
            memory=CalibrationMemory(),
            teacher=TeacherSnapshot(),
        )
        bd.add_calibration_episodes("t1", _make_episodes(5, prefix="t1"))
        assert bd.memory.num_episodes("t1") == 5

    def test_l1_loss_type_propagates(self):
        """BehaviorDistillation with loss_type='l1' uses L1 distillation."""
        bd = BehaviorDistillation(
            memory=CalibrationMemory(),
            teacher=TeacherSnapshot(),
            distillation_weight=1.0,
            loss_type="l1",
        )
        task_loss = jnp.array(0.0)
        # student=3, teacher=0 → L1 = mean(|3|) = 3.0
        student = jnp.full((1, 1, 1), 3.0)
        teacher = jnp.full((1, 1, 1), 0.0)

        total, metrics = bd.compute_total_loss(task_loss, student, teacher)
        np.testing.assert_allclose(float(metrics["distill_loss"]), 3.0, atol=1e-6)
        np.testing.assert_allclose(float(total), 3.0, atol=1e-6)
