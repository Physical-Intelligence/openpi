"""Tests for continual_harness module."""

from __future__ import annotations

import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.research.shared.episode_schema import Action
from openpi.research.shared.episode_schema import Episode
from openpi.research.shared.episode_schema import EpisodeLabels
from openpi.research.shared.episode_schema import EpisodeMetadata
from openpi.research.shared.episode_schema import EpisodeStep
from openpi.research.shared.episode_schema import Observation
from openpi.research.shared.scorer_base import Scorer
from openpi.research.shared.scorer_base import ScorerResult
from openpi.research.spacecil.behavior_distillation import BehaviorDistillation
from openpi.research.spacecil.behavior_distillation import CalibrationMemory
from openpi.research.spacecil.behavior_distillation import TeacherSnapshot
from openpi.research.spacecil.continual_harness import ContinualHarness
from openpi.research.spacecil.continual_harness import ContinualResult
from openpi.research.spacecil.metrics import average_success
from openpi.research.spacecil.metrics import backward_transfer
from openpi.research.spacecil.metrics import forgetting
from openpi.research.spacecil.task_adapter_bank import TaskAdapterBank

# ---------------------------------------------------------------------------
# Mock / helper objects
# ---------------------------------------------------------------------------


class MockLoRAModel(nnx.Module):
    """Minimal nnx.Module with LoRA params for TaskAdapterBank compatibility."""

    def __init__(self, rngs: nnx.Rngs) -> None:
        self.backbone_w = nnx.Param(jnp.ones((4, 4)))
        self.lora_a = nnx.Param(jnp.zeros((4, 2)))
        self.lora_b = nnx.Param(jnp.zeros((2, 4)))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x


def _make_mock_model(seed: int = 0) -> MockLoRAModel:
    return MockLoRAModel(rngs=nnx.Rngs(seed))


class FixedScorer(Scorer):
    """Scorer that returns a deterministic success based on a fixed rate.

    For testing, if ``success_rate >= 1.0`` all episodes succeed,
    if ``success_rate <= 0.0`` all fail. Otherwise uses a simple threshold
    on episode index modulo for determinism.
    """

    def __init__(self, success_rate: float) -> None:
        self._rate = success_rate
        self._call_count = 0

    def score(self, episode: Episode) -> ScorerResult:
        # Deterministic: succeed on (call_count / total) < rate
        self._call_count += 1
        success = self._rate >= 1.0 or (self._rate > 0.0 and self._call_count % 2 == 1 and self._rate >= 0.5)
        # Simpler: just always succeed or always fail based on threshold
        success = self._rate >= 0.5
        return ScorerResult(success=success, confidence=self._rate)


class AlwaysSucceedScorer(Scorer):
    """Scorer that always reports success."""

    def score(self, episode: Episode) -> ScorerResult:
        return ScorerResult(success=True, confidence=1.0)


class AlwaysFailScorer(Scorer):
    """Scorer that always reports failure."""

    def score(self, episode: Episode) -> ScorerResult:
        return ScorerResult(success=False, confidence=0.0)


class FractionSucceedScorer(Scorer):
    """Scorer that succeeds for exactly the first ``n`` calls out of total.

    Reset via ``reset()`` between evaluation rounds.
    """

    def __init__(self, succeed_count: int) -> None:
        self._succeed_count = succeed_count
        self._calls = 0

    def reset(self) -> None:
        self._calls = 0

    def score(self, episode: Episode) -> ScorerResult:
        self._calls += 1
        success = self._calls <= self._succeed_count
        return ScorerResult(success=success, confidence=1.0 if success else 0.0)


def _make_episode(task_id: str = "task_0", env_id: str = "env_0", num_steps: int = 3) -> Episode:
    """Create a minimal valid Episode for testing."""
    steps = []
    for t in range(num_steps):
        obs = Observation(
            wrist_rgb=np.zeros((64, 64, 3), dtype=np.uint8),
            joint_position=np.full((7,), float(t) * 0.1, dtype=np.float32),
            joint_velocity=np.zeros((7,), dtype=np.float32),
            gripper_position=np.array([0.5], dtype=np.float32),
        )
        act = Action(
            joint_pos=np.zeros((7,), dtype=np.float32),
            gripper_cmd=0.5,
        )
        steps.append(EpisodeStep(observation=obs, action=act, timestamp_s=float(t)))
    return Episode(
        metadata=EpisodeMetadata(task_id=task_id, env_id=env_id),
        labels=EpisodeLabels(success=True),
        steps=steps,
        prompt=f"do {task_id}",
    )


def _mock_train_fn(task_id: str) -> tuple[MockLoRAModel, list[dict]]:
    """Mock training function that returns a mock model and info."""
    model = _make_mock_model()
    info = [{"loss": 0.5, "step": 1}, {"loss": 0.3, "step": 2}]
    return model, info


def _make_harness(
    task_ids: list[str],
    scorer_factory: type[Scorer] | None = None,
    *,
    with_distillation: bool = False,
    episodes_per_task: int = 5,
) -> ContinualHarness:
    """Build a ContinualHarness with mock components."""
    scorers: dict[str, Scorer] = {}
    eval_episodes: dict[str, list[Episode]] = {}
    for tid in task_ids:
        if scorer_factory is not None:
            scorers[tid] = scorer_factory()
        else:
            scorers[tid] = AlwaysSucceedScorer()
        eval_episodes[tid] = [_make_episode(task_id=tid) for _ in range(episodes_per_task)]

    distillation = None
    if with_distillation:
        distillation = BehaviorDistillation(
            memory=CalibrationMemory(),
            teacher=TeacherSnapshot(),
            distillation_weight=0.5,
        )

    return ContinualHarness(
        task_sequence=task_ids,
        adapter_bank=TaskAdapterBank(),
        distillation=distillation,
        scorers=scorers,
        eval_episodes=eval_episodes,
    )


# ---------------------------------------------------------------------------
# ContinualResult tests
# ---------------------------------------------------------------------------


class TestContinualResult:
    def test_continual_result_creation(self) -> None:
        """Create ContinualResult, verify fields."""
        mat = np.array([[0.8, 0.0], [0.7, 0.9]])
        result = ContinualResult(
            task_sequence=["t0", "t1"],
            result_matrix=mat,
            per_step_info=[{"a": 1}, {"b": 2}],
        )
        assert result.task_sequence == ["t0", "t1"]
        np.testing.assert_array_equal(result.result_matrix, mat)
        assert len(result.per_step_info) == 2

    def test_continual_result_matrix_shape(self) -> None:
        """result_matrix shape matches [len(task_sequence), len(task_sequence)]."""
        for n in [1, 2, 3, 5]:
            tasks = [f"t{i}" for i in range(n)]
            mat = np.zeros((n, n))
            result = ContinualResult(task_sequence=tasks, result_matrix=mat, per_step_info=[])
            assert result.result_matrix.shape == (n, n)

    def test_continual_result_frozen(self) -> None:
        """ContinualResult is frozen — cannot mutate fields."""
        result = ContinualResult(
            task_sequence=["t0"],
            result_matrix=np.zeros((1, 1)),
            per_step_info=[],
        )
        with pytest.raises(AttributeError):
            result.task_sequence = ["t1"]  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ContinualHarness.train_task tests
# ---------------------------------------------------------------------------


class TestTrainTask:
    def test_harness_train_single_task(self) -> None:
        """train_task with mock train_fn: adapter registered + frozen."""
        harness = _make_harness(["task_0"])
        info = harness.train_task("task_0", _mock_train_fn)

        # Adapter was registered and frozen.
        assert harness.adapter_bank.num_adapters == 1
        assert "task_0" in harness.adapter_bank.registered_tasks
        assert harness.adapter_bank.is_frozen("task_0")

        # Training info returned.
        assert len(info) == 2
        assert info[0]["loss"] == 0.5

    def test_harness_train_task_with_distillation(self) -> None:
        """With distillation configured, update_teacher is called."""
        harness = _make_harness(["task_0"], with_distillation=True)
        assert harness.distillation is not None

        harness.train_task("task_0", _mock_train_fn)

        # Teacher snapshot should have been taken.
        assert harness.distillation.teacher.has_snapshot

    def test_harness_train_task_without_distillation(self) -> None:
        """distillation=None works fine — no error raised."""
        harness = _make_harness(["task_0"], with_distillation=False)
        assert harness.distillation is None

        # Should not raise.
        info = harness.train_task("task_0", _mock_train_fn)
        assert len(info) == 2

    def test_harness_train_task_calls_train_fn_with_task_id(self) -> None:
        """train_fn is called with the correct task_id argument."""
        harness = _make_harness(["my_task"])
        called_with: list[str] = []

        def tracking_train_fn(task_id: str) -> tuple[MockLoRAModel, list[dict]]:
            called_with.append(task_id)
            return _make_mock_model(), []

        harness.train_task("my_task", tracking_train_fn)
        assert called_with == ["my_task"]

    def test_train_task_tracks_trained_tasks(self) -> None:
        """trained_tasks property is updated after training."""
        harness = _make_harness(["t0", "t1"])
        harness.train_task("t0", _mock_train_fn)
        assert harness.trained_tasks == ["t0"]

        harness.train_task("t1", _mock_train_fn)
        assert harness.trained_tasks == ["t0", "t1"]


# ---------------------------------------------------------------------------
# ContinualHarness.evaluate_all_tasks tests
# ---------------------------------------------------------------------------


class TestEvaluateAllTasks:
    def test_evaluate_all_tasks_scores(self) -> None:
        """Mock scorers return known values, verify returned dict."""
        harness = _make_harness(["t0", "t1"])
        harness.scorers["t0"] = AlwaysSucceedScorer()
        harness.scorers["t1"] = AlwaysFailScorer()

        # Train both tasks so they become "seen".
        harness.train_task("t0", _mock_train_fn)
        harness.train_task("t1", _mock_train_fn)

        scores = harness.evaluate_all_tasks()
        assert scores["t0"] == 1.0
        assert scores["t1"] == 0.0

    def test_evaluate_only_seen_tasks(self) -> None:
        """Only tasks trained so far get evaluated."""
        harness = _make_harness(["t0", "t1", "t2"])

        # Only train t0.
        harness.train_task("t0", _mock_train_fn)
        scores = harness.evaluate_all_tasks()

        assert "t0" in scores
        assert "t1" not in scores
        assert "t2" not in scores

    def test_evaluate_no_episodes_returns_zero(self) -> None:
        """Task with no eval episodes gets score 0.0."""
        harness = _make_harness(["t0"])
        harness.eval_episodes["t0"] = []  # empty
        harness.train_task("t0", _mock_train_fn)

        scores = harness.evaluate_all_tasks()
        assert scores["t0"] == 0.0

    def test_evaluate_no_scorer_returns_zero(self) -> None:
        """Task with no scorer entry gets score 0.0."""
        harness = _make_harness(["t0"])
        del harness.scorers["t0"]  # remove scorer
        harness.train_task("t0", _mock_train_fn)

        scores = harness.evaluate_all_tasks()
        assert scores["t0"] == 0.0


# ---------------------------------------------------------------------------
# ContinualHarness.run_sequence tests
# ---------------------------------------------------------------------------


class TestRunSequence:
    def test_run_sequence_two_tasks(self) -> None:
        """2-task mock sequence, verify result_matrix shape [2, 2]."""
        harness = _make_harness(["t0", "t1"])
        result = harness.run_sequence(_mock_train_fn)

        assert result.result_matrix.shape == (2, 2)
        assert result.task_sequence == ["t0", "t1"]

    def test_run_sequence_result_matrix_values(self) -> None:
        """Known mock scores, verify R[i][j] values."""
        harness = _make_harness(["t0", "t1"])
        harness.scorers["t0"] = AlwaysSucceedScorer()
        harness.scorers["t1"] = AlwaysSucceedScorer()

        result = harness.run_sequence(_mock_train_fn)

        # After training t0 (row 0): only t0 evaluated → R[0,0]=1.0, R[0,1]=NaN
        assert result.result_matrix[0, 0] == 1.0
        assert np.isnan(result.result_matrix[0, 1])

        # After training t1 (row 1): both evaluated → R[1,0]=1.0, R[1,1]=1.0
        assert result.result_matrix[1, 0] == 1.0
        assert result.result_matrix[1, 1] == 1.0

    def test_run_sequence_mixed_success(self) -> None:
        """Mixed success/failure scorers produce correct matrix values."""
        harness = _make_harness(["t0", "t1"])
        harness.scorers["t0"] = AlwaysSucceedScorer()
        harness.scorers["t1"] = AlwaysFailScorer()

        result = harness.run_sequence(_mock_train_fn)

        # Row 0: only t0 evaluated → 1.0
        assert result.result_matrix[0, 0] == 1.0
        # Row 1: t0=1.0, t1=0.0
        assert result.result_matrix[1, 0] == 1.0
        assert result.result_matrix[1, 1] == 0.0

    def test_run_sequence_three_tasks(self) -> None:
        """3-task sequence for non-trivial matrix."""
        tasks = ["t0", "t1", "t2"]
        harness = _make_harness(tasks)
        result = harness.run_sequence(_mock_train_fn)

        assert result.result_matrix.shape == (3, 3)
        assert result.task_sequence == tasks

        # Upper triangle should be NaN (unseen tasks).
        assert np.isnan(result.result_matrix[0, 1])
        assert np.isnan(result.result_matrix[0, 2])
        assert np.isnan(result.result_matrix[1, 2])

        # Diagonal and below should be filled (all succeed scorers).
        assert result.result_matrix[0, 0] == 1.0
        assert result.result_matrix[1, 0] == 1.0
        assert result.result_matrix[1, 1] == 1.0
        assert result.result_matrix[2, 0] == 1.0
        assert result.result_matrix[2, 1] == 1.0
        assert result.result_matrix[2, 2] == 1.0

    def test_run_sequence_per_step_info(self) -> None:
        """per_step_info populated with one entry per task."""
        harness = _make_harness(["t0", "t1"])
        result = harness.run_sequence(_mock_train_fn)

        assert len(result.per_step_info) == 2
        assert result.per_step_info[0]["task_id"] == "t0"
        assert result.per_step_info[1]["task_id"] == "t1"
        # Each should contain train_info list.
        assert "train_info" in result.per_step_info[0]
        assert len(result.per_step_info[0]["train_info"]) == 2  # from _mock_train_fn

    def test_harness_empty_task_sequence(self) -> None:
        """Empty task list → empty result."""
        harness = _make_harness([])
        result = harness.run_sequence(_mock_train_fn)

        assert result.task_sequence == []
        assert result.result_matrix.shape == (0, 0)
        assert result.per_step_info == []

    def test_run_sequence_with_distillation(self) -> None:
        """run_sequence with distillation enabled updates teacher after each task."""
        harness = _make_harness(["t0", "t1"], with_distillation=True)
        assert harness.distillation is not None

        result = harness.run_sequence(_mock_train_fn)

        # Teacher should have been updated (snapshot exists).
        assert harness.distillation.teacher.has_snapshot
        assert result.result_matrix.shape == (2, 2)


# ---------------------------------------------------------------------------
# Integration: adapter bank state after sequence
# ---------------------------------------------------------------------------


class TestAdapterBankAfterSequence:
    def test_adapter_bank_populated_after_sequence(self) -> None:
        """All task adapters in bank after run_sequence."""
        tasks = ["t0", "t1", "t2"]
        harness = _make_harness(tasks)
        harness.run_sequence(_mock_train_fn)

        assert harness.adapter_bank.num_adapters == 3
        assert harness.adapter_bank.registered_tasks == tasks
        for tid in tasks:
            assert harness.adapter_bank.is_frozen(tid)

    def test_adapters_have_lora_params(self) -> None:
        """Registered adapters contain LoRA parameter keys."""
        harness = _make_harness(["t0"])
        harness.run_sequence(_mock_train_fn)

        pure = harness.adapter_bank.get_adapter("t0")
        assert "lora_a" in pure
        assert "lora_b" in pure


# ---------------------------------------------------------------------------
# Integration: metrics on ContinualResult
# ---------------------------------------------------------------------------


class TestMetricsIntegration:
    def test_metrics_integration_all_succeed(self) -> None:
        """Feed ContinualResult into metrics functions — all succeed case."""
        harness = _make_harness(["t0", "t1", "t2"])
        result = harness.run_sequence(_mock_train_fn)

        # Replace NaN with 0 for metrics computation on full matrix.
        clean_matrix = np.nan_to_num(result.result_matrix, nan=0.0)

        avg = average_success(clean_matrix)
        assert 0.0 <= avg <= 1.0

        bt = backward_transfer(clean_matrix)
        assert isinstance(bt, float)

        fgt = forgetting(clean_matrix)
        assert isinstance(fgt, float)
        assert fgt >= 0.0  # forgetting is non-negative

    def test_metrics_integration_mixed(self) -> None:
        """Feed ContinualResult into metrics functions — mixed scores."""
        harness = _make_harness(["t0", "t1"])
        harness.scorers["t0"] = AlwaysSucceedScorer()
        harness.scorers["t1"] = AlwaysFailScorer()

        result = harness.run_sequence(_mock_train_fn)
        clean_matrix = np.nan_to_num(result.result_matrix, nan=0.0)

        avg = average_success(clean_matrix)
        # Final row: [1.0, 0.0] → avg = 0.5
        assert avg == pytest.approx(0.5)

        bt = backward_transfer(clean_matrix)
        assert isinstance(bt, float)

    def test_metrics_single_task(self) -> None:
        """Metrics handle single-task matrix gracefully."""
        harness = _make_harness(["t0"])
        result = harness.run_sequence(_mock_train_fn)

        avg = average_success(result.result_matrix)
        assert avg == pytest.approx(1.0)

        bt = backward_transfer(result.result_matrix)
        assert bt == 0.0  # single task → no backward transfer

        fgt = forgetting(result.result_matrix)
        assert fgt == 0.0  # single task → no forgetting
