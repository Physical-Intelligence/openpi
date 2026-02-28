"""Tests for missing_corner_harness module."""

from __future__ import annotations

import numpy as np
import pytest

from openpi.research.lunarcompose.missing_corner_harness import (
    MissingCornerHarness,
    MissingCornerResult,
    _CANONICAL_SPLITS,
)
from openpi.research.shared.episode_schema import (
    Action,
    Episode,
    EpisodeLabels,
    EpisodeMetadata,
    EpisodeStep,
    Observation,
)
from openpi.research.shared.scorer_base import Scorer, ScorerResult

# ---------------------------------------------------------------------------
# Constants matching the paper's canonical grid
# ---------------------------------------------------------------------------

_TASK_IDS = ["payload", "latch", "clean", "connector"]
_ENV_IDS = ["nominal", "shadow", "contamination"]
_ALL_CELLS = {(t, e) for t in _TASK_IDS for e in _ENV_IDS}  # 12 cells


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_episode(task_id: str, env_id: str, success: bool = True) -> Episode:
    """Create a minimal Episode for testing."""
    obs = Observation(
        wrist_rgb=np.zeros((64, 64, 3), dtype=np.uint8),
        joint_position=np.zeros(7),
        joint_velocity=np.zeros(7),
        gripper_position=np.zeros(1),
    )
    action = Action(joint_pos=np.zeros(7), gripper_cmd=0.0)
    step = EpisodeStep(observation=obs, action=action)
    return Episode(
        metadata=EpisodeMetadata(task_id=task_id, env_id=env_id),
        labels=EpisodeLabels(success=success),
        steps=[step],
        prompt=f"do {task_id}",
    )


class _MockScorer(Scorer):
    """Scorer that always returns a fixed success value."""

    def __init__(self, always_succeed: bool = True) -> None:
        self._succeed = always_succeed

    def score(self, episode: Episode) -> ScorerResult:
        return ScorerResult(
            success=self._succeed,
            confidence=1.0 if self._succeed else 0.0,
        )


class _CellAwareScorer(Scorer):
    """Scorer that returns success based on a per-cell lookup."""

    def __init__(self, success_cells: set[tuple[str, str]]) -> None:
        self._success_cells = success_cells

    def score(self, episode: Episode) -> ScorerResult:
        cell = (episode.metadata.task_id, episode.metadata.env_id)
        success = cell in self._success_cells
        return ScorerResult(success=success, confidence=1.0 if success else 0.0)


def _make_harness(
    scorers: dict[str, Scorer] | None = None,
    eval_episodes: dict[tuple[str, str], list[Episode]] | None = None,
) -> MissingCornerHarness:
    """Create a harness with default mock data for all 12 cells."""
    if scorers is None:
        scorers = {t: _MockScorer(always_succeed=True) for t in _TASK_IDS}
    if eval_episodes is None:
        eval_episodes = {(t, e): [_make_episode(t, e)] for t in _TASK_IDS for e in _ENV_IDS}
    return MissingCornerHarness(
        task_ids=list(_TASK_IDS),
        env_ids=list(_ENV_IDS),
        scorers=scorers,
        eval_episodes=eval_episodes,
    )


# ---------------------------------------------------------------------------
# Smoke test (preserved from original)
# ---------------------------------------------------------------------------


def test_missing_corner_harness_module_imports():
    """Verify the missing_corner_harness module can be imported."""
    from openpi.research.lunarcompose import missing_corner_harness  # noqa: F401


# ---------------------------------------------------------------------------
# Behavioral tests
# ---------------------------------------------------------------------------


class TestSplitCoverage:
    """test_split_coverage: Every rotation has full task and env coverage in train."""

    @pytest.mark.parametrize("rotation", [0, 1, 2])
    def test_every_task_in_train(self, rotation: int) -> None:
        harness = _make_harness()
        train_cells, _test_cells = harness.generate_split(rotation)
        train_tasks = {t for t, _e in train_cells}
        for task_id in _TASK_IDS:
            assert task_id in train_tasks, f"Rotation {rotation}: task '{task_id}' missing from train cells"

    @pytest.mark.parametrize("rotation", [0, 1, 2])
    def test_every_env_in_train(self, rotation: int) -> None:
        harness = _make_harness()
        train_cells, _test_cells = harness.generate_split(rotation)
        train_envs = {e for _t, e in train_cells}
        for env_id in _ENV_IDS:
            assert env_id in train_envs, f"Rotation {rotation}: env '{env_id}' missing from train cells"

    @pytest.mark.parametrize("rotation", [0, 1, 2])
    def test_split_sizes(self, rotation: int) -> None:
        harness = _make_harness()
        train_cells, test_cells = harness.generate_split(rotation)
        assert len(train_cells) == 8, f"Expected 8 train cells, got {len(train_cells)}"
        assert len(test_cells) == 4, f"Expected 4 test cells, got {len(test_cells)}"


class TestNoLeakage:
    """test_no_leakage: Train and test are disjoint and cover all cells."""

    @pytest.mark.parametrize("rotation", [0, 1, 2])
    def test_disjoint(self, rotation: int) -> None:
        harness = _make_harness()
        train_cells, test_cells = harness.generate_split(rotation)
        overlap = train_cells & test_cells
        assert len(overlap) == 0, f"Rotation {rotation}: {len(overlap)} cells in both train and test: {overlap}"

    @pytest.mark.parametrize("rotation", [0, 1, 2])
    def test_complete_partition(self, rotation: int) -> None:
        harness = _make_harness()
        train_cells, test_cells = harness.generate_split(rotation)
        union = train_cells | test_cells
        assert union == _ALL_CELLS, (
            f"Rotation {rotation}: union of train+test != all cells. "
            f"Missing: {_ALL_CELLS - union}, Extra: {union - _ALL_CELLS}"
        )


class TestConstraintViolationRaises:
    """test_constraint_violation_raises: Invalid splits are caught."""

    def test_missing_task_raises(self) -> None:
        harness = _make_harness()
        # Manually set train_cells that exclude all "connector" cells
        harness.train_cells = {(t, e) for t in ["payload", "latch", "clean"] for e in _ENV_IDS}
        harness.test_cells = {("connector", e) for e in _ENV_IDS}
        with pytest.raises(ValueError, match="connector"):
            harness.validate_split()

    def test_missing_env_raises(self) -> None:
        harness = _make_harness()
        # Manually set train_cells that exclude all "contamination" cells
        harness.train_cells = {(t, e) for t in _TASK_IDS for e in ["nominal", "shadow"]}
        harness.test_cells = {(t, "contamination") for t in _TASK_IDS}
        with pytest.raises(ValueError, match="contamination"):
            harness.validate_split()

    def test_overlap_raises(self) -> None:
        harness = _make_harness()
        overlap_cell = ("payload", "nominal")
        harness.train_cells = _ALL_CELLS.copy()
        harness.test_cells = {overlap_cell}
        with pytest.raises(ValueError, match="appear in both"):
            harness.validate_split()

    def test_incomplete_partition_raises(self) -> None:
        harness = _make_harness()
        # Leave out one cell entirely
        harness.train_cells = _ALL_CELLS - {("payload", "nominal"), ("latch", "shadow")}
        harness.test_cells = {("payload", "nominal")}
        # ("latch", "shadow") is missing from both
        with pytest.raises(ValueError, match="does not cover all cells"):
            harness.validate_split()

    def test_invalid_rotation_raises(self) -> None:
        harness = _make_harness()
        with pytest.raises(ValueError, match="Invalid rotation"):
            harness.generate_split(rotation=5)


class TestResultStructure:
    """test_result_structure: MissingCornerResult has correct types."""

    def test_result_fields(self) -> None:
        harness = _make_harness()
        harness.generate_split(rotation=0)
        result = harness.evaluate_all_cells()

        assert isinstance(result, MissingCornerResult)
        assert isinstance(result.per_cell_success, dict)
        assert isinstance(result.seen_mean, float)
        assert isinstance(result.unseen_mean, float)
        assert isinstance(result.seen_unseen_gap, float)
        assert isinstance(result.per_task_breakdown, dict)
        assert isinstance(result.per_env_breakdown, dict)
        assert isinstance(result.rotation_id, int)
        assert isinstance(result.timestamp, str)

    def test_result_cell_count(self) -> None:
        harness = _make_harness()
        harness.generate_split(rotation=0)
        result = harness.evaluate_all_cells()

        # Should have all 12 cells
        assert len(result.per_cell_success) == 12

    def test_result_breakdown_keys(self) -> None:
        harness = _make_harness()
        harness.generate_split(rotation=0)
        result = harness.evaluate_all_cells()

        assert set(result.per_task_breakdown.keys()) == set(_TASK_IDS)
        assert set(result.per_env_breakdown.keys()) == set(_ENV_IDS)

    def test_result_rotation_id(self) -> None:
        for rotation in [0, 1, 2]:
            harness = _make_harness()
            harness.generate_split(rotation=rotation)
            result = harness.evaluate_all_cells()
            assert result.rotation_id == rotation

    def test_result_is_frozen(self) -> None:
        harness = _make_harness()
        harness.generate_split(rotation=0)
        result = harness.evaluate_all_cells()
        with pytest.raises(AttributeError):
            result.seen_mean = 0.99  # type: ignore[misc]

    def test_no_split_raises_runtime_error(self) -> None:
        harness = _make_harness()
        with pytest.raises(RuntimeError, match="No split has been generated"):
            harness.evaluate_all_cells()


class TestMockEvaluation:
    """test_mock_evaluation: Per-cell scores match expected scorer outputs."""

    def test_all_succeed(self) -> None:
        scorers = {t: _MockScorer(always_succeed=True) for t in _TASK_IDS}
        harness = _make_harness(scorers=scorers)
        harness.generate_split(rotation=0)
        result = harness.evaluate_all_cells()

        for cell, rate in result.per_cell_success.items():
            assert rate == pytest.approx(1.0), f"Cell {cell} expected 1.0, got {rate}"

    def test_all_fail(self) -> None:
        scorers = {t: _MockScorer(always_succeed=False) for t in _TASK_IDS}
        harness = _make_harness(scorers=scorers)
        harness.generate_split(rotation=0)
        result = harness.evaluate_all_cells()

        for cell, rate in result.per_cell_success.items():
            assert rate == pytest.approx(0.0), f"Cell {cell} expected 0.0, got {rate}"

    def test_cell_aware_scoring(self) -> None:
        """Scorer that succeeds only for specific cells produces correct per-cell rates."""
        success_cells = {
            ("payload", "nominal"),
            ("latch", "shadow"),
            ("clean", "contamination"),
        }
        scorers = {t: _CellAwareScorer(success_cells) for t in _TASK_IDS}
        harness = _make_harness(scorers=scorers)
        harness.generate_split(rotation=0)
        result = harness.evaluate_all_cells()

        for cell, rate in result.per_cell_success.items():
            expected = 1.0 if cell in success_cells else 0.0
            assert rate == pytest.approx(expected), f"Cell {cell} expected {expected}, got {rate}"

    def test_empty_episodes_default_to_zero(self) -> None:
        """Cells with no episodes should score 0.0."""
        eval_episodes: dict[tuple[str, str], list[Episode]] = {}  # no episodes at all
        harness = _make_harness(eval_episodes=eval_episodes)
        harness.generate_split(rotation=0)
        result = harness.evaluate_all_cells()

        for cell, rate in result.per_cell_success.items():
            assert rate == pytest.approx(0.0), f"Cell {cell} expected 0.0, got {rate}"

    def test_multiple_episodes_per_cell(self) -> None:
        """Mean success rate with mixed pass/fail episodes."""
        # Cell (payload, nominal): 2 succeed, 1 fail → 2/3
        eval_episodes: dict[tuple[str, str], list[Episode]] = {}
        for t in _TASK_IDS:
            for e in _ENV_IDS:
                eval_episodes[(t, e)] = [_make_episode(t, e)]

        # Override payload/nominal with 3 episodes
        eval_episodes[("payload", "nominal")] = [
            _make_episode("payload", "nominal"),
            _make_episode("payload", "nominal"),
            _make_episode("payload", "nominal"),
        ]

        # Scorer: first two episodes succeed, third fails
        call_count: dict[str, int] = {"n": 0}

        class _CountingScorer(Scorer):
            def score(self, episode: Episode) -> ScorerResult:
                cell = (episode.metadata.task_id, episode.metadata.env_id)
                if cell == ("payload", "nominal"):
                    call_count["n"] += 1
                    # 2 out of 3 succeed
                    return ScorerResult(
                        success=call_count["n"] <= 2,
                        confidence=1.0,
                    )
                return ScorerResult(success=True, confidence=1.0)

        scorers = {t: _CountingScorer() for t in _TASK_IDS}
        harness = _make_harness(scorers=scorers, eval_episodes=eval_episodes)
        harness.generate_split(rotation=0)
        result = harness.evaluate_all_cells()

        assert result.per_cell_success[("payload", "nominal")] == pytest.approx(2.0 / 3.0, abs=1e-6)


class TestSeenUnseenGap:
    """test_seen_unseen_gap: Verify gap computation with known values."""

    def test_gap_with_known_values(self) -> None:
        """Seen cells average 0.8, unseen cells average 0.4, gap = 0.4."""
        harness = _make_harness()
        harness.generate_split(rotation=0)

        # Build cell-aware scorer: seen cells succeed 80%, unseen 40%
        seen = harness.train_cells
        unseen = harness.test_cells

        # For simplicity: give each cell 5 episodes, seen cells 4/5 succeed,
        # unseen cells 2/5 succeed.
        eval_episodes: dict[tuple[str, str], list[Episode]] = {}
        for t in _TASK_IDS:
            for e in _ENV_IDS:
                eval_episodes[(t, e)] = [_make_episode(t, e) for _ in range(5)]

        # Scorer that returns success based on position in call sequence per cell
        cell_call_count: dict[tuple[str, str], int] = {}

        class _GapScorer(Scorer):
            def score(self, episode: Episode) -> ScorerResult:
                cell = (episode.metadata.task_id, episode.metadata.env_id)
                cell_call_count.setdefault(cell, 0)
                cell_call_count[cell] += 1
                idx = cell_call_count[cell]

                if cell in seen:
                    # 4 out of 5 succeed → 0.8
                    success = idx <= 4
                else:
                    # 2 out of 5 succeed → 0.4
                    success = idx <= 2

                return ScorerResult(success=success, confidence=1.0)

        scorers = {t: _GapScorer() for t in _TASK_IDS}
        harness_eval = MissingCornerHarness(
            task_ids=list(_TASK_IDS),
            env_ids=list(_ENV_IDS),
            scorers=scorers,
            eval_episodes=eval_episodes,
            train_cells=harness.train_cells,
            test_cells=harness.test_cells,
        )
        harness_eval._rotation_id = 0

        result = harness_eval.evaluate_all_cells()

        assert result.seen_mean == pytest.approx(0.8, abs=1e-6)
        assert result.unseen_mean == pytest.approx(0.4, abs=1e-6)
        assert result.seen_unseen_gap == pytest.approx(0.4, abs=1e-6)

    def test_zero_gap_when_all_same(self) -> None:
        """When all cells have the same success rate, gap should be 0."""
        scorers = {t: _MockScorer(always_succeed=True) for t in _TASK_IDS}
        harness = _make_harness(scorers=scorers)
        harness.generate_split(rotation=0)
        result = harness.evaluate_all_cells()

        assert result.seen_mean == pytest.approx(1.0)
        assert result.unseen_mean == pytest.approx(1.0)
        assert result.seen_unseen_gap == pytest.approx(0.0)


class TestConvenienceAliases:
    """Test seen_cells() and unseen_cells() convenience methods."""

    def test_seen_equals_train(self) -> None:
        harness = _make_harness()
        harness.generate_split(rotation=0)
        assert harness.seen_cells() == harness.train_cells

    def test_unseen_equals_test(self) -> None:
        harness = _make_harness()
        harness.generate_split(rotation=0)
        assert harness.unseen_cells() == harness.test_cells

    def test_aliases_return_copies(self) -> None:
        """Aliases should return copies, not references to internal state."""
        harness = _make_harness()
        harness.generate_split(rotation=0)
        seen = harness.seen_cells()
        seen.add(("fake", "cell"))
        assert ("fake", "cell") not in harness.train_cells


class TestCanonicalSplits:
    """Verify the hardcoded canonical splits match the paper's definitions."""

    def test_rotation_0_test_cells(self) -> None:
        expected_test = {
            ("payload", "contamination"),
            ("latch", "shadow"),
            ("clean", "nominal"),
            ("connector", "shadow"),
        }
        assert _CANONICAL_SPLITS[0] == expected_test

    def test_rotation_1_test_cells(self) -> None:
        expected_test = {
            ("payload", "nominal"),
            ("latch", "contamination"),
            ("clean", "shadow"),
            ("connector", "contamination"),
        }
        assert _CANONICAL_SPLITS[1] == expected_test

    def test_rotation_2_test_cells(self) -> None:
        expected_test = {
            ("payload", "shadow"),
            ("latch", "nominal"),
            ("clean", "contamination"),
            ("connector", "nominal"),
        }
        assert _CANONICAL_SPLITS[2] == expected_test

    def test_all_rotations_have_4_test_cells(self) -> None:
        for rotation, test_cells in _CANONICAL_SPLITS.items():
            assert len(test_cells) == 4, f"Rotation {rotation}: expected 4 test cells, got {len(test_cells)}"
