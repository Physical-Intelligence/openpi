"""Tests for scorer_base module."""

import numpy as np
import pytest

from openpi.research.shared.episode_schema import (
    Action,
    Episode,
    EpisodeLabels,
    EpisodeMetadata,
    EpisodeStep,
    Observation,
)
from openpi.research.shared.scorer_base import (
    ConnectorMatingScorer,
    LatchActuationScorer,
    PayloadTransferScorer,
    Scorer,
    ScorerResult,
    SurfaceCleaningScorer,
)


def _make_episode(
    n_steps: int,
    success: bool = True,
    joint_displacement: float = 1.0,
    gripper_closed: bool = True,
    joint_velocity: float = 0.01,
) -> Episode:
    """Create a mock episode for testing scorers."""
    metadata = EpisodeMetadata(task_id="test_task", env_id="test_env")
    labels = EpisodeLabels(success=success)

    initial_joint = np.zeros(7, dtype=np.float32)
    final_joint = np.ones(7, dtype=np.float32) * joint_displacement
    gripper_val = np.array([0.1], dtype=np.float32) if gripper_closed else np.array([0.9], dtype=np.float32)
    wrist_rgb = np.zeros((48, 64, 3), dtype=np.uint8)

    steps: list[EpisodeStep] = []
    for i in range(n_steps):
        t = i / max(n_steps - 1, 1)  # 0..1
        joint_pos = initial_joint + (final_joint - initial_joint) * t
        obs = Observation(
            wrist_rgb=wrist_rgb,
            joint_position=joint_pos.astype(np.float32),
            joint_velocity=np.ones(7, dtype=np.float32) * joint_velocity,
            gripper_position=gripper_val,
        )
        act = Action(joint_pos=joint_pos.astype(np.float32), gripper_cmd=0.1 if gripper_closed else 0.9)
        steps.append(EpisodeStep(observation=obs, action=act, timestamp_s=float(i) * 0.1))

    return Episode(metadata=metadata, labels=labels, steps=steps, prompt="test prompt")


# ---------------------------------------------------------------------------
# Existing tests (kept as-is)
# ---------------------------------------------------------------------------


def test_scorer_result_success():
    result = ScorerResult(success=True, confidence=0.95)
    assert result.success is True
    assert result.confidence == 0.95
    assert result.fail_type is None
    assert result.details == {}


def test_scorer_result_failure():
    result = ScorerResult(
        success=False,
        confidence=0.8,
        fail_type="timeout",
        details={"elapsed_s": 30.0},
    )
    assert result.success is False
    assert result.fail_type == "timeout"
    assert result.details["elapsed_s"] == 30.0


# ---------------------------------------------------------------------------
# ABC enforcement
# ---------------------------------------------------------------------------


def test_scorer_abc_enforcement():
    """Subclassing Scorer without implementing score() should raise TypeError."""

    class BadScorer(Scorer):
        pass

    with pytest.raises(TypeError):
        BadScorer()


# ---------------------------------------------------------------------------
# PayloadTransferScorer
# ---------------------------------------------------------------------------


def test_payload_transfer_success():
    ep = _make_episode(n_steps=10, joint_displacement=1.0, gripper_closed=True)
    scorer = PayloadTransferScorer(goal_region_threshold=0.1)
    result = scorer.score(ep)
    assert result.success is True
    assert result.confidence == 0.8


def test_payload_transfer_drop():
    ep = _make_episode(n_steps=10, joint_displacement=1.0, gripper_closed=False)
    scorer = PayloadTransferScorer(goal_region_threshold=0.1)
    result = scorer.score(ep)
    assert result.success is False
    assert result.fail_type == "drop"
    assert result.confidence == 0.7


def test_payload_transfer_timeout():
    ep = _make_episode(n_steps=10, joint_displacement=0.0, gripper_closed=True)
    scorer = PayloadTransferScorer(goal_region_threshold=0.1)
    result = scorer.score(ep)
    assert result.success is False
    assert result.fail_type == "timeout"
    assert result.confidence == 0.6


def test_payload_transfer_no_data():
    ep = _make_episode(n_steps=0)
    scorer = PayloadTransferScorer()
    result = scorer.score(ep)
    assert result.success is False
    assert result.fail_type == "no_data"
    assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# LatchActuationScorer
# ---------------------------------------------------------------------------


def test_latch_actuation_success():
    ep = _make_episode(n_steps=10, joint_displacement=1.0)
    scorer = LatchActuationScorer(actuation_threshold=0.5)
    result = scorer.score(ep)
    assert result.success is True
    assert result.confidence == 0.85


def test_latch_actuation_timeout():
    ep = _make_episode(n_steps=10, joint_displacement=0.1)
    scorer = LatchActuationScorer(actuation_threshold=0.5)
    result = scorer.score(ep)
    assert result.success is False
    assert result.fail_type == "timeout"
    assert result.confidence == 0.7


def test_latch_actuation_no_data():
    ep = _make_episode(n_steps=0)
    scorer = LatchActuationScorer()
    result = scorer.score(ep)
    assert result.success is False
    assert result.fail_type == "no_data"


# ---------------------------------------------------------------------------
# SurfaceCleaningScorer
# ---------------------------------------------------------------------------


def test_surface_cleaning_success():
    """Many varied joint positions should yield high coverage."""
    metadata = EpisodeMetadata(task_id="test_task", env_id="test_env")
    labels = EpisodeLabels(success=True)
    wrist_rgb = np.zeros((48, 64, 3), dtype=np.uint8)

    steps: list[EpisodeStep] = []
    for i in range(50):
        # Spread joint positions across wide range
        joint_pos = np.random.RandomState(seed=i).uniform(-2.0, 2.0, size=7).astype(np.float32)
        obs = Observation(
            wrist_rgb=wrist_rgb,
            joint_position=joint_pos,
            joint_velocity=np.zeros(7, dtype=np.float32),
            gripper_position=np.array([0.5], dtype=np.float32),
        )
        act = Action(joint_pos=joint_pos, gripper_cmd=0.5)
        steps.append(EpisodeStep(observation=obs, action=act))

    ep = Episode(metadata=metadata, labels=labels, steps=steps, prompt="clean")
    scorer = SurfaceCleaningScorer(coverage_threshold=0.5)
    result = scorer.score(ep)
    assert result.success is True
    assert result.confidence == 0.7


def test_surface_cleaning_timeout():
    """All same position should yield low coverage."""
    ep = _make_episode(n_steps=10, joint_displacement=0.0)
    scorer = SurfaceCleaningScorer(coverage_threshold=0.5)
    result = scorer.score(ep)
    assert result.success is False
    assert result.fail_type == "timeout"
    assert result.confidence == 0.6


def test_surface_cleaning_coverage_in_details():
    """Verify details['coverage'] exists and is float in [0, 1]."""
    ep = _make_episode(n_steps=10, joint_displacement=1.0)
    scorer = SurfaceCleaningScorer(coverage_threshold=0.5)
    result = scorer.score(ep)
    assert "coverage" in result.details
    cov = result.details["coverage"]
    assert isinstance(cov, float)
    assert 0.0 <= cov <= 1.0


# ---------------------------------------------------------------------------
# ConnectorMatingScorer
# ---------------------------------------------------------------------------


def test_connector_mating_success():
    ep = _make_episode(n_steps=10, gripper_closed=True, joint_velocity=0.001)
    scorer = ConnectorMatingScorer(stability_window=5, stability_threshold=0.02)
    result = scorer.score(ep)
    assert result.success is True
    assert result.confidence == 0.85


def test_connector_mating_contact():
    """Closed gripper but high velocity → contact but unstable."""
    ep = _make_episode(n_steps=10, gripper_closed=True, joint_velocity=0.5)
    scorer = ConnectorMatingScorer(stability_window=5, stability_threshold=0.02)
    result = scorer.score(ep)
    assert result.success is False
    assert result.fail_type == "contact"
    assert result.confidence == 0.6


def test_connector_mating_no_data():
    ep = _make_episode(n_steps=0)
    scorer = ConnectorMatingScorer()
    result = scorer.score(ep)
    assert result.success is False
    assert result.fail_type == "no_data"


# ---------------------------------------------------------------------------
# Cross-scorer checks
# ---------------------------------------------------------------------------


def test_all_scorers_confidence_range():
    """For each scorer, verify 0 <= confidence <= 1 on various episodes."""
    scorers: list[Scorer] = [
        PayloadTransferScorer(),
        LatchActuationScorer(),
        SurfaceCleaningScorer(),
        ConnectorMatingScorer(),
    ]
    episodes = [
        _make_episode(n_steps=0),
        _make_episode(n_steps=1, joint_displacement=0.0, gripper_closed=False),
        _make_episode(n_steps=10, joint_displacement=1.0, gripper_closed=True),
        _make_episode(n_steps=20, joint_displacement=0.5, gripper_closed=False, joint_velocity=0.5),
    ]
    for scorer in scorers:
        for ep in episodes:
            result = scorer.score(ep)
            assert 0.0 <= result.confidence <= 1.0, (
                f"{scorer.__class__.__name__} returned confidence {result.confidence}"
            )


def test_all_scorers_return_scorer_result():
    """Verify all scorers return ScorerResult type."""
    scorers: list[Scorer] = [
        PayloadTransferScorer(),
        LatchActuationScorer(),
        SurfaceCleaningScorer(),
        ConnectorMatingScorer(),
    ]
    ep = _make_episode(n_steps=10)
    for scorer in scorers:
        result = scorer.score(ep)
        assert isinstance(result, ScorerResult), (
            f"{scorer.__class__.__name__} returned {type(result)}, expected ScorerResult"
        )
