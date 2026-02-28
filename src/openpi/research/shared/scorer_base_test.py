"""Tests for scorer_base module."""

from openpi.research.shared.scorer_base import ScorerResult


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
