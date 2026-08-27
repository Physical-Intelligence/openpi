import numpy as np
import pytest

from openpi.shared.normalize import NormStats
from openpi.training.norm_stats_report import analyze_norm_stats
from openpi.training.norm_stats_report import format_norm_stats_report


def _stats(dim: int = 2) -> NormStats:
    return NormStats(
        mean=np.zeros(dim),
        std=np.ones(dim),
        q01=np.full(dim, -1.0),
        q99=np.full(dim, 1.0),
    )


def test_matching_samples_report_outside_fraction_and_largest_dimension():
    values = np.array([[0.0, 0.0], [2.0, 0.0], [3.0, 2.0]])

    report = analyze_norm_stats("actions", values, _stats())

    assert report.dimensions_match
    assert report.outside_quantile_fraction == pytest.approx(3 / 6)
    assert report.largest_mismatch_dim == 0


def test_dimension_mismatch_skips_quantile_comparison():
    report = analyze_norm_stats("state", np.ones((4, 3)), _stats(dim=2))

    assert not report.dimensions_match
    assert report.outside_quantile_fraction is None
    assert report.largest_mismatch_dim is None


def test_counts_nonfinite_values_without_corrupting_other_checks():
    values = np.array([[0.0, np.nan], [np.inf, 0.0], [0.5, 0.0]])

    report = analyze_norm_stats("actions", values, _stats())

    assert report.nonfinite_values == 2
    assert report.outside_quantile_fraction == 0.0


def test_identifies_near_constant_dimensions():
    values = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])

    report = analyze_norm_stats("state", values, _stats())

    assert report.near_constant_dims == (0,)


def test_rejects_scalar_samples():
    with pytest.raises(ValueError, match="at least one dimension"):
        analyze_norm_stats("state", np.asarray(1.0), _stats(dim=1))


def test_format_includes_warning_for_dimension_mismatch():
    report = analyze_norm_stats("state", np.ones((4, 3)), _stats(dim=2))

    output = format_norm_stats_report(report)

    assert "state" in output
    assert "norm-stat dimensions:        2" in output
    assert "result:                      WARNING" in output


def test_format_warns_when_too_many_values_are_outside_quantiles():
    report = analyze_norm_stats("actions", np.array([[2.0, 0.0], [0.0, 0.0]]), _stats())

    output = format_norm_stats_report(report, outside_warning_threshold=0.05)

    assert "outside configured q01-q99:  25.0%" in output
    assert "result:                      WARNING" in output
