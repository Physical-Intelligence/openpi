"""Read-only compatibility checks for data and configured normalization statistics."""

import dataclasses

import numpy as np

from openpi.shared.normalize import NormStats


@dataclasses.dataclass(frozen=True)
class NormStatsReport:
    key: str
    data_dim: int
    stats_dim: int
    num_values: int
    nonfinite_values: int
    near_constant_dims: tuple[int, ...]
    outside_quantile_fraction: float | None
    largest_mismatch_dim: int | None

    @property
    def dimensions_match(self) -> bool:
        return self.data_dim == self.stats_dim


def analyze_norm_stats(
    key: str,
    values: np.ndarray,
    stats: NormStats,
    *,
    near_constant_threshold: float = 1e-6,
) -> NormStatsReport:
    """Compare transformed samples with one configured normalization-stat entry."""
    values = np.asarray(values)
    if values.ndim == 0:
        raise ValueError(f"Expected '{key}' samples to have at least one dimension.")

    values = values.reshape(-1, values.shape[-1])
    data_dim = values.shape[-1]
    stats_dim = np.asarray(stats.mean).shape[-1]
    finite = np.isfinite(values)
    nonfinite_values = int(values.size - np.count_nonzero(finite))

    finite_values = np.where(finite, values, np.nan)
    with np.errstate(invalid="ignore"):
        std = np.nanstd(finite_values, axis=0)
    near_constant_dims = tuple(np.flatnonzero(np.isfinite(std) & (std <= near_constant_threshold)).tolist())

    outside_fraction = None
    largest_mismatch_dim = None
    if stats.q01 is not None and stats.q99 is not None and data_dim == stats_dim:
        q01 = np.asarray(stats.q01)
        q99 = np.asarray(stats.q99)
        valid = finite & np.isfinite(q01)[None, :] & np.isfinite(q99)[None, :]
        outside = valid & ((values < q01) | (values > q99))
        valid_counts = valid.sum(axis=0)
        outside_counts = outside.sum(axis=0)
        total_valid = int(valid_counts.sum())
        if total_valid:
            outside_fraction = float(outside_counts.sum() / total_valid)
            per_dim = np.divide(
                outside_counts,
                valid_counts,
                out=np.full(data_dim, np.nan, dtype=np.float64),
                where=valid_counts > 0,
            )
            if np.any(np.isfinite(per_dim)):
                largest_mismatch_dim = int(np.nanargmax(per_dim))

    return NormStatsReport(
        key=key,
        data_dim=data_dim,
        stats_dim=stats_dim,
        num_values=values.size,
        nonfinite_values=nonfinite_values,
        near_constant_dims=near_constant_dims,
        outside_quantile_fraction=outside_fraction,
        largest_mismatch_dim=largest_mismatch_dim,
    )


def format_norm_stats_report(report: NormStatsReport, *, outside_warning_threshold: float = 0.05) -> str:
    """Format a compatibility report for terminal output."""
    outside = "unavailable" if report.outside_quantile_fraction is None else f"{report.outside_quantile_fraction:.1%}"
    mismatch = "unavailable" if report.largest_mismatch_dim is None else str(report.largest_mismatch_dim)
    result = "PASS"
    if (
        not report.dimensions_match
        or report.nonfinite_values
        or report.outside_quantile_fraction is None
        or report.outside_quantile_fraction > outside_warning_threshold
    ):
        result = "WARNING"

    return "\n".join(
        [
            report.key,
            f"  data dimensions:             {report.data_dim}",
            f"  norm-stat dimensions:        {report.stats_dim}",
            f"  non-finite values:           {report.nonfinite_values}",
            f"  near-constant dimensions:    {list(report.near_constant_dims)}",
            f"  outside configured q01-q99:  {outside}",
            f"  largest mismatch dimension:  {mismatch}",
            f"  result:                      {result}",
        ]
    )
