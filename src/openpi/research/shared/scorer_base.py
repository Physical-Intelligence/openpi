"""Base scorer interface and per-task scorer implementations.

Scorer design principle (from masterplan):
The scorer must not become the hidden confounder of the paper.

Every scorer must be validated against manual labels on a pilot subset.
"""

import dataclasses
from typing import Any, Protocol


@dataclasses.dataclass(frozen=True)
class ScorerResult:
    """Result from scoring a single episode."""

    success: bool
    confidence: float
    fail_type: str | None = None
    details: dict[str, Any] = dataclasses.field(default_factory=dict)


class Scorer(Protocol):
    """Protocol for task-specific scorers."""

    def score(self, episode: Any) -> ScorerResult:
        """Score a single episode for task success."""
        ...
