"""Embodiment configuration for multi-robot data pipelines."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import openpi.transforms as _transforms


@dataclasses.dataclass(frozen=True)
class EmbodimentConfig:
    """Configuration for a single embodiment (robot) in a multi-embodiment pipeline.

    Each embodiment has its own data path, transforms, normalization statistics,
    key mappings, and sampling weight. This replaces the monolithic approach of
    applying one set of transforms to all datasets.
    """

    # Human-readable name (e.g., "rby1_gripper", "agibot_dexhand").
    name: str
    # Integer tag for identifying this embodiment in batches.
    tag_id: int
    # Dimensionality of the action space for this robot.
    action_dim: int
    # LeRobot repo_id or local path for this embodiment's dataset.
    data_path: str

    # Per-embodiment repack transform that maps dataset-specific keys to a
    # standard internal schema. Replaces a single global repack mapping.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Per-embodiment data transforms (applied before normalization).
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)

    # Mapping from dataset keys to internal schema keys.
    # Used to build a RepackTransform when repack_transforms is not provided directly.
    key_mapping: dict[str, str] = dataclasses.field(default_factory=dict)

    # Sampling weight for balancing data from this embodiment relative to others.
    # Higher weight means more samples will be drawn from this embodiment.
    sampling_weight: float = 1.0

    # If true, load prompt from the LeRobot task field.
    prompt_from_task: bool = False
    # Default prompt to inject if no prompt is present in the data.
    default_prompt: str | None = None

    # Action sequence key names (for LeRobot delta_timestamps).
    action_sequence_keys: Sequence[str] = ("actions",)
    # If true, use quantile normalization instead of z-score.
    use_quantile_norm: bool = False

    def get_repack_transforms(self) -> _transforms.Group:
        """Return the repack transforms, building from key_mapping if needed."""
        if self.repack_transforms.inputs or self.repack_transforms.outputs:
            return self.repack_transforms
        if self.key_mapping:
            return _transforms.Group(
                inputs=[_transforms.RepackTransform(self.key_mapping)]
            )
        return _transforms.Group()
