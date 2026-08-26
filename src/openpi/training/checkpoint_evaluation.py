"""Resolve the dataset and metric used for reward-gated checkpoints."""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class CheckpointEvaluationPlan:
    """Immutable checkpoint-evaluation provenance and loader configuration."""

    loader_config: Any
    source_repo_id: str
    metric_name: str
    reward_log_key: str
    loss_log_key: str
    uses_heldout_data: bool


def resolve_checkpoint_evaluation(config: Any) -> CheckpointEvaluationPlan:
    """Build a loader config for checkpoint scoring without mutating training data.

    When a held-out repo is configured, it reuses the training data factory and
    training normalization assets, changing only the LeRobot repo selected by
    the evaluation loader. This prevents checkpoint selection from fitting
    normalization statistics on held-out examples.
    """

    training_repo_id = config.data.repo_id
    if not isinstance(training_repo_id, str) or not training_repo_id:
        raise ValueError("Checkpoint evaluation requires a concrete training repo id")

    heldout_repo_id = config.checkpoint_eval_repo_id
    if heldout_repo_id is None:
        return CheckpointEvaluationPlan(
            loader_config=config,
            source_repo_id=training_repo_id,
            metric_name="offline_imitation_reward",
            reward_log_key="eval/offline_imitation_reward",
            loss_log_key="eval/offline_imitation_loss",
            uses_heldout_data=False,
        )
    if not isinstance(heldout_repo_id, str) or not heldout_repo_id:
        raise ValueError("checkpoint_eval_repo_id must be a non-empty string when set")
    if heldout_repo_id == training_repo_id:
        raise ValueError("Held-out checkpoint evaluation repo must differ from the training repo")

    evaluation_assets = dataclasses.replace(config.data.assets, asset_id=training_repo_id)
    evaluation_data = dataclasses.replace(
        config.data,
        repo_id=heldout_repo_id,
        assets=evaluation_assets,
    )
    return CheckpointEvaluationPlan(
        loader_config=dataclasses.replace(config, data=evaluation_data),
        source_repo_id=heldout_repo_id,
        metric_name="heldout_offline_imitation_reward",
        reward_log_key="eval/heldout_offline_imitation_reward",
        loss_log_key="eval/heldout_offline_imitation_loss",
        uses_heldout_data=True,
    )
