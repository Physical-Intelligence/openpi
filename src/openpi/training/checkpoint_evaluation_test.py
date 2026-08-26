import dataclasses

import pytest

from openpi.training.checkpoint_evaluation import resolve_checkpoint_evaluation


@dataclasses.dataclass(frozen=True)
class _Assets:
    assets_dir: str | None = None
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class _Data:
    repo_id: str
    assets: _Assets = dataclasses.field(default_factory=_Assets)
    use_depth_image: bool = True


@dataclasses.dataclass(frozen=True)
class _Config:
    data: _Data
    checkpoint_eval_repo_id: str | None = None


def test_default_checkpoint_evaluation_preserves_legacy_training_batches() -> None:
    config = _Config(data=_Data(repo_id="local/train"))

    plan = resolve_checkpoint_evaluation(config)

    assert plan.loader_config is config
    assert plan.source_repo_id == "local/train"
    assert plan.metric_name == "offline_imitation_reward"
    assert not plan.uses_heldout_data


def test_heldout_checkpoint_evaluation_changes_only_repo_and_reuses_training_norm_assets() -> None:
    config = _Config(
        data=_Data(repo_id="local/train", assets=_Assets(assets_dir="/assets")),
        checkpoint_eval_repo_id="local/eval",
    )

    plan = resolve_checkpoint_evaluation(config)

    assert plan.loader_config is not config
    assert plan.loader_config.data.repo_id == "local/eval"
    assert plan.loader_config.data.assets == _Assets(assets_dir="/assets", asset_id="local/train")
    assert plan.loader_config.data.use_depth_image
    assert config.data.repo_id == "local/train"
    assert config.data.assets.asset_id is None
    assert plan.source_repo_id == "local/eval"
    assert plan.metric_name == "heldout_offline_imitation_reward"
    assert plan.reward_log_key == "eval/heldout_offline_imitation_reward"
    assert plan.loss_log_key == "eval/heldout_offline_imitation_loss"
    assert plan.uses_heldout_data


@pytest.mark.parametrize(
    ("eval_repo_id", "message"),
    [("", "non-empty string"), ("local/train", "must differ")],
)
def test_heldout_checkpoint_evaluation_rejects_empty_or_training_repo(eval_repo_id: str, message: str) -> None:
    config = _Config(data=_Data(repo_id="local/train"), checkpoint_eval_repo_id=eval_repo_id)

    with pytest.raises(ValueError, match=message):
        resolve_checkpoint_evaluation(config)
