import pytest

from openpi.policies import policy_config as _policy_config


def test_infer_checkpoint_asset_id_single(tmp_path):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    (assets_dir / "sim1").mkdir()

    assert _policy_config._infer_checkpoint_asset_id(assets_dir) == "sim1"


def test_infer_checkpoint_asset_id_none(tmp_path):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        _policy_config._infer_checkpoint_asset_id(assets_dir)


def test_infer_checkpoint_asset_id_multiple(tmp_path):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    (assets_dir / "a").mkdir()
    (assets_dir / "b").mkdir()

    with pytest.raises(ValueError):
        _policy_config._infer_checkpoint_asset_id(assets_dir)

