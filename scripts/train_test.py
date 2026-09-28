import dataclasses
import os
import pathlib

import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.nnx_utils as nnx_utils
from openpi.training import config as _config

from . import train


@pytest.mark.parametrize("config_name", ["debug"])
def test_train(tmp_path: pathlib.Path, config_name: str):
    config = dataclasses.replace(
        _config._CONFIGS_DICT[config_name],  # noqa: SLF001
        batch_size=2,
        checkpoint_base_dir=str(tmp_path / "checkpoint"),
        exp_name="test",
        overwrite=False,
        resume=False,
        num_train_steps=2,
        log_interval=1,
    )
    train.main(config)

    # test resuming
    config = dataclasses.replace(config, resume=True, num_train_steps=4)
    train.main(config)


def test_train_with_freeze_filter_and_ema(tmp_path: pathlib.Path):
    # Freeze the LLM backbone, with EMA enabled for the trainable params.
    config = dataclasses.replace(
        _config._CONFIGS_DICT["debug"],  # noqa: SLF001
        batch_size=2,
        checkpoint_base_dir=str(tmp_path / "checkpoint"),
        exp_name="test",
        overwrite=False,
        resume=False,
        num_train_steps=2,
        log_interval=1,
        freeze_filter=nnx.All(nnx_utils.PathRegex(".*llm.*"), nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*"))),
        ema_decay=0.99,
    )
    train.main(config)
    params_before = traverse_util.flatten_dict(
        _model.restore_params(config.checkpoint_dir / "1" / "params", restore_type=np.ndarray)
    )

    # test resuming
    config = dataclasses.replace(config, resume=True, num_train_steps=4)
    train.main(config)
    params_after = traverse_util.flatten_dict(
        _model.restore_params(config.checkpoint_dir / "3" / "params", restore_type=np.ndarray)
    )

    # The saved params contain the full model, including the frozen params that EMA does not track.
    assert params_before.keys() == params_after.keys()

    frozen_paths = [path for path in params_before if config.freeze_filter(path, None)]
    trainable_paths = [path for path in params_before if not config.freeze_filter(path, None)]
    assert frozen_paths
    assert trainable_paths

    # Frozen params are saved exactly as they are, while trainable params are updated by training.
    for path in frozen_paths:
        np.testing.assert_array_equal(params_before[path], params_after[path])
    assert any(not np.array_equal(params_before[path], params_after[path]) for path in trainable_paths)
