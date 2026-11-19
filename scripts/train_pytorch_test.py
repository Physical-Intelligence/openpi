import dataclasses
import os
import types

import pytest
import torch

os.environ["JAX_PLATFORMS"] = "cpu"

import openpi.models_pytorch.gemma_pytorch as _gemma_pytorch
import openpi.models_pytorch.pi0_pytorch as _pi0_pytorch
from openpi.training import config as _config
import openpi.training.optimizer as _optimizer

from . import train_pytorch

_PI0Pytorch = _pi0_pytorch.PI0Pytorch
_WIDTH = 32


def _transformers_replace_installed() -> bool:
    try:
        from transformers.models.siglip import check
    except ImportError:
        return False
    return check.check_whether_transformers_replace_is_installed_correctly()


class _StandInModel(torch.nn.Module):
    """Takes the place of PI0Pytorch in train_loop: one linear layer whose weights all start at 1.0."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = torch.nn.Linear(_WIDTH, _WIDTH, bias=False)
        torch.nn.init.ones_(self.layer.weight)
        # Store the weights the way PI0Pytorch does for `config.dtype`.
        _gemma_pytorch.PaliGemmaWithExpertModel.to_bfloat16_for_selected_params(self, config.dtype)
        self.initial_weight = self.layer.weight.detach().clone()
        self.autocast_dtypes = []

    def forward(self, observation, actions):
        device_type = self.layer.weight.device.type
        enabled = torch.is_autocast_enabled(device_type)
        self.autocast_dtypes.append(torch.get_autocast_dtype(device_type) if enabled else None)
        prediction = self.layer(observation["state"].to(self.layer.weight.dtype))
        return torch.nn.functional.mse_loss(prediction.float(), actions, reduction="none")


def _train_one_step(tmp_path, monkeypatch, precision: str) -> _StandInModel:
    """Runs one step of train_loop at learning rate 1e-5, with the stand-in model and a single fixed batch."""
    models = []

    def make_model(config):
        models.append(_StandInModel(config))
        return models[-1]

    # Every element of the gradient is positive, so the first AdamW step moves every weight down by about 1e-5.
    batch = ({"state": torch.ones(4, _WIDTH)}, torch.zeros(4, _WIDTH))
    data_config = types.SimpleNamespace(norm_stats=None, asset_id=None)
    monkeypatch.setattr(_pi0_pytorch, "PI0Pytorch", make_model)
    monkeypatch.setattr(train_pytorch, "build_datasets", lambda config: ([batch], data_config))

    base = _config.get_config("debug")
    config = dataclasses.replace(
        base,
        model=dataclasses.replace(base.model),  # train_loop sets the dtype on this object.
        pytorch_training_precision=precision,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=0, peak_lr=1e-5, decay_steps=1, decay_lr=1e-5),
        checkpoint_base_dir=str(tmp_path),
        num_train_steps=1,
    )
    train_pytorch.train_loop(config)
    (model,) = models
    return model


def test_bfloat16_weights_drop_small_updates(tmp_path, monkeypatch):
    model = _train_one_step(tmp_path, monkeypatch, "bfloat16")

    assert model.layer.weight.dtype == torch.bfloat16
    # The bfloat16 numbers next to 1.0 are 2**-8 below and 2**-7 above it, so a step of 1e-5 rounds back to 1.0.
    assert torch.equal(model.layer.weight.detach().cpu(), model.initial_weight)


@pytest.mark.parametrize(
    ("precision", "autocast_dtype"), [("mixed", torch.bfloat16), ("float32", None)], ids=["mixed", "float32"]
)
def test_float32_weights_keep_small_updates(tmp_path, monkeypatch, precision, autocast_dtype):
    model = _train_one_step(tmp_path, monkeypatch, precision)

    assert model.autocast_dtypes == [autocast_dtype]
    assert model.layer.weight.dtype == torch.float32
    assert (model.layer.weight.detach().cpu() != model.initial_weight).all()


@pytest.mark.skipif(
    not _transformers_replace_installed(), reason="transformers_replace is not installed into site-packages"
)
@pytest.mark.parametrize("config_name", ["pi0_libero", "pi05_libero"])
def test_mixed_precision_stores_all_trainable_weights_in_float32(tmp_path, monkeypatch, config_name):
    weight_dtype = _train_one_step(tmp_path, monkeypatch, "mixed").config.dtype
    model_config = dataclasses.replace(_config.get_config(config_name).model, dtype=weight_dtype)

    with torch.device("meta"):
        model = _PI0Pytorch(model_config)

    assert {p.dtype for p in model.parameters() if p.requires_grad} == {torch.float32}
