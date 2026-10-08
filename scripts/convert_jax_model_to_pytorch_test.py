import json
import types

import numpy as np
import orbax.checkpoint as ocp
import pytest
import safetensors.torch
import torch
import tyro

from examples import convert_jax_model_to_pytorch as conversion
from openpi.models import pi0_config


@pytest.mark.parametrize("pi05", [False, True])
@pytest.mark.parametrize("config_dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("precision", [None, "float32", "bfloat16"])
def test_conversion_preserves_requested_precision(tmp_path, monkeypatch, pi05, config_dtype, precision):
    """Exercise Orbax restore, conversion, serialization, and fp32 training reload.

    Replace only the large model and its architecture-specific weight slicing.
    Values deliberately contain bits that cannot be represented in bfloat16.
    """
    weight = np.array([[1.001, -0.1001], [0.123456789, 2.003]], dtype=np.float32)
    bias = np.array([0.1001, -1.001], dtype=np.float32)
    projection_names = ["action_in_proj", "action_out_proj"]
    projection_names += (
        ["time_mlp_in", "time_mlp_out"]
        if pi05
        else [
            "state_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
        ]
    )
    params = {"PaliGemma": {"backbone": weight, "expert": weight}}
    params.update({name: {"kernel": weight, "bias": bias} for name in projection_names})
    checkpoint_dir = tmp_path / "jax"
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(checkpoint_dir / "params", {"params": params})

    class SmallPI0(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.paligemma_with_expert = torch.nn.Module()
            for name in ("paligemma", "gemma_expert"):
                self.paligemma_with_expert.add_module(
                    name, torch.nn.Linear(2, 2, bias=False, dtype=getattr(torch, config.dtype))
                )
            for name in projection_names:
                self.add_module(name, torch.nn.Linear(2, 2))

    def slice_paligemma(params, config):
        return {"paligemma_with_expert.paligemma.weight": torch.from_numpy(params["backbone"])}, params

    def slice_gemma(params, *args, **kwargs):
        return {"paligemma_with_expert.gemma_expert.weight": torch.from_numpy(params["expert"])}

    monkeypatch.setattr(conversion.openpi.models_pytorch.pi0_pytorch, "PI0Pytorch", SmallPI0)
    monkeypatch.setattr(conversion, "slice_paligemma_state_dict", slice_paligemma)
    monkeypatch.setattr(conversion, "slice_gemma_state_dict", slice_gemma)
    model_config = pi0_config.Pi0Config(dtype=config_dtype, pi05=pi05, pytorch_compile_mode=None)
    monkeypatch.setattr(conversion._config, "get_config", lambda _: types.SimpleNamespace(model=model_config))  # noqa: SLF001

    output_dir = tmp_path / "pytorch"
    args = [
        "--checkpoint_dir",
        str(checkpoint_dir),
        "--config_name",
        "test_precision",
        "--output_path",
        str(output_dir),
    ]
    if precision is not None:
        args.extend(["--precision", precision])
    tyro.cli(conversion.main, args=args)

    output_precision = precision or "float32"
    output_dtype = getattr(torch, output_precision)
    expected = {
        "paligemma_with_expert.paligemma.weight": torch.from_numpy(weight),
        "paligemma_with_expert.gemma_expert.weight": torch.from_numpy(weight),
    }
    for name in projection_names:
        expected[f"{name}.weight"] = torch.from_numpy(weight).T
        expected[f"{name}.bias"] = torch.from_numpy(bias)
    saved = safetensors.torch.load_file(output_dir / "model.safetensors")
    assert saved.keys() == expected.keys()
    for name, value in expected.items():
        torch.testing.assert_close(saved[name], value.to(output_dtype), rtol=0, atol=0)
    assert json.loads((output_dir / "config.json").read_text())["precision"] == output_precision
    assert model_config.dtype == config_dtype

    # The training loader copies into fp32 parameters; this cannot undo bf16 rounding.
    training_model = SmallPI0(pi0_config.Pi0Config(dtype="float32"))
    safetensors.torch.load_model(training_model, output_dir / "model.safetensors")
    for name, value in training_model.state_dict().items():
        torch.testing.assert_close(value, expected[name].to(output_dtype).float(), rtol=0, atol=0)
