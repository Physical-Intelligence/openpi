from __future__ import annotations

import torch
from torch import nn

from openpi.models_pytorch import lora


class _Transformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.down_proj = nn.Linear(8, 8)


class _ToyPi0(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.paligemma_with_expert = nn.Module()
        self.paligemma_with_expert.paligemma = nn.Module()
        self.paligemma_with_expert.paligemma.language_model = _Transformer()
        self.paligemma_with_expert.gemma_expert = _Transformer()
        self.action_in_proj = nn.Linear(8, 8)
        self.action_out_proj = nn.Linear(8, 8)
        self.time_mlp_in = nn.Linear(8, 8)
        self.time_mlp_out = nn.Linear(8, 8)


def test_lora_linear_starts_as_exact_base() -> None:
    base = nn.Linear(8, 4)
    inputs = torch.randn(3, 8)
    expected = base(inputs)

    adapted = lora.LoRALinear(base, rank=2, alpha=2.0)

    torch.testing.assert_close(adapted(inputs), expected)
    assert adapted.base.weight.requires_grad is False
    assert adapted.lora_a.requires_grad is True
    assert adapted.lora_b.requires_grad is True


def test_apply_pi0_lora_freezes_base_and_round_trips_trainable_state() -> None:
    model = _ToyPi0()
    replaced = lora.apply_pi0_lora(
        model,
        paligemma_rank=2,
        action_expert_rank=4,
        paligemma_alpha=2.0,
        action_expert_alpha=4.0,
    )

    assert len(replaced) == 4
    assert model.paligemma_with_expert.paligemma.language_model.q_proj.base.weight.requires_grad is False
    assert model.action_out_proj.weight.requires_grad is True
    total, trainable = lora.parameter_counts(model)
    assert 0 < trainable < total

    for _, parameter in lora.trainable_named_parameters(model):
        parameter.data.uniform_(-0.1, 0.1)
    state = lora.trainable_state_dict(model)

    restored = _ToyPi0()
    lora.apply_pi0_lora(
        restored,
        paligemma_rank=2,
        action_expert_rank=4,
        paligemma_alpha=2.0,
        action_expert_alpha=4.0,
    )
    lora.load_trainable_state_dict(restored, state)

    restored_state = dict(lora.trainable_named_parameters(restored))
    for name, tensor in state.items():
        torch.testing.assert_close(restored_state[name], tensor)
