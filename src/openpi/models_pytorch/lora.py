"""Small, dependency-free LoRA utilities for the PyTorch pi0/pi0.5 model."""

from __future__ import annotations

from collections.abc import Iterable
import math

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812


class LoRALinear(nn.Module):
    """A frozen linear layer with a trainable low-rank residual."""

    def __init__(self, base: nn.Linear, *, rank: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")

        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

        for parameter in self.base.parameters():
            parameter.requires_grad_(requires_grad=False)

        factory_kwargs = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, **factory_kwargs))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, **factory_kwargs))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    @property
    def weight(self) -> torch.Tensor:
        """Expose the base weight for existing dtype/device checks."""
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = self.base(inputs)
        residual = F.linear(F.linear(self.dropout(inputs), self.lora_a), self.lora_b)
        return base + residual * self.scaling


def _replace_linear_children(
    module: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    target_names: frozenset[str],
) -> list[str]:
    replaced: list[str] = []
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            continue
        if isinstance(child, nn.Linear) and name in target_names:
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
            replaced.append(name)
            continue
        replaced.extend(
            f"{name}.{nested_name}"
            for nested_name in _replace_linear_children(
                child,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                target_names=target_names,
            )
        )
    return replaced


def apply_pi0_lora(
    model: nn.Module,
    *,
    paligemma_rank: int,
    action_expert_rank: int,
    paligemma_alpha: float,
    action_expert_alpha: float,
    dropout: float = 0.0,
    train_action_heads: bool = True,
) -> list[str]:
    """Freeze pi0.5 and add LoRA to both language and action transformers."""
    for parameter in model.parameters():
        parameter.requires_grad_(requires_grad=False)

    target_names = frozenset({"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"})
    roots = model.paligemma_with_expert
    replaced = [
        f"paligemma.language_model.{name}"
        for name in _replace_linear_children(
            roots.paligemma.language_model,
            rank=paligemma_rank,
            alpha=paligemma_alpha,
            dropout=dropout,
            target_names=target_names,
        )
    ]
    replaced.extend(
        f"gemma_expert.{name}"
        for name in _replace_linear_children(
            roots.gemma_expert,
            rank=action_expert_rank,
            alpha=action_expert_alpha,
            dropout=dropout,
            target_names=target_names,
        )
    )
    if not replaced:
        raise RuntimeError("No pi0.5 transformer linear layers matched the LoRA targets")

    if train_action_heads:
        for head_name in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"):
            head = getattr(model, head_name, None)
            if head is not None:
                for parameter in head.parameters():
                    parameter.requires_grad_(requires_grad=True)

    return replaced


def trainable_named_parameters(model: nn.Module) -> Iterable[tuple[str, nn.Parameter]]:
    return ((name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad)


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return only adapter/action-head tensors, detached and safe to serialize."""
    return {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in trainable_named_parameters(model)
    }


def load_trainable_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    expected = {name for name, _ in trainable_named_parameters(model)}
    received = set(state)
    if missing := expected - received:
        raise ValueError(f"Adapter checkpoint is missing trainable tensors: {sorted(missing)[:8]}")
    if unexpected := received - expected:
        raise ValueError(f"Adapter checkpoint has unexpected tensors: {sorted(unexpected)[:8]}")
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(f"Unexpected adapter keys: {incompatible.unexpected_keys[:8]}")


def parameter_counts(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable
