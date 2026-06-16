"""PyTorch LoRA implementation for pi0.5 training."""

import logging
import math
from dataclasses import dataclass
from dataclasses import field
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapters."""

    # LoRA rank - lower rank = fewer parameters but less capacity
    rank: int
    # LoRA scaling factor: output = base_output + lora_output * (alpha / rank)
    alpha: float = 1.0
    # Enable rank-stabilized LoRA (rsLoRA): uses alpha / sqrt(rank) instead
    # Reference: https://arxiv.org/pdf/2312.03732
    rslora: bool = False
    # Dropout rate for LoRA layers (applied to input before LoRA projection)
    dropout: float = 0.0
    # Initialization standard deviation for LoRA A matrix
    init_std: float = 0.01

    @property
    def scaling_value(self) -> float:
        """Get the scaling value based on whether rsLoRA is enabled."""
        return self.alpha / math.sqrt(self.rank) if self.rslora else self.alpha / self.rank


@dataclass
class LoRATrainingConfig:
    """Training-specific LoRA configuration for PyTorch training."""

    # Whether to enable LoRA training
    enabled: bool = False

    # LoRA rank for attention modules (q_proj, k_proj, v_proj, o_proj)
    attn_rank: int = 16
    # LoRA rank for FFN modules (gate_proj, up_proj, down_proj)
    ffn_rank: int = 16

    # LoRA alpha for attention modules
    attn_alpha: float = 16.0
    # LoRA alpha for FFN modules
    ffn_alpha: float = 16.0

    # Use rank-stabilized LoRA
    use_rslora: bool = False
    # Dropout rate for LoRA layers
    dropout: float = 0.0

    # Which modules to apply LoRA to
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj",      # FFN
    ])

    # Which parts of the model to apply LoRA to
    apply_to: Literal["all", "paligemma_only", "expert_only", "paligemma_attn", "expert_attn"] = "all"

    # Whether to train the vision encoder (SigLIP)
    train_vision_encoder: bool = True

    # Whether to also train non-LoRA layers (e.g., action projections, time MLPs)
    train_non_lora_layers: bool = True
    # Specific non-LoRA module names to train (if train_non_lora_layers is True)
    trainable_modules: list[str] = field(default_factory=lambda: [
        "action_in_proj", "action_out_proj",  # Action projections
        "time_mlp_in", "time_mlp_out",        # Time embeddings (pi05)
        "state_proj", "action_time_mlp_in", "action_time_mlp_out",  # pi0
    ])

    def get_attn_config(self) -> LoRAConfig:
        """Get LoRA config for attention modules."""
        return LoRAConfig(
            rank=self.attn_rank,
            alpha=self.attn_alpha,
            rslora=self.use_rslora,
            dropout=self.dropout,
        )

    def get_ffn_config(self) -> LoRAConfig:
        """Get LoRA config for FFN modules."""
        return LoRAConfig(
            rank=self.ffn_rank,
            alpha=self.ffn_alpha,
            rslora=self.use_rslora,
            dropout=self.dropout,
        )

    def get_lora_configs(self) -> dict[str, LoRAConfig]:
        """Get a dictionary of LoRA configs for apply_lora_to_model."""
        return {
            "attn": self.get_attn_config(),
            "ffn": self.get_ffn_config(),
        }


class LoRALinear(nn.Module):
    """Linear layer with LoRA support."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_config: LoRAConfig | None = None,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_config = lora_config

        # Original linear layer
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

        # Initialize original weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        # LoRA parameters
        if lora_config is not None:
            self.lora_a = nn.Parameter(
                torch.empty(lora_config.rank, in_features, device=device, dtype=dtype)
            )
            self.lora_b = nn.Parameter(
                torch.zeros(out_features, lora_config.rank, device=device, dtype=dtype)
            )
            # Initialize LoRA A with normal distribution
            nn.init.normal_(self.lora_a, std=0.01)
            # LoRA B is initialized to zeros (so initial LoRA contribution is 0)

            if lora_config.dropout > 0:
                self.lora_dropout = nn.Dropout(p=lora_config.dropout)
            else:
                self.lora_dropout = nn.Identity()
        else:
            self.register_parameter("lora_a", None)
            self.register_parameter("lora_b", None)
            self.lora_dropout = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original linear transformation
        result = F.linear(x, self.weight, self.bias)

        # Add LoRA contribution if enabled
        if self.lora_config is not None and self.lora_a is not None and self.lora_b is not None:
            lora_input = self.lora_dropout(x)
            # LoRA: x @ A.T @ B.T = x @ (B @ A).T
            lora_out = F.linear(F.linear(lora_input, self.lora_a), self.lora_b)
            result = result + lora_out * self.lora_config.scaling_value

        return result

    def merge_lora_weights(self):
        """Merge LoRA weights into the original weights (for inference)."""
        if self.lora_config is not None and self.lora_a is not None and self.lora_b is not None:
            with torch.no_grad():
                # W' = W + B @ A * scaling
                self.weight.data += (self.lora_b @ self.lora_a) * self.lora_config.scaling_value
            # Clear LoRA parameters
            self.lora_a = None
            self.lora_b = None
            self.lora_config = None

    def extra_repr(self) -> str:
        lora_info = ""
        if self.lora_config is not None:
            lora_info = f", lora_rank={self.lora_config.rank}, lora_alpha={self.lora_config.alpha}"
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}{lora_info}"


def apply_lora_to_linear(
    linear: nn.Linear,
    lora_config: LoRAConfig,
) -> LoRALinear:
    """Convert an existing nn.Linear layer to LoRALinear.

    This preserves the original weights and adds LoRA parameters.
    """
    lora_linear = LoRALinear(
        in_features=linear.in_features,
        out_features=linear.out_features,
        lora_config=lora_config,
        bias=linear.bias is not None,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )
    # Copy original weights
    lora_linear.weight.data = linear.weight.data.clone()
    if linear.bias is not None:
        lora_linear.bias.data = linear.bias.data.clone()

    return lora_linear


def get_lora_config_from_variant(variant: str) -> dict[str, LoRAConfig] | None:
    """Get LoRA config based on variant name.

    Follows the JAX implementation convention where variant names with '_lora'
    suffix indicate LoRA-enabled models.
    """
    if variant == "gemma_2b_lora":
        return {
            "attn": LoRAConfig(rank=16, alpha=16.0),
            "ffn": LoRAConfig(rank=16, alpha=16.0),
        }
    if variant == "gemma_300m_lora":
        return {
            "attn": LoRAConfig(rank=32, alpha=32.0),
            "ffn": LoRAConfig(rank=32, alpha=32.0),
        }
    return None


def apply_lora_to_model(
    model: nn.Module,
    lora_configs: dict[str, LoRAConfig],
    target_modules: list[str] | None = None,
) -> None:
    """Apply LoRA to specified modules in a model.

    Args:
        model: The model to apply LoRA to.
        lora_configs: Dictionary mapping module type ('attn' or 'ffn') to LoRAConfig.
        target_modules: List of module names to apply LoRA to. If None, applies to all
                       attention and FFN modules based on common naming patterns.
    """
    if target_modules is None:
        # Default target modules for Gemma-style models
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj",      # FFN (MLP)
        ]

    attn_modules = {"q_proj", "k_proj", "v_proj", "o_proj"}
    ffn_modules = {"gate_proj", "up_proj", "down_proj"}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if this module should have LoRA applied
            module_name = name.split(".")[-1]
            if module_name not in target_modules:
                continue

            # Determine which LoRA config to use
            if module_name in attn_modules and "attn" in lora_configs:
                lora_config = lora_configs["attn"]
            elif module_name in ffn_modules and "ffn" in lora_configs:
                lora_config = lora_configs["ffn"]
            else:
                continue

            # Get parent module and attribute name
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent_name, attr_name = parts
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name

            # Replace with LoRA linear
            lora_linear = apply_lora_to_linear(module, lora_config)
            setattr(parent, attr_name, lora_linear)


def freeze_non_lora_params(model: nn.Module) -> tuple[int, int]:
    """Freeze all parameters except LoRA parameters.

    Args:
        model: The model to freeze.

    Returns:
        Tuple of (frozen_params_count, trainable_params_count).
    """
    frozen_count = 0
    trainable_count = 0

    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False
            frozen_count += param.numel()

    return frozen_count, trainable_count


def apply_lora_to_pi0_pytorch(
    model: nn.Module,
    lora_config: LoRATrainingConfig,
) -> tuple[int, int]:
    """Apply LoRA to PI0Pytorch model based on training config."""
    if not lora_config.enabled:
        logging.info("LoRA is disabled, skipping LoRA application")
        return 0, sum(p.numel() for p in model.parameters())

    lora_configs = lora_config.get_lora_configs()
    target_modules = lora_config.target_modules

    # Determine which sub-models to apply LoRA to
    apply_to_paligemma = lora_config.apply_to in ["all", "paligemma_only", "paligemma_attn"]
    apply_to_expert = lora_config.apply_to in ["all", "expert_only", "expert_attn"]

    # Adjust target modules based on apply_to setting
    if lora_config.apply_to in ["paligemma_attn", "expert_attn"]:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    lora_applied_count = 0

    # Helper function to check if a module should have LoRA applied
    def should_apply_lora(full_name: str, module_name: str) -> bool:
        nonlocal lora_applied_count

        if module_name not in target_modules:
            return False

        # Vision tower is handled separately (not via LoRA, but full fine-tuning if enabled)
        if "vision_tower" in full_name or "vision_model" in full_name:
            return False

        # Check if this is in paligemma language model or expert based on path
        is_paligemma = "paligemma" in full_name and "language_model" in full_name
        is_expert = "gemma_expert" in full_name

        if is_paligemma and apply_to_paligemma:
            return True
        if is_expert and apply_to_expert:
            return True
        return False

    # Determine which LoRA config to use based on module type
    attn_modules = {"q_proj", "k_proj", "v_proj", "o_proj"}
    ffn_modules = {"gate_proj", "up_proj", "down_proj"}

    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            module_name = name.split(".")[-1]

            if not should_apply_lora(name, module_name):
                continue

            # Determine which LoRA config to use
            if module_name in attn_modules and "attn" in lora_configs:
                lora_cfg = lora_configs["attn"]
            elif module_name in ffn_modules and "ffn" in lora_configs:
                lora_cfg = lora_configs["ffn"]
            else:
                continue

            # Get parent module and attribute name
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent_name, attr_name = parts
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name

            # Replace with LoRA linear
            lora_linear = apply_lora_to_linear(module, lora_cfg)
            setattr(parent, attr_name, lora_linear)
            lora_applied_count += 1

    logging.info(f"Applied LoRA to {lora_applied_count} linear layers")

    # Freeze non-LoRA parameters
    frozen_count, trainable_count = freeze_for_lora_training(model, lora_config)

    return frozen_count, trainable_count


def freeze_for_lora_training(
    model: nn.Module,
    lora_config: LoRATrainingConfig,
) -> tuple[int, int]:
    """Freeze parameters for LoRA training."""
    frozen_count = 0
    trainable_count = 0
    vision_trainable = 0

    trainable_modules = set(lora_config.trainable_modules) if lora_config.train_non_lora_layers else set()

    for name, param in model.named_parameters():
        # Check if this is a LoRA parameter
        is_lora_param = "lora_" in name

        # Check if this is a vision encoder parameter
        is_vision_param = "vision_tower" in name or "vision_model" in name

        # Check if this is a trainable non-LoRA module
        is_trainable_module = False
        if lora_config.train_non_lora_layers:
            for trainable_name in trainable_modules:
                if trainable_name in name:
                    is_trainable_module = True
                    break

        # Determine if parameter should be trainable
        should_train = False
        if is_lora_param:
            should_train = True
        elif is_vision_param:
            # Vision encoder: train if train_vision_encoder is True
            should_train = lora_config.train_vision_encoder
            if should_train:
                vision_trainable += param.numel()
        elif is_trainable_module:
            should_train = True

        if should_train:
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False
            frozen_count += param.numel()

    logging.info(f"LoRA training: {trainable_count:,} trainable params, {frozen_count:,} frozen params")
    logging.info(f"Trainable ratio: {trainable_count / (trainable_count + frozen_count) * 100:.4f}%")
    if lora_config.train_vision_encoder:
        logging.info(f"Vision encoder (SigLIP) is TRAINABLE: {vision_trainable:,} params (JAX-consistent)")
    else:
        logging.info("Vision encoder (SigLIP) is FROZEN (memory-efficient mode)")

    return frozen_count, trainable_count