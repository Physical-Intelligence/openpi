#!/usr/bin/env python3
"""
Load a JAX model and print all parameter keys, with optional conversion to PyTorch.

This script loads a JAX model checkpoint using orbax and can either:
1. Print out all the parameter keys in a hierarchical structure for inspection
2. Convert the JAX model to PyTorch format using our PI0Pytorch model

Usage:
    # Just inspect keys:
    python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /path/to/checkpoint --inspect_only
    python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /path/to/checkpoint --inspect_only

    # Convert to PyTorch:
    python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /path/to/checkpoint --output_path /path/to/output
    python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /path/to/checkpoint --output_path /path/to/output

Example:
    # pi0_droid
    python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi0_droid --output_path /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi0_droid_pytorch

    # pi0_aloha_sim
    python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi0_aloha_sim --output_path /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi0_aloha_sim_pytorch

    # pi05_droid
    python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi05_droid --output_path /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch
"""

import dataclasses
import json
import logging
import math
import os
import pathlib
import shutil
from typing import Literal

from flax.nnx import traversals
import numpy as np
import orbax.checkpoint as ocp
import safetensors
import torch
import tyro

import openpi.models.gemma
import openpi.models.model
import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
from openpi.training import utils
import openpi.training.config as _config

logger = logging.getLogger(__name__)


def _has_lora(model_config: openpi.models.pi0_config.Pi0Config) -> bool:
    """Return True if either expert config declares LoRA adapters."""
    paligemma_cfg = openpi.models.gemma.get_config(model_config.paligemma_variant)
    action_expert_cfg = openpi.models.gemma.get_config(model_config.action_expert_variant)
    return bool(getattr(paligemma_cfg, "lora_configs", None)) or bool(getattr(action_expert_cfg, "lora_configs", None))


def _lora_scale(config) -> float:
    """Replicate ``openpi.models.lora.LoRAConfig.scaling_value`` without instantiating it."""
    if config is None:
        return 1.0
    rank = float(config.rank)
    alpha = float(config.alpha)
    if bool(getattr(config, "rslora", False)):
        return alpha / math.sqrt(rank)
    return alpha / rank


def _merge_einsum_lora(
    state_dict: dict,
    *,
    base_key: str,
    lora_a_key: str,
    lora_b_key: str,
    einsum_expr: str,
    scale: float,
) -> None:
    """Merge a pair of LoRA adapters into a base einsum weight in-place.

    The merge is computed in float32 to avoid bfloat16 cancellation; the resulting
    base weight is stored as float32 so the downstream slicing keeps full precision.
    """
    if lora_a_key not in state_dict or lora_b_key not in state_dict:
        return
    base = np.asarray(state_dict[base_key], dtype=np.float32)
    lora_a = np.asarray(state_dict.pop(lora_a_key), dtype=np.float32)
    lora_b = np.asarray(state_dict.pop(lora_b_key), dtype=np.float32)
    delta = np.einsum(einsum_expr, lora_a, lora_b, optimize=True)
    state_dict[base_key] = base + delta * scale


def _merge_attn_vec_lora(
    state_dict: dict,
    *,
    base_key: str,
    lora_a_key: str,
    lora_b_key: str,
    scale: float,
) -> None:
    """Merge attn_vec_einsum LoRA with the openpi-specific sum-over-N correction.

    In ``openpi.models.gemma.Attention``, the post-attention projection runs as
    ``out_einsum("BTNH,NHD->BTD", encoded)``. ``openpi.models.lora.Einsum`` then
    builds the LoRA path as two einsums::

        lora = einsum("BTNH,NHL->BTL", x, lora_a)      # sums over N and H
        lora = einsum("BTL,NLD->BTD", lora, lora_b)    # also sums over N

    Because the second einsum sums over the head dimension ``N`` (it is present
    in ``lora_b`` but absent from the output), the equivalent merged delta is::

        delta[n, h, d] = sum_l lora_a[n, h, l] * (sum_{n'} lora_b[n', l, d])

    i.e. ``lora_b`` is summed over ``N`` first. A standard per-head outer
    product would not reproduce the runtime forward pass.
    """
    if lora_a_key not in state_dict or lora_b_key not in state_dict:
        return
    base = np.asarray(state_dict[base_key], dtype=np.float32)
    lora_a = np.asarray(state_dict.pop(lora_a_key), dtype=np.float32)
    lora_b = np.asarray(state_dict.pop(lora_b_key), dtype=np.float32)
    # shapes: base (L, N, H, D), lora_a (L, N, H, rank), lora_b (L, N, rank, D)
    lora_b_sum_n = np.sum(lora_b, axis=1)  # (L, rank, D)
    delta = np.einsum("lnhr,lrd->lnhd", lora_a, lora_b_sum_n, optimize=True)
    state_dict[base_key] = base + delta * scale


def _merge_mlp_linear_lora(
    state_dict: dict,
    *,
    base_key: str,
    lora_a_key: str,
    lora_b_key: str,
    scale: float,
) -> None:
    """Merge LoRA into the MLP ``linear`` weight (shape ``(L, hidden, features)``)."""
    if lora_a_key not in state_dict or lora_b_key not in state_dict:
        return
    base = np.asarray(state_dict[base_key], dtype=np.float32)
    lora_a = np.asarray(state_dict.pop(lora_a_key), dtype=np.float32)
    lora_b = np.asarray(state_dict.pop(lora_b_key), dtype=np.float32)
    delta = np.einsum("lhr,lrf->lhf", lora_a, lora_b, optimize=True)
    state_dict[base_key] = base + delta * scale


def merge_lora_into_base(
    flat_state_dict: dict,
    model_config: openpi.models.pi0_config.Pi0Config,
) -> dict:
    """Merge any LoRA adapter weights in ``flat_state_dict`` into the base weights.

    The flattened JAX checkpoint contains both base parameters (``.../w``,
    ``.../gating_einsum``, ``.../linear``) and LoRA adapters (``.../lora_a``,
    ``.../lora_b``, ``.../gating_einsum_lora_a``, ...). After this call, the
    LoRA adapter keys are removed and the corresponding base keys contain
    ``base + delta * scaling`` so the downstream slicing path produces a
    PyTorch checkpoint that is numerically equivalent to the JAX runtime.

    Two openpi-specific behaviors are reproduced here:

    1. ``attn_vec_einsum`` uses ``sum_N(lora_b)`` (see ``_merge_attn_vec_lora``).
    2. ``openpi.models.lora.FeedForward._dot`` adds the LoRA delta to the
       MLP weights *without* applying the alpha/rank scaling factor, so the
       merged MLP weights must omit it (``scale=1.0``).

    This function is a no-op for non-LoRA configs (their gemma configs have
    empty ``lora_configs`` dicts), so it is safe to call unconditionally.
    """
    paligemma_cfg = openpi.models.gemma.get_config(model_config.paligemma_variant)
    action_expert_cfg = openpi.models.gemma.get_config(model_config.action_expert_variant)

    # Match the suffix convention used by ``slice_paligemma_state_dict``: some
    # checkpoints keep an extra ``/value`` leaf, others do not.
    suffix = "/value" if "img/embedding/kernel/value" in flat_state_dict else ""

    def _merge_attention(prefix: str, gemma_cfg) -> None:
        lora_configs = getattr(gemma_cfg, "lora_configs", None) or {}
        attn_cfg = lora_configs.get("attn")
        if attn_cfg is None:
            return
        scale = _lora_scale(attn_cfg)
        # Combined QKV: shape (L, 3, num_heads, width, head_dim) when num_heads == num_kv_heads.
        qkv_key = f"{prefix}/qkv_einsum/w{suffix}"
        if qkv_key in flat_state_dict:
            _merge_einsum_lora(
                flat_state_dict,
                base_key=qkv_key,
                lora_a_key=f"{prefix}/qkv_einsum/lora_a{suffix}",
                lora_b_key=f"{prefix}/qkv_einsum/lora_b{suffix}",
                einsum_expr="lqndr,lqnrh->lqndh",
                scale=scale,
            )
        else:
            # Separate Q and KV einsums.
            _merge_einsum_lora(
                flat_state_dict,
                base_key=f"{prefix}/q_einsum/w{suffix}",
                lora_a_key=f"{prefix}/q_einsum/lora_a{suffix}",
                lora_b_key=f"{prefix}/q_einsum/lora_b{suffix}",
                einsum_expr="lndr,lnrh->lndh",
                scale=scale,
            )
            _merge_einsum_lora(
                flat_state_dict,
                base_key=f"{prefix}/kv_einsum/w{suffix}",
                lora_a_key=f"{prefix}/kv_einsum/lora_a{suffix}",
                lora_b_key=f"{prefix}/kv_einsum/lora_b{suffix}",
                einsum_expr="labdr,labrh->labdh",
                scale=scale,
            )
        _merge_attn_vec_lora(
            flat_state_dict,
            base_key=f"{prefix}/attn_vec_einsum/w{suffix}",
            lora_a_key=f"{prefix}/attn_vec_einsum/lora_a{suffix}",
            lora_b_key=f"{prefix}/attn_vec_einsum/lora_b{suffix}",
            scale=scale,
        )

    def _merge_mlp(prefix: str, gemma_cfg) -> None:
        lora_configs = getattr(gemma_cfg, "lora_configs", None) or {}
        ffn_cfg = lora_configs.get("ffn")
        if ffn_cfg is None:
            return
        # NOTE: openpi.models.lora.FeedForward._dot adds the LoRA delta without
        # applying alpha/rank scaling, so the merged weight must do the same.
        scale = 1.0
        _merge_einsum_lora(
            flat_state_dict,
            base_key=f"{prefix}/gating_einsum{suffix}",
            lora_a_key=f"{prefix}/gating_einsum_lora_a{suffix}",
            lora_b_key=f"{prefix}/gating_einsum_lora_b{suffix}",
            einsum_expr="lafr,larh->lafh",
            scale=scale,
        )
        _merge_mlp_linear_lora(
            flat_state_dict,
            base_key=f"{prefix}/linear{suffix}",
            lora_a_key=f"{prefix}/linear_lora_a{suffix}",
            lora_b_key=f"{prefix}/linear_lora_b{suffix}",
            scale=scale,
        )

    # PaliGemma expert (index 0).
    _merge_attention("llm/layers/attn", paligemma_cfg)
    _merge_mlp("llm/layers/mlp", paligemma_cfg)
    # Action expert MLP lives under mlp_1 in the flattened tree.
    _merge_mlp("llm/layers/mlp_1", action_expert_cfg)
    # Action expert attention keys carry the _1 suffix on the einsum leaf name.
    lora_configs = getattr(action_expert_cfg, "lora_configs", None) or {}
    expert_attn_cfg = lora_configs.get("attn")
    if expert_attn_cfg is not None:
        scale = _lora_scale(expert_attn_cfg)
        qkv_key = f"llm/layers/attn/qkv_einsum_1/w{suffix}"
        if qkv_key in flat_state_dict:
            _merge_einsum_lora(
                flat_state_dict,
                base_key=qkv_key,
                lora_a_key=f"llm/layers/attn/qkv_einsum_1/lora_a{suffix}",
                lora_b_key=f"llm/layers/attn/qkv_einsum_1/lora_b{suffix}",
                einsum_expr="lqndr,lqnrh->lqndh",
                scale=scale,
            )
        else:
            _merge_einsum_lora(
                flat_state_dict,
                base_key=f"llm/layers/attn/q_einsum_1/w{suffix}",
                lora_a_key=f"llm/layers/attn/q_einsum_1/lora_a{suffix}",
                lora_b_key=f"llm/layers/attn/q_einsum_1/lora_b{suffix}",
                einsum_expr="lndr,lnrh->lndh",
                scale=scale,
            )
            _merge_einsum_lora(
                flat_state_dict,
                base_key=f"llm/layers/attn/kv_einsum_1/w{suffix}",
                lora_a_key=f"llm/layers/attn/kv_einsum_1/lora_a{suffix}",
                lora_b_key=f"llm/layers/attn/kv_einsum_1/lora_b{suffix}",
                einsum_expr="labdr,labrh->labdh",
                scale=scale,
            )
        _merge_attn_vec_lora(
            flat_state_dict,
            base_key=f"llm/layers/attn/attn_vec_einsum_1/w{suffix}",
            lora_a_key=f"llm/layers/attn/attn_vec_einsum_1/lora_a{suffix}",
            lora_b_key=f"llm/layers/attn/attn_vec_einsum_1/lora_b{suffix}",
            scale=scale,
        )

    return flat_state_dict


def slice_paligemma_state_dict(state_dict, config):
    """Convert PaliGemma JAX parameters to PyTorch format."""
    suffix = "/value" if "img/embedding/kernel/value" in state_dict else ""

    # patch embeddings
    jax_key = f"img/embedding/kernel{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key).transpose(3, 2, 0, 1)

    jax_key = f"img/embedding/bias{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.bias"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # positional embeddings
    jax_key = f"img/pos_embedding{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.position_embedding.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key).reshape(-1, config.vision_config.hidden_size)

    # extract vision layers to be sliced at index 0. There are 27 layers in the base model.
    encoderblock_layernorm0_scale = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_0/scale{suffix}")
    encoderblock_layernorm0_bias = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_0/bias{suffix}")
    encoderblock_layernorm1_scale = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_1/scale{suffix}")
    encoderblock_layernorm1_bias = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_1/bias{suffix}")

    encoderblock_mlp_dense0_kernel = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel{suffix}")
    encoderblock_mlp_dense0_bias = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_0/bias{suffix}")
    encoderblock_mlp_dense1_kernel = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel{suffix}")
    encoderblock_mlp_dense1_bias = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_1/bias{suffix}")

    encoderblock_attention_0_key_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/kernel{suffix}"
    )
    encoderblock_attention_0_key_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/bias{suffix}"
    )
    encoderblock_attention_0_value_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/kernel{suffix}"
    )
    encoderblock_attention_0_value_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/bias{suffix}"
    )
    encoderblock_attention_0_query_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/kernel{suffix}"
    )
    encoderblock_attention_0_query_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/bias{suffix}"
    )
    encoderblock_attention_0_out_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/kernel{suffix}"
    )
    encoderblock_attention_0_out_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/bias{suffix}"
    )

    for i in range(config.vision_config.num_hidden_layers):
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm1.weight"
        ] = encoderblock_layernorm0_scale[i].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm1.bias"
        ] = encoderblock_layernorm0_bias[i]
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm2.weight"
        ] = encoderblock_layernorm1_scale[i].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm2.bias"
        ] = encoderblock_layernorm1_bias[i]
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.weight"
        ] = encoderblock_mlp_dense0_kernel[i].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.bias"
        ] = encoderblock_mlp_dense0_bias[i]
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.weight"
        ] = encoderblock_mlp_dense1_kernel[i].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.bias"
        ] = encoderblock_mlp_dense1_bias[i]
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.weight"
        ] = encoderblock_attention_0_key_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.bias"
        ] = encoderblock_attention_0_key_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.weight"
        ] = encoderblock_attention_0_value_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.bias"
        ] = encoderblock_attention_0_value_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.weight"
        ] = encoderblock_attention_0_query_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.bias"
        ] = encoderblock_attention_0_query_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.weight"
        ] = encoderblock_attention_0_out_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.bias"
        ] = encoderblock_attention_0_out_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)

    jax_key = f"img/Transformer/encoder_norm/scale{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key).transpose()

    jax_key = f"img/Transformer/encoder_norm/bias{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.bias"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # multimodal projector
    jax_key = f"img/head/kernel{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key).transpose()

    jax_key = f"img/head/bias{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.bias"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # text decoder (gemma)
    jax_key = f"llm/embedder/input_embedding{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # pop the einsum attention + mlp representations
    llm_attention_attn_vec_einsum = state_dict.pop(f"llm/layers/attn/attn_vec_einsum/w{suffix}")
    llm_attention_kv_einsum = state_dict.pop(f"llm/layers/attn/kv_einsum/w{suffix}")
    llm_attention_q_einsum = state_dict.pop(f"llm/layers/attn/q_einsum/w{suffix}")

    llm_mlp_gating_einsum = state_dict.pop(f"llm/layers/mlp/gating_einsum{suffix}")
    llm_mlp_linear = state_dict.pop(f"llm/layers/mlp/linear{suffix}")

    llm_input_layernorm = state_dict.pop(f"llm/layers/pre_attention_norm/scale{suffix}")
    llm_post_attention_layernorm = state_dict.pop(f"llm/layers/pre_ffw_norm/scale{suffix}")

    for i in range(config.text_config.num_hidden_layers):
        q_proj_weight_reshaped = (
            llm_attention_q_einsum[i]
            .transpose(0, 2, 1)
            .reshape(
                config.text_config.num_attention_heads * config.text_config.head_dim, config.text_config.hidden_size
            )
        )
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.q_proj.weight"] = (
            q_proj_weight_reshaped
        )

        k_proj_weight_reshaped = llm_attention_kv_einsum[i, 0, 0].transpose()
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.k_proj.weight"] = (
            k_proj_weight_reshaped
        )
        v_proj_weight_reshaped = llm_attention_kv_einsum[i, 1, 0].transpose()
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.v_proj.weight"] = (
            v_proj_weight_reshaped
        )

        o_proj_weight_reshaped = (
            llm_attention_attn_vec_einsum[i]
            .transpose(2, 0, 1)
            .reshape(
                config.text_config.num_attention_heads * config.text_config.head_dim, config.text_config.hidden_size
            )
        )
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.o_proj.weight"] = (
            o_proj_weight_reshaped
        )

        gate_proj_weight = llm_mlp_gating_einsum[i, 0]
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.mlp.gate_proj.weight"] = (
            gate_proj_weight.transpose()
        )
        up_proj_weight = llm_mlp_gating_einsum[i, 1]
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.mlp.up_proj.weight"] = (
            up_proj_weight.transpose()
        )
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.mlp.down_proj.weight"] = (
            llm_mlp_linear[i].transpose()
        )
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.input_layernorm.weight"] = (
            llm_input_layernorm[i]
        )
        state_dict[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.post_attention_layernorm.weight"
        ] = llm_post_attention_layernorm[i]

    jax_key = f"llm/final_norm/scale{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.language_model.norm.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    expert_dict = {}
    final_state_dict = {}

    # Expert-related keys to extract (including pi05 Dense layer parameters)
    expert_keys = [
        f"llm/final_norm_1/scale{suffix}",
        f"llm/final_norm_1/Dense_0/bias{suffix}",
        f"llm/final_norm_1/Dense_0/kernel{suffix}",
        f"llm/layers/attn/attn_vec_einsum_1/w{suffix}",
        f"llm/layers/attn/kv_einsum_1/w{suffix}",
        f"llm/layers/attn/q_einsum_1/w{suffix}",
        f"llm/layers/mlp_1/gating_einsum{suffix}",
        f"llm/layers/mlp_1/linear{suffix}",
        f"llm/layers/pre_attention_norm_1/scale{suffix}",
        f"llm/layers/pre_attention_norm_1/Dense_0/bias{suffix}",
        f"llm/layers/pre_attention_norm_1/Dense_0/kernel{suffix}",
        f"llm/layers/pre_ffw_norm_1/scale{suffix}",
        f"llm/layers/pre_ffw_norm_1/Dense_0/bias{suffix}",
        f"llm/layers/pre_ffw_norm_1/Dense_0/kernel{suffix}",
    ]

    for key, value in state_dict.items():
        if key not in expert_keys:
            final_state_dict[key] = torch.from_numpy(value)
        else:
            expert_dict[key] = value

    return final_state_dict, expert_dict


def slice_gemma_state_dict(state_dict, config, *, num_expert, checkpoint_dir, pi05):
    """Convert Gemma JAX parameters to PyTorch format."""
    # Add missing attributes to config if they don't exist
    if not hasattr(config, "vocab_size"):
        config.vocab_size = 257152  # PALIGEMMA_VOCAB_SIZE
    if not hasattr(config, "hidden_size"):
        config.hidden_size = config.width
    if not hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = config.depth
    if not hasattr(config, "num_attention_heads"):
        config.num_attention_heads = config.num_heads

    suffix = "/value" if f"llm/layers/attn/attn_vec_einsum_{num_expert}/w/value" in state_dict else ""

    llm_attention_attn_vec_einsum = state_dict.pop(f"llm/layers/attn/attn_vec_einsum_{num_expert}/w{suffix}")
    llm_attention_kv_einsum = state_dict.pop(f"llm/layers/attn/kv_einsum_{num_expert}/w{suffix}")
    llm_attention_q_einsum = state_dict.pop(f"llm/layers/attn/q_einsum_{num_expert}/w{suffix}")

    llm_mlp_gating_einsum = state_dict.pop(f"llm/layers/mlp_{num_expert}/gating_einsum{suffix}")
    llm_mlp_linear = state_dict.pop(f"llm/layers/mlp_{num_expert}/linear{suffix}")

    # Check if we have Dense layers (for pi05/adaptive normalization) or scale layers (for regular pi0)
    if pi05:
        # Pi05 with adaptive normalization
        llm_input_layernorm_bias = state_dict.pop(f"llm/layers/pre_attention_norm_{num_expert}/Dense_0/bias{suffix}")
        llm_post_attention_layernorm_bias = state_dict.pop(f"llm/layers/pre_ffw_norm_{num_expert}/Dense_0/bias{suffix}")
        llm_input_layernorm_kernel = state_dict.pop(
            f"llm/layers/pre_attention_norm_{num_expert}/Dense_0/kernel{suffix}"
        )
        llm_post_attention_layernorm_kernel = state_dict.pop(
            f"llm/layers/pre_ffw_norm_{num_expert}/Dense_0/kernel{suffix}"
        )
    else:
        # Regular pi0 with standard RMSNorm
        llm_input_layernorm = state_dict.pop(f"llm/layers/pre_attention_norm_{num_expert}/scale{suffix}")
        llm_post_attention_layernorm = state_dict.pop(f"llm/layers/pre_ffw_norm_{num_expert}/scale{suffix}")

    for i in range(config.num_hidden_layers):
        q_proj_weight_reshaped = (
            llm_attention_q_einsum[i]
            .transpose(0, 2, 1)
            .reshape(config.num_attention_heads * config.head_dim, config.hidden_size)
        )
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.q_proj.weight"] = (
            q_proj_weight_reshaped
        )

        k_proj_weight_reshaped = llm_attention_kv_einsum[i, 0, 0].transpose()
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.k_proj.weight"] = (
            k_proj_weight_reshaped
        )
        v_proj_weight_reshaped = llm_attention_kv_einsum[i, 1, 0].transpose()
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.v_proj.weight"] = (
            v_proj_weight_reshaped
        )

        o_proj_weight_reshaped = (
            llm_attention_attn_vec_einsum[i]
            .reshape(config.num_attention_heads * config.head_dim, config.hidden_size)
            .transpose(1, 0)
        )
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.o_proj.weight"] = (
            o_proj_weight_reshaped
        )

        gate_proj_weight = llm_mlp_gating_einsum[i, 0]
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.mlp.gate_proj.weight"] = (
            gate_proj_weight.transpose()
        )
        up_proj_weight = llm_mlp_gating_einsum[i, 1]
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.mlp.up_proj.weight"] = (
            up_proj_weight.transpose()
        )
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.mlp.down_proj.weight"] = llm_mlp_linear[
            i
        ].transpose()

        if pi05:
            # Pi05 with adaptive normalization - use Dense layer parameters directly
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.input_layernorm.dense.bias"] = (
                llm_input_layernorm_bias[i]
            )
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.post_attention_layernorm.dense.bias"] = (
                llm_post_attention_layernorm_bias[i]
            )
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.input_layernorm.dense.weight"] = (
                llm_input_layernorm_kernel[i].transpose()
            )
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.post_attention_layernorm.dense.weight"] = (
                llm_post_attention_layernorm_kernel[i].transpose()
            )
        else:
            # Regular pi0 with standard RMSNorm
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.input_layernorm.weight"] = (
                llm_input_layernorm[i]
            )
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.post_attention_layernorm.weight"] = (
                llm_post_attention_layernorm[i]
            )

    # Handle final norm layer
    if pi05:
        # Pi05 with adaptive normalization - use Dense layer parameters directly
        final_norm_bias = state_dict.pop(f"llm/final_norm_{num_expert}/Dense_0/bias{suffix}")
        final_norm_kernel = state_dict.pop(f"llm/final_norm_{num_expert}/Dense_0/kernel{suffix}")
        state_dict["paligemma_with_expert.gemma_expert.model.norm.dense.bias"] = final_norm_bias
        state_dict["paligemma_with_expert.gemma_expert.model.norm.dense.weight"] = final_norm_kernel.transpose()
    else:
        # Regular pi0 with standard RMSNorm
        state_dict["paligemma_with_expert.gemma_expert.model.norm.weight"] = state_dict.pop(
            f"llm/final_norm_{num_expert}/scale{suffix}"
        )

        # state_dict["paligemma_with_expert.gemma_expert.lm_head.weight"] = embedding_vector # weights are tied.

    final_state_dict = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            final_state_dict[key] = torch.from_numpy(value)
        else:
            final_state_dict[key] = value

    return final_state_dict


def slice_initial_orbax_checkpoint(checkpoint_dir: str, restore_precision: str | None = None):
    """Load and process params by restoring via JAX model loader first.
    This respects dtype conversions that occur during model restore.
    """
    # Use repository restore utility to load a pure dict of params (value suffix removed)
    params = openpi.models.model.restore_params(
        f"{checkpoint_dir}/params/", restore_type=np.ndarray, dtype=restore_precision
    )

    return {"paligemma_params": traversals.flatten_mapping(params["PaliGemma"], sep="/"), "projection_params": params}


def load_jax_model_and_print_keys(checkpoint_dir: str):
    """
    Load JAX model from checkpoint and print all parameter keys.

    Args:
        checkpoint_dir: Path to the checkpoint directory
    """
    checkpoint_dir = os.path.abspath(checkpoint_dir) if not checkpoint_dir.startswith("gs://") else checkpoint_dir
    # Initialize checkpointer
    checkpointer = ocp.PyTreeCheckpointer()
    metadata = checkpointer.metadata(f"{checkpoint_dir}/params")
    print(utils.array_tree_to_info(metadata))


def convert_pi0_checkpoint(
    checkpoint_dir: str, precision: str, output_path: str, model_config: openpi.models.pi0_config.Pi0Config
):
    """
    Convert PI0 JAX checkpoint to PyTorch format.

    Args:
        checkpoint_dir: Path to the JAX checkpoint
        precision: Model precision (float32, bfloat16)
        output_path: Path to save the converted PyTorch model
        model_config: Model config
    """
    print(f"Converting PI0 checkpoint from {checkpoint_dir} to {output_path}")
    print(f"Model config: {model_config}")

    if precision not in ("float32", "bfloat16"):
        raise ValueError(f"Invalid precision: {precision}")

    # Break down orbax ckpts by restoring via JAX to respect dtype
    initial_params = slice_initial_orbax_checkpoint(checkpoint_dir=checkpoint_dir, restore_precision="float32")

    # Merge any LoRA adapters into the base weights before slicing. This is a
    # no-op for non-LoRA configs. Without it, ``load_state_dict(..., strict=False)``
    # below would silently drop the adapter weights and the converted checkpoint
    # would diverge from the JAX runtime (see openpi issue #958).
    if _has_lora(model_config):
        logger.info(
            "LoRA detected in %s / %s; merging adapters before conversion.",
            model_config.paligemma_variant,
            model_config.action_expert_variant,
        )
        initial_params["paligemma_params"] = merge_lora_into_base(initial_params["paligemma_params"], model_config)

    # Process projection params
    if model_config.pi05:
        keys = [
            "action_in_proj",
            "action_out_proj",
            "time_mlp_in",
            "time_mlp_out",
        ]
    else:
        keys = [
            "state_proj",
            "action_in_proj",
            "action_out_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
        ]

    projection_params = {}
    for key in keys:
        kernel_params = initial_params["projection_params"][key]["kernel"]
        bias_params = initial_params["projection_params"][key]["bias"]
        if isinstance(kernel_params, dict):
            weight = kernel_params["value"]
            bias = bias_params["value"]
        else:
            weight = kernel_params
            bias = bias_params

        pytorch_weight_key = f"{key}.weight"
        pytorch_bias_key = f"{key}.bias"

        projection_params[pytorch_weight_key] = torch.from_numpy(np.array(weight)).T
        projection_params[pytorch_bias_key] = torch.from_numpy(np.array(bias))

    # Create bridge configs from the selected model config.
    paligemma_text_config = openpi.models.gemma.get_config(model_config.paligemma_variant)

    class PaliGemmaConfig:
        def __init__(self):
            self.vision_config = type(
                "obj",
                (object,),
                {
                    "hidden_size": 1152,
                    "num_hidden_layers": 27,
                    "num_attention_heads": 16,
                    "intermediate_size": 4304,
                    "patch_size": 14,
                    "projection_dim": 2048,
                },
            )()
            self.text_config = type(
                "obj",
                (object,),
                {
                    "hidden_size": paligemma_text_config.width,
                    "num_hidden_layers": paligemma_text_config.depth,
                    "num_attention_heads": paligemma_text_config.num_heads,
                    "head_dim": paligemma_text_config.head_dim,
                    "intermediate_size": paligemma_text_config.mlp_dim,
                },
            )()

    paligemma_config = PaliGemmaConfig()
    action_expert_config = openpi.models.gemma.get_config(model_config.action_expert_variant)

    # Process PaliGemma weights
    paligemma_params, expert_params = slice_paligemma_state_dict(initial_params["paligemma_params"], paligemma_config)

    # Process Gemma weights from expert_params
    gemma_params = slice_gemma_state_dict(
        expert_params, action_expert_config, num_expert=1, checkpoint_dir=checkpoint_dir, pi05=model_config.pi05
    )

    # Instantiate with the target dtype so load_state_dict does not quantize
    # float32 source tensors into the config default before saving.
    pi0_model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(
        dataclasses.replace(model_config, dtype=precision)
    )

    # Combine all parameters (no prefix needed for our model structure)
    all_params = {**paligemma_params, **gemma_params, **projection_params}

    # Load state dict.
    load_result = pi0_model.load_state_dict(all_params, strict=False)
    # If LoRA adapters were not merged correctly they would show up here as
    # unexpected_keys containing ``lora_a``/``lora_b``. Surface that clearly
    # rather than silently producing a divergent checkpoint.
    lora_unexpected = [k for k in load_result.unexpected_keys if "lora" in k.lower()]
    if lora_unexpected:
        raise RuntimeError(
            "LoRA adapter weights remain after conversion; merge_lora_into_base() did not "
            f"consume them. Unexpected keys: {lora_unexpected[:8]}" + ("..." if len(lora_unexpected) > 8 else "")
        )

    if precision == "float32":
        pi0_model = pi0_model.to(torch.float32)
    elif precision == "bfloat16":
        pi0_model = pi0_model.to(torch.bfloat16)

    # Save the converted model using safetensors
    os.makedirs(output_path, exist_ok=True)

    # Save model weights as SafeTensors using save_model to handle tied weights
    safetensors.torch.save_model(pi0_model, os.path.join(output_path, "model.safetensors"))

    # Copy assets folder if it exists
    assets_source = pathlib.Path(checkpoint_dir).parent / "assets"
    if assets_source.exists():
        assets_dest = pathlib.Path(output_path) / "assets"
        if assets_dest.exists():
            shutil.rmtree(assets_dest)
        shutil.copytree(assets_source, assets_dest)

    # Save config as JSON for reference
    config_dict = {
        "action_dim": model_config.action_dim,
        "action_horizon": model_config.action_horizon,
        "paligemma_variant": model_config.paligemma_variant,
        "action_expert_variant": model_config.action_expert_variant,
        "precision": precision,
    }
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    print("Model conversion completed successfully!")
    print(f"Model saved to {output_path}")


def main(
    checkpoint_dir: str,
    config_name: str,
    output_path: str | None = None,
    precision: Literal["float32", "bfloat16"] | None = None,
    *,
    inspect_only: bool = False,
):
    """Load JAX model and optionally convert to PyTorch.

    Args:
        checkpoint_dir: Path to the JAX checkpoint directory
        output_path: Path to save converted PyTorch model (required for conversion)
        precision: Precision for model conversion. When omitted, defaults to
            ``float32`` for LoRA-adapted configs and ``bfloat16`` otherwise;
            bfloat16 storage of merged LoRA weights drifts ~10x against the
            JAX original, so float32 is the safe default in that case.
        inspect_only: Only inspect parameter keys, don't convert
    """
    model_config = _config.get_config(config_name).model
    if not isinstance(model_config, openpi.models.pi0_config.Pi0Config):
        raise ValueError(f"Config {config_name} is not a Pi0Config")

    has_lora = _has_lora(model_config)
    if precision is None:
        precision = "float32" if has_lora else "bfloat16"
    elif has_lora and precision != "float32":
        logger.warning(
            "Converting a LoRA-adapted checkpoint with --precision=%s. The merged LoRA "
            "weights have a wider dynamic range than the base weights; storing them as "
            "%s typically introduces ~10x more numerical drift versus the JAX original. "
            "Use --precision float32 for parity.",
            precision,
            precision,
        )

    if inspect_only:
        load_jax_model_and_print_keys(checkpoint_dir)
    else:
        if not output_path:
            print("Error: --output_path is required for conversion. Use --inspect_only to only view keys.")
            return
        convert_pi0_checkpoint(checkpoint_dir, precision, output_path, model_config)


if __name__ == "__main__":
    tyro.cli(main)
