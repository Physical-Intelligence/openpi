"""SDPA backend introspection for PyTorch training logs."""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

import torch

JointAttentionMode = Literal["eager", "sdpa", "auto"]
SubmoduleAttentionMode = Literal["eager", "sdpa"]

_FLASH_SDPA_BACKEND_NAMES = frozenset({"flash_attention", "flash"})

_LAST_FUSED_SDP_CHOICE_ERROR: str | None = None

# Matches aten::SDPBackend ordering used by torch.ops.aten._fused_sdp_choice.
_FUSED_SDP_CHOICE_NAMES: dict[int, str] = {
    -1: "error",
    0: "math",
    1: "flash_attention",
    2: "efficient_attention",
    3: "cudnn_attention",
}


def _sdp_backend_name_from_choice(choice: int) -> str:
    try:
        from torch.nn.attention import SDPBackend

        for attr in dir(SDPBackend):
            if not attr.isupper() or attr.startswith("_"):
                continue
            member = getattr(SDPBackend, attr)
            try:
                if int(member) == choice:
                    return attr.lower()
            except (TypeError, ValueError):
                continue
    except ImportError:
        pass
    return _FUSED_SDP_CHOICE_NAMES.get(choice, f"unknown({choice})")


def get_sdpa_enabled_backends() -> dict[str, bool]:
    """Return which SDPA implementations PyTorch allows (torch.backends.cuda)."""
    flags: dict[str, bool] = {}
    for name in ("flash_sdp", "mem_efficient_sdp", "math_sdp", "cudnn_sdp"):
        fn = getattr(torch.backends.cuda, f"{name}_enabled", None)
        if callable(fn):
            flags[name] = bool(fn())
    return flags


def probe_fused_sdp_choice_from_tensors(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None,
    *,
    scaling: float,
    dropout: float = 0.0,
) -> tuple[int, str] | None:
    """Run torch.ops.aten._fused_sdp_choice on the exact Q/K/V/mask passed to SDPA."""
    global _LAST_FUSED_SDP_CHOICE_ERROR  # noqa: PLW0603

    if query.device.type not in ("cuda", "hip"):
        _LAST_FUSED_SDP_CHOICE_ERROR = f"unsupported device type {query.device.type!r}"
        return None

    op = getattr(torch.ops.aten, "_fused_sdp_choice", None)
    if op is None:
        _LAST_FUSED_SDP_CHOICE_ERROR = "torch.ops.aten._fused_sdp_choice not available"
        return None

    # SigLIP etc. use attn_mask=None; probing with a synthetic 4D mask often yields math, not flash.
    mask = attn_mask
    fallback_mask = None
    if mask is None:
        seq_len = key.shape[-2]
        fallback_mask = torch.zeros(
            query.shape[0],
            1,
            query.shape[-2],
            seq_len,
            device=query.device,
            dtype=query.dtype,
        )

    mask_candidates: list[torch.Tensor | None] = [None, fallback_mask] if mask is None else [mask]
    call_variants: list = []
    for mask_arg in mask_candidates:
        call_variants.extend(
            [
                lambda ma=mask_arg: op(
                    query, key, value, ma, dropout, False, scale=scaling, enable_gqa=False
                ),
                lambda ma=mask_arg: op(query, key, value, ma, dropout, False, scale=scaling),
                lambda ma=mask_arg: op(query, key, value, ma, dropout, False),
            ]
        )
        if mask_arg is not None:
            call_variants.extend(
                [
                    lambda ma=mask_arg: op(query, key, value, ma, dropout, False, scaling, False),
                    lambda ma=mask_arg: op(query, key, value, ma, dropout, False, scaling),
                ]
            )
    choice: int | None = None
    last_exc: Exception | None = None
    with torch.no_grad():
        for call in call_variants:
            try:
                choice = int(call())
                break
            except Exception as exc:
                last_exc = exc
                continue
    if choice is None:
        _LAST_FUSED_SDP_CHOICE_ERROR = repr(last_exc) if last_exc else "all call variants failed"
        return None

    _LAST_FUSED_SDP_CHOICE_ERROR = None
    return choice, _sdp_backend_name_from_choice(choice)


def probe_fused_sdp_choice(
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    scaling: float,
    dropout: float = 0.0,
) -> tuple[int, str] | None:
    """Run _fused_sdp_choice with joint-attention-like tensors (4D additive mask)."""
    if device.type not in ("cuda", "hip"):
        return None

    q = torch.zeros(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    attn_mask = torch.zeros(batch_size, 1, seq_len, seq_len, device=device, dtype=dtype)
    return probe_fused_sdp_choice_from_tensors(
        q, k, v, attn_mask, scaling=scaling, dropout=dropout
    )


def _format_probe_selected(probe: tuple[int, str] | None) -> str:
    if probe is None:
        if _LAST_FUSED_SDP_CHOICE_ERROR:
            return f"unavailable ({_LAST_FUSED_SDP_CHOICE_ERROR})"
        return "unavailable"
    name, choice = probe[1], probe[0]
    return f"{name} (choice={choice})"


def resolve_pytorch_joint_attention(
    *,
    joint_attention: JointAttentionMode,
    use_joint_sdpa_legacy: bool,
) -> JointAttentionMode:
    """Apply legacy ``pytorch_use_joint_sdpa`` override."""
    if use_joint_sdpa_legacy:
        return "sdpa"
    return joint_attention


def configure_submodule_attention(
    paligemma_with_expert: Any, implementation: SubmoduleAttentionMode
) -> None:
    """Set HF ``_attn_implementation`` for vision / language / expert (non-joint paths)."""
    pg = paligemma_with_expert.paligemma
    for cfg in (
        pg.config.vision_config,
        pg.config.text_config,
        paligemma_with_expert.gemma_expert.model.config,
    ):
        cfg._attn_implementation = implementation  # noqa: SLF001


_AUTO_JOINT_RESOLVED: str | None = None


def resolve_joint_use_sdpa(
    mode: JointAttentionMode,
    *,
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float,
) -> tuple[bool, str]:
    """Whether joint attention should call SDPA (True) or eager matmul (False)."""
    global _AUTO_JOINT_RESOLVED  # noqa: PLW0603

    if mode == "eager":
        return False, "eager"
    if mode == "sdpa":
        return True, "sdpa"

    if _AUTO_JOINT_RESOLVED is not None:
        return _AUTO_JOINT_RESOLVED == "sdpa", _AUTO_JOINT_RESOLVED

    from transformers.models.gemma import modeling_gemma

    key_states = modeling_gemma.repeat_kv(key, module.num_key_value_groups)
    value_states = modeling_gemma.repeat_kv(value, module.num_key_value_groups)
    attn_mask = None
    if attention_mask is not None:
        attn_mask = attention_mask[:, :, :, : key_states.shape[-2]]
    dropout_p = dropout if module.training else 0.0

    probe = probe_fused_sdp_choice_from_tensors(
        query, key_states, value_states, attn_mask, scaling=scaling, dropout=dropout_p
    )
    if probe is not None and probe[1] in _FLASH_SDPA_BACKEND_NAMES:
        _AUTO_JOINT_RESOLVED = "sdpa"
    elif attn_mask is not None:
        _AUTO_JOINT_RESOLVED = "eager"
    else:
        _AUTO_JOINT_RESOLVED = "sdpa"

    return _AUTO_JOINT_RESOLVED == "sdpa", _AUTO_JOINT_RESOLVED


def _paligemma_from_probe_model(model: Any) -> Any:
    if hasattr(model, "paligemma_with_expert"):
        return model.paligemma_with_expert.paligemma
    return model.paligemma


def _probe_siglip_sdpa(
    model: Any, device: torch.device, batch_size: int
) -> tuple[int, str] | None:
    paligemma = _paligemma_from_probe_model(model)
    vision = paligemma.vision_tower
    vision_model = vision.vision_model if hasattr(vision, "vision_model") else vision
    attn = vision_model.encoder.layers[0].self_attn
    cfg = attn.config
    image_size = getattr(cfg, "image_size", 224)
    if isinstance(image_size, (list, tuple)):
        image_size = int(image_size[0])
    patch_size = int(getattr(cfg, "patch_size", 16))
    seq_len = (int(image_size) // patch_size) ** 2
    dtype = next(attn.q_proj.parameters()).dtype
    q = torch.zeros(batch_size, attn.num_heads, seq_len, attn.head_dim, device=device, dtype=dtype)
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    return probe_fused_sdp_choice_from_tensors(q, k, v, None, scaling=float(attn.scale))


def _probe_joint_sdpa(
    model: Any, device: torch.device, batch_size: int, seq_len: int
) -> tuple[int, str] | None:
    attn = model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn
    cfg = attn.config
    dtype = next(attn.q_proj.parameters()).dtype
    return probe_fused_sdp_choice(
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=cfg.num_attention_heads,
        head_dim=attn.head_dim,
        scaling=attn.scaling,
    )


def log_joint_sdpa_backend_info(
    *,
    joint_attention: JointAttentionMode,
    submodule_attn: SubmoduleAttentionMode,
    device: torch.device,
    model: Any | None = None,
    batch_size: int = 2,
    seq_len: int = 512,
) -> None:
    """Log attention modes, enabled SDPA backends, and one-line probe summary."""
    enabled = get_sdpa_enabled_backends()
    enabled_str = ", ".join(f"{k}={v}" for k, v in enabled.items()) if enabled else "(none)"

    on_gpu = model is not None and device.type in ("cuda", "hip")

    if submodule_attn == "sdpa" and on_gpu:
        siglip_sel = _format_probe_selected(_probe_siglip_sdpa(model, device, batch_size))
    elif submodule_attn == "sdpa":
        siglip_sel = "unavailable (no GPU)"
    else:
        siglip_sel = "eager (HF, not SDPA)"

    if joint_attention == "auto":
        joint_sel = "eager matmul (auto; Pi0 training uses 4D mask, not joint SDPA)"
    elif joint_attention == "eager":
        joint_sel = "eager matmul"
    elif on_gpu:
        joint_sel = _format_probe_selected(_probe_joint_sdpa(model, device, batch_size, seq_len))
    else:
        joint_sel = "unavailable (no GPU)"

    logging.info(f"Attention: joint={joint_attention}, submodules(HF)={submodule_attn}")
    logging.info(f"SDPA backends enabled (torch.backends.cuda): {enabled_str}")
    logging.info(f"SDPA probe selected: SigLIP={siglip_sel}; joint={joint_sel}")

    rocm_backend = os.environ.get("PYTORCH_ROCM_SDPA_BACKEND")
    on_rocm = device.type == "hip" or getattr(torch.version, "hip", None) is not None
    if rocm_backend is not None:
        logging.info(f"PYTORCH_ROCM_SDPA_BACKEND={rocm_backend}")
    elif on_rocm:
        logging.info("PYTORCH_ROCM_SDPA_BACKEND=(not set)")
