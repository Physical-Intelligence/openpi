"""SDPA backend introspection for PyTorch training logs."""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

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
    if query.device.type not in ("cuda", "hip"):
        return None

    op = getattr(torch.ops.aten, "_fused_sdp_choice", None)
    if op is None:
        return None

    mask = attn_mask
    if mask is None:
        # _fused_sdp_choice expects a tensor; use a zero mask when SDPA has no mask.
        seq_len = key.shape[-2]
        mask = torch.zeros(
            query.shape[0],
            1,
            query.shape[-2],
            seq_len,
            device=query.device,
            dtype=query.dtype,
        )

    try:
        with torch.no_grad():
            choice = int(op(query, key, value, mask, dropout, False, scaling, False))
    except TypeError:
        try:
            with torch.no_grad():
                choice = int(op(query, key, value, mask, dropout, False, scaling))
        except TypeError:
            try:
                with torch.no_grad():
                    choice = int(op(query, key, value, mask, dropout, False))
            except Exception:
                return None
    except Exception:
        return None

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
    """Run torch.ops.aten._fused_sdp_choice with joint-attention-like tensors."""
    if device.type not in ("cuda", "hip"):
        return None

    q = torch.zeros(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    # Additive mask matching pi0_pytorch._prepare_attention_masks_4d.
    attn_mask = torch.zeros(batch_size, 1, seq_len, seq_len, device=device, dtype=dtype)

    return probe_fused_sdp_choice_from_tensors(
        q, k, v, attn_mask, scaling=scaling, dropout=dropout
    )


def _tensor_probe_meta(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None,
) -> dict[str, Any]:
    def _shape_str(t: torch.Tensor) -> str:
        return "x".join(str(d) for d in t.shape)

    meta: dict[str, Any] = {
        "q": _shape_str(query),
        "k": _shape_str(key),
        "v": _shape_str(value),
        "dtype": str(query.dtype).removeprefix("torch."),
    }
    if attn_mask is not None:
        meta["mask"] = _shape_str(attn_mask)
    return meta


_RUNTIME_JOINT_SDPA_PROBE_LOGGED = False


def maybe_log_runtime_joint_sdpa_probe(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
) -> None:
    """Log SDPA backend selection once on the first real joint-attention forward."""
    global _RUNTIME_JOINT_SDPA_PROBE_LOGGED  # noqa: PLW0603
    if _RUNTIME_JOINT_SDPA_PROBE_LOGGED:
        return
    _RUNTIME_JOINT_SDPA_PROBE_LOGGED = True

    enabled = get_sdpa_enabled_backends()
    probe_meta = _tensor_probe_meta(query, key, value, attn_mask)
    probe: tuple[int, str] | None = None
    try:
        probe = probe_fused_sdp_choice_from_tensors(
            query, key, value, attn_mask, scaling=scaling, dropout=dropout
        )
    except Exception as exc:
        logging.warning("SDPA runtime kernel probe failed: %s", exc)

    lines = ["SDPA runtime probe (first joint forward, actual 4D mask):"]
    if enabled:
        parts = [f"{k}={v}" for k, v in enabled.items()]
        lines.append(f"  backends enabled: {', '.join(parts)}")
    meta_str = ", ".join(f"{k}={v}" for k, v in probe_meta.items())
    lines.append(f"  tensors: {meta_str}")
    if probe is None:
        lines.append("  selected: unavailable (_fused_sdp_choice failed)")
    else:
        choice, name = probe
        lines.append(f"  selected: {name} (choice={choice})")
    lines.append(
        "  note: _fused_sdp_choice predicts the SDPA dispatch; "
        "use torch profiler to confirm if kernels differ."
    )
    for line in lines:
        logging.info(line)


def probe_shapes_from_pi0_model(model: Any) -> dict[str, Any]:
    """Infer probe tensor shapes from a PI0Pytorch model."""
    attn = model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn
    cfg = attn.config
    num_heads = cfg.num_attention_heads
    head_dim = attn.head_dim
    scaling = attn.scaling
    param = next(attn.q_proj.parameters())
    dtype = param.dtype
    return {
        "num_heads": num_heads,
        "head_dim": head_dim,
        "scaling": scaling,
        "dtype": dtype,
    }


def format_joint_sdpa_log_lines(
    *,
    use_joint_sdpa: bool,
    device: torch.device,
    enabled_backends: dict[str, bool],
    probe: tuple[int, str] | None,
    probe_meta: dict[str, Any] | None = None,
) -> list[str]:
    lines = [f"Joint attention SDPA: enabled={use_joint_sdpa}"]
    if enabled_backends:
        parts = [f"{k}={v}" for k, v in enabled_backends.items()]
        lines.append(f"SDPA backends enabled (torch.backends.cuda): {', '.join(parts)}")

    rocm_backend = os.environ.get("PYTORCH_ROCM_SDPA_BACKEND")
    on_rocm = device.type == "hip" or getattr(torch.version, "hip", None) is not None
    if rocm_backend is not None:
        lines.append(f"PYTORCH_ROCM_SDPA_BACKEND={rocm_backend}")
    elif on_rocm:
        lines.append("PYTORCH_ROCM_SDPA_BACKEND=(not set)")

    if not use_joint_sdpa:
        lines.append("SDPA kernel probe: skipped (joint SDPA disabled)")
        return lines

    if probe is None:
        lines.append("SDPA startup probe: unavailable (_fused_sdp_choice failed or unsupported device)")
        lines.append(
            "SDPA runtime probe: will log on first joint forward with real batch/seq/mask (if joint SDPA enabled)"
        )
        return lines

    choice, name = probe
    meta = probe_meta or {}
    meta_str = ", ".join(f"{k}={v}" for k, v in meta.items())
    lines.append(f"SDPA startup probe (synthetic zero tensors, {meta_str}): selected={name} (choice={choice})")
    lines.append(
        "SDPA runtime probe: will log on first joint forward with real batch/seq/mask (if joint SDPA enabled)"
    )
    return lines


def log_joint_sdpa_backend_info(
    *,
    use_joint_sdpa: bool,
    device: torch.device,
    model: Any | None = None,
    batch_size: int = 2,
    seq_len: int = 512,
) -> None:
    """Log joint SDPA switch, enabled backends, env overrides, and probe result."""
    enabled = get_sdpa_enabled_backends()

    probe: tuple[int, str] | None = None
    probe_meta: dict[str, Any] | None = None

    if use_joint_sdpa and model is not None and device.type in ("cuda", "hip"):
        try:
            shapes = probe_shapes_from_pi0_model(model)
            probe_meta = {
                "batch": batch_size,
                "seq": seq_len,
                "heads": shapes["num_heads"],
                "head_dim": shapes["head_dim"],
                "dtype": str(shapes["dtype"]).removeprefix("torch."),
            }
            probe = probe_fused_sdp_choice(
                device=device,
                dtype=shapes["dtype"],
                batch_size=batch_size,
                seq_len=seq_len,
                num_heads=shapes["num_heads"],
                head_dim=shapes["head_dim"],
                scaling=shapes["scaling"],
            )
        except Exception as exc:
            logging.warning("SDPA kernel probe failed: %s", exc)

    for line in format_joint_sdpa_log_lines(
        use_joint_sdpa=use_joint_sdpa,
        device=device,
        enabled_backends=enabled,
        probe=probe,
        probe_meta=probe_meta,
    ):
        logging.info(line)
