import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class AtomicWeightLoader(WeightLoader):
    """Loads pi05_base into a Pi0Atomic model.

    The atomic action expert is a Gemmoe sparse-MoE block whose shared expert
    (``moe_1/expert_0``) keeps the dense GeGLU layout (``gating_einsum`` + ``linear``).
    We copy pi05_base's dense action FFN (``mlp_1``) into that shared expert; the
    extra atomic-skill experts, ``sigma_emb``, the reasoning head, and any LoRA
    adapters are not present in pi05_base and are left at model init (back-filled
    from the reference params, then kept by the training loop's overlay).
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        raw = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        flat = dict(flax.traverse_util.flatten_dict(raw, sep="/"))
        for k in list(flat):
            if k.endswith("mlp_1/gating_einsum"):
                flat[k[: -len("mlp_1/gating_einsum")] + "moe_1/expert_0/gating_einsum"] = flat[k]
            elif k.endswith("mlp_1/linear"):
                flat[k[: -len("mlp_1/linear")] + "moe_1/expert_0/linear"] = flat[k]
        remapped = flax.traverse_util.unflatten_dict(flat, sep="/")
        # Keep only keys that exist in the model (drops the now-unused ``mlp_1``) and
        # back-fill everything else (extra experts, sigma_emb, reasoning head, LoRA) from
        # the reference params so the returned tree matches the model structure exactly.
        return _merge_params(remapped, params, missing_regex=".*")


_STREAM2_SUFFIX_PAIRS = [
    ("q_einsum_2", "q_einsum_1"),
    ("kv_einsum_2", "kv_einsum_1"),
    ("qkv_einsum_2", "qkv_einsum_1"),
    ("attn_vec_einsum_2", "attn_vec_einsum_1"),
    ("pre_attention_norm_2", "pre_attention_norm_1"),
    ("pre_ffw_norm_2", "pre_ffw_norm_1"),
    ("final_norm_2", "final_norm_1"),
]


@dataclasses.dataclass(frozen=True)
class ActionMoeWeightLoader(WeightLoader):
    """Loads pi05_base into the action-MoE VLA models (target / trace_vla_moe / trace_vla_actionmoe).

    Fans pi05_base's dense action FFN ``mlp_1`` (GeGLU: ``gating_einsum`` + ``linear``) into K
    SwiGLU experts at ``moe_1/expert_{k}`` (``w1`` <- gate, ``w3`` <- value, ``w2`` <- linear).
    The dense FFN uses GELU and the MoE expert uses SiLU; finetuning bridges the mismatch quickly.

    ``moe_target`` selects which stream's MoE the single pi05 dense FFN seeds:
      - ``"moe_1"`` — the action MoE (target_vla_actionmoe, trace_vla_moe, trace_vla_actionmoe).
      - ``"moe_2"`` — the trace MoE (plain trace_vla, whose action stream stays dense).

    Note there is deliberately no "seed both" mode: pi05_base has exactly one dense action FFN,
    and no config has two pi05-FFN-shaped MoE streams. ``trace_vla_moe`` *is* MoE on both streams,
    but its trace MoE (``trace_moe_small``) is a shrunk width whose kernels don't match pi05's FFN,
    so it cannot be copied and is left at random init — only its action MoE (``moe_1``) is seeded.

    Optional remaps for the trace variant whose trace stream is a *dense* second expert:
      - ``copy_stream2_attn``: replicate stream-1 (``*_1``) attention/norm weights into stream-2
        (``*_2``) — used by trace_vla_actionmoe.
      - ``copy_mlp2``: copy the dense ``mlp_1`` into ``mlp_2`` (the dense trace FFN) — same variant.

    Everything not present in pi05_base (a shrunk trace MoE, completion/time/target heads, LoRA
    adapters) is left at model init via the training loop's overlay.
    """

    params_path: str
    num_action_experts: int = 5
    moe_target: str = "moe_1"  # which stream's MoE gets the experts: "moe_1" (action) or "moe_2" (trace)
    copy_stream2_attn: bool = False
    copy_mlp2: bool = False

    def load(self, params: at.Params) -> at.Params:
        raw = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        flat = dict(flax.traverse_util.flatten_dict(raw))

        if self.copy_stream2_attn:
            for trg_suffix, src_suffix in _STREAM2_SUFFIX_PAIRS:
                for k in list(flat):
                    if src_suffix not in k or any("lora" in str(seg) for seg in k):
                        continue
                    flat[tuple(trg_suffix if seg == src_suffix else seg for seg in k)] = flat[k]

        for k in [k for k in flat if k[-2:] == ("mlp_1", "gating_einsum")]:
            gating = flat[k]  # (L, 2, in, hidden): [...,0,...] -> w1 (gate); [...,1,...] -> w3 (value)
            base = k[:-2]  # path up to "layers"
            for e in range(self.num_action_experts):
                flat[(*base, self.moe_target, f"expert_{e}", "w1", "kernel")] = gating[..., 0, :, :]
                flat[(*base, self.moe_target, f"expert_{e}", "w3", "kernel")] = gating[..., 1, :, :]
            if self.copy_mlp2:
                flat[(*base, "mlp_2", "gating_einsum")] = gating
        for k in [k for k in flat if k[-2:] == ("mlp_1", "linear")]:
            linear = flat[k]  # (L, hidden, in) -> w2
            base = k[:-2]
            for e in range(self.num_action_experts):
                flat[(*base, self.moe_target, f"expert_{e}", "w2", "kernel")] = linear
            if self.copy_mlp2:
                flat[(*base, "mlp_2", "linear")] = linear

        remapped = flax.traverse_util.unflatten_dict(flat)
        return _merge_params(remapped, params, missing_regex=".*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
