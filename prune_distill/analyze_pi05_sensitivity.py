from __future__ import annotations

import csv
import dataclasses
import json
import logging
import os
import pathlib
import re
import shutil
import sys
from collections.abc import Callable
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import tyro

from openpi.models import model as _model
from openpi.shared import nnx_utils
from prune_distill import train_prefix_distill as distill


LOGGER = logging.getLogger("pi05_sensitivity")

TARGET_SUFFIXES = (
    "student/embedder/input_embedding",
    "student/layers/attn/q_einsum/w",
    "student/layers/attn/kv_einsum/w",
    "student/layers/attn/attn_vec_einsum/w",
    "student/layers/mlp/gating_einsum",
    "student/layers/mlp/linear",
    "student/layers/pre_attention_norm/scale",
    "student/layers/pre_ffw_norm/scale",
    "student/final_norm/scale",
)


@dataclasses.dataclass(frozen=True)
class SensitivityConfig:
    exp_name: str = "pi05_sensitivity"
    teacher_checkpoint: str = "/root/pi_train/pi05_libero/params"
    student_checkpoint: str | None = None
    output_dir: str = "/root/openpi-wr/checkpoints/prune_distill/sensitivity"
    train_config_name: str = "pi05_libero"
    dataset_path: str = "/root/flatten_fold_v2"
    norm_stats_assets_dir: str = "/root/pi_train/pi05_libero/assets"
    norm_stats_asset_id: str = "physical-intelligence/libero"
    max_examples: int | None = 2_048
    max_episodes: int | None = 8
    batch_size: int = 2
    num_workers: int = 2
    max_batches: int = 4
    eval_top_k: int = 24
    seed: int = 42
    dtype: str = "bfloat16"
    hidden_loss_weight: float = 1.0
    cosine_loss_weight: float = 0.1
    quant_bits: tuple[int, ...] = (8, 4)
    prune_ratios: tuple[float, ...] = (0.1, 0.3, 0.5)
    overwrite: bool = False


@dataclasses.dataclass(frozen=True)
class Candidate:
    path: str
    key: tuple[Any, ...]
    layer_idx: int | None
    num_params: int

    @property
    def name(self) -> str:
        if self.layer_idx is None:
            return self.path
        return f"{self.path}:layer_{self.layer_idx:02d}"

    @property
    def family(self) -> str:
        return self.path.removeprefix("student/")


def init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def to_distill_config(config: SensitivityConfig) -> distill.DistillConfig:
    return distill.DistillConfig(
        exp_name=config.exp_name,
        teacher_checkpoint=config.teacher_checkpoint,
        output_dir=str(pathlib.Path(config.output_dir).parent),
        train_config_name=config.train_config_name,
        dataset_path=config.dataset_path,
        norm_stats_assets_dir=config.norm_stats_assets_dir,
        norm_stats_asset_id=config.norm_stats_asset_id,
        max_examples=config.max_examples,
        max_episodes=config.max_episodes,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        num_train_steps=1,
        log_interval=1,
        save_interval=1,
        seed=config.seed,
        hidden_loss_weight=config.hidden_loss_weight,
        cosine_loss_weight=config.cosine_loss_weight,
        dtype=config.dtype,
    )


def prepare_output_dir(config: SensitivityConfig) -> pathlib.Path:
    output_dir = pathlib.Path(config.output_dir) / config.exp_name
    if output_dir.exists():
        if not config.overwrite:
            raise FileExistsError(f"{output_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def maybe_load_student_checkpoint(model: distill.PrefixDistillModel, config: SensitivityConfig) -> None:
    if config.student_checkpoint is None:
        return
    student_params = _model.restore_params(config.student_checkpoint, restore_type=np.ndarray)
    student_state = nnx.state(model.student)
    student_state.replace_by_pure_dict(student_params)
    nnx.update(model.student, student_state)
    LOGGER.info("Loaded student checkpoint from %s", config.student_checkpoint)


def collect_batches(config: SensitivityConfig) -> list[tuple[_model.Observation, jax.Array]]:
    distill_config = to_distill_config(config)
    train_config = distill.build_data_config(distill_config)
    loader = distill.create_distill_data_loader(train_config, distill_config)
    batches = []
    batch_iter = iter(loader)
    for _ in range(config.max_batches):
        try:
            batches.append(next(batch_iter))
        except StopIteration:
            break
    if not batches:
        raise ValueError("No batches were loaded for sensitivity analysis.")
    LOGGER.info("Collected %d batches for sensitivity scoring.", len(batches))
    return batches


def init_model(config: SensitivityConfig) -> distill.PrefixDistillModel:
    state = distill.init_state(to_distill_config(config))
    model = nnx.merge(state.model_def, state.params)
    maybe_load_student_checkpoint(model, config)
    return model


def should_analyze_path(path: str) -> bool:
    return any(path == suffix for suffix in TARGET_SUFFIXES)


def collect_candidates(model: distill.PrefixDistillModel) -> list[Candidate]:
    trainable_state = nnx.state(model).filter(distill.TRAINABLE_FILTER)
    candidates: list[Candidate] = []
    for key, value in trainable_state.flat_state().items():
        path = "/".join(str(part) for part in key)
        if not should_analyze_path(path):
            continue
        shape = value.value.shape
        if "/layers/" in path and shape:
            for layer_idx in range(shape[0]):
                candidates.append(
                    Candidate(
                        path=path,
                        key=key,
                        layer_idx=layer_idx,
                        num_params=int(np.prod(shape[1:])),
                    )
                )
        else:
            candidates.append(
                Candidate(
                    path=path,
                    key=key,
                    layer_idx=None,
                    num_params=int(np.prod(shape)),
                )
            )
    LOGGER.info("Collected %d candidate tensors/slices.", len(candidates))
    return candidates


def batch_rng(config: SensitivityConfig, batch_idx: int) -> jax.Array:
    return jax.random.fold_in(jax.random.key(config.seed + 17), batch_idx)


def evaluate_model(
    model: distill.PrefixDistillModel,
    batches: list[tuple[_model.Observation, jax.Array]],
    config: SensitivityConfig,
) -> dict[str, float]:
    totals = {
        "loss": 0.0,
        "hidden_mse": 0.0,
        "cosine_loss": 0.0,
        "valid_tokens": 0.0,
    }
    for batch_idx, (observation, _) in enumerate(batches):
        loss, metrics = model.compute_loss(
            batch_rng(config, batch_idx),
            observation,
            hidden_loss_weight=config.hidden_loss_weight,
            cosine_loss_weight=config.cosine_loss_weight,
            train=False,
        )
        totals["loss"] += float(loss)
        totals["hidden_mse"] += float(metrics["hidden_mse"])
        totals["cosine_loss"] += float(metrics["cosine_loss"])
        totals["valid_tokens"] += float(metrics["valid_tokens"])
    count = float(len(batches))
    return {name: value / count for name, value in totals.items()}


def compute_candidate_scores(
    model: distill.PrefixDistillModel,
    candidates: list[Candidate],
    batches: list[tuple[_model.Observation, jax.Array]],
    config: SensitivityConfig,
) -> dict[str, dict[str, float]]:
    scores = {
        candidate.name: {
            "grad_norm": 0.0,
            "taylor_sum": 0.0,
            "taylor_mean": 0.0,
            "param_norm": 0.0,
        }
        for candidate in candidates
    }

    for batch_idx, (observation, _) in enumerate(batches):
        def loss_fn(module: distill.PrefixDistillModel):
            return module.compute_loss(
                batch_rng(config, batch_idx),
                observation,
                hidden_loss_weight=config.hidden_loss_weight,
                cosine_loss_weight=config.cosine_loss_weight,
                train=False,
            )

        diff_state = nnx.DiffState(0, distill.TRAINABLE_FILTER)
        (_, _), grads = nnx.value_and_grad(loss_fn, has_aux=True, argnums=diff_state)(model)
        param_flat = nnx.state(model).filter(distill.TRAINABLE_FILTER).flat_state()
        grad_flat = grads.flat_state()

        for candidate in candidates:
            param = param_flat[candidate.key].value.astype(jnp.float32)
            grad = grad_flat[candidate.key].value.astype(jnp.float32)
            if candidate.layer_idx is not None:
                param = param[candidate.layer_idx]
                grad = grad[candidate.layer_idx]
            grad_norm = float(jnp.linalg.norm(grad))
            taylor = float(jnp.sum(jnp.abs(param * grad)))
            taylor_mean = taylor / max(candidate.num_params, 1)
            param_norm = float(jnp.linalg.norm(param))
            item = scores[candidate.name]
            item["grad_norm"] += grad_norm
            item["taylor_sum"] += taylor
            item["taylor_mean"] += taylor_mean
            item["param_norm"] += param_norm

    count = float(len(batches))
    for item in scores.values():
        for name in ("grad_norm", "taylor_sum", "taylor_mean", "param_norm"):
            item[name] /= count
    return scores


def fake_quantize(x: jax.Array, bits: int) -> jax.Array:
    if bits < 2:
        raise ValueError(f"bits must be >= 2, got {bits}")
    x32 = x.astype(jnp.float32)
    max_abs = jnp.max(jnp.abs(x32))
    if float(max_abs) == 0.0:
        return x
    qmax = (1 << (bits - 1)) - 1
    scale = max_abs / qmax
    quantized = jnp.clip(jnp.round(x32 / scale), -qmax, qmax) * scale
    return quantized.astype(x.dtype)


def magnitude_prune(x: jax.Array, ratio: float) -> jax.Array:
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"prune ratio must be in [0, 1], got {ratio}")
    if ratio == 0.0:
        return x
    if ratio == 1.0:
        return jnp.zeros_like(x)

    x32 = x.astype(jnp.float32)
    flat_abs = jnp.abs(x32).reshape(-1)
    num_pruned = int(round(ratio * flat_abs.size))
    if num_pruned <= 0:
        return x
    if num_pruned >= flat_abs.size:
        return jnp.zeros_like(x)

    threshold = jnp.partition(flat_abs, num_pruned - 1)[num_pruned - 1]
    mask = jnp.abs(x32) > threshold
    return jnp.where(mask, x32, 0.0).astype(x.dtype)


def perturb_candidate(
    model: distill.PrefixDistillModel,
    candidate: Candidate,
    transform: Callable[[jax.Array], jax.Array],
) -> tuple[nnx.State, jax.Array]:
    candidate_filter = nnx.All(nnx.Param, nnx_utils.PathRegex(re.escape(candidate.path)))
    candidate_state = nnx.state(model, candidate_filter)
    flat = candidate_state.flat_state()
    if len(flat) != 1:
        raise ValueError(f"Expected exactly one tensor for {candidate.path}, found {list(flat)}")
    variable = next(iter(flat.values()))
    original = variable.value
    updated = transform(original if candidate.layer_idx is None else original[candidate.layer_idx])
    if candidate.layer_idx is None:
        variable.value = updated
    else:
        variable.value = original.at[candidate.layer_idx].set(updated)
    nnx.update(model, candidate_state)
    return candidate_state, original


def restore_candidate(
    model: distill.PrefixDistillModel,
    candidate_state: nnx.State,
    original: jax.Array,
) -> None:
    variable = next(iter(candidate_state.flat_state().values()))
    variable.value = original
    nnx.update(model, candidate_state)


def evaluate_perturbation(
    model: distill.PrefixDistillModel,
    candidate: Candidate,
    batches: list[tuple[_model.Observation, jax.Array]],
    config: SensitivityConfig,
    transform: Callable[[jax.Array], jax.Array],
) -> dict[str, float]:
    candidate_state, original = perturb_candidate(model, candidate, transform)
    try:
        return evaluate_model(model, batches, config)
    finally:
        restore_candidate(model, candidate_state, original)


def metric_key(prefix: str, value: int | float, name: str) -> str:
    if isinstance(value, int):
        return f"{prefix}_{value}_{name}"
    return f"{prefix}_{int(round(value * 100)):02d}p_{name}"


def summarize_by_family(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}
    for record in records:
        family = record["family"]
        if family not in grouped:
            grouped[family] = {"taylor_sum": 0.0, "grad_norm": 0.0}
            counts[family] = 0
        grouped[family]["taylor_sum"] += float(record["taylor_sum"])
        grouped[family]["grad_norm"] += float(record["grad_norm"])
        counts[family] += 1

    rows = []
    for family, values in grouped.items():
        count = counts[family]
        rows.append(
            {
                "family": family,
                "count": count,
                "mean_taylor_sum": values["taylor_sum"] / count,
                "mean_grad_norm": values["grad_norm"] / count,
            }
        )
    rows.sort(key=lambda row: row["mean_taylor_sum"], reverse=True)
    return rows


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(config: SensitivityConfig) -> None:
    init_logging()
    output_dir = prepare_output_dir(config)
    (output_dir / "config.json").write_text(json.dumps(dataclasses.asdict(config), indent=2))

    LOGGER.info("Building pi05 sensitivity model and data.")
    model = init_model(config)
    batches = collect_batches(config)
    candidates = collect_candidates(model)

    LOGGER.info("Evaluating baseline distillation metrics.")
    baseline = evaluate_model(model, batches, config)
    LOGGER.info("Baseline metrics: %s", baseline)

    LOGGER.info("Computing distillation gradient/Taylor sensitivity.")
    scores = compute_candidate_scores(model, candidates, batches, config)

    records: list[dict[str, Any]] = []
    for candidate in candidates:
        item = scores[candidate.name]
        records.append(
            {
                "name": candidate.name,
                "path": candidate.path,
                "family": candidate.family,
                "layer_idx": candidate.layer_idx,
                "num_params": candidate.num_params,
                "param_norm": item["param_norm"],
                "grad_norm": item["grad_norm"],
                "taylor_sum": item["taylor_sum"],
                "taylor_mean": item["taylor_mean"],
            }
        )

    records.sort(key=lambda row: row["taylor_sum"], reverse=True)
    eval_records = records[: config.eval_top_k]
    LOGGER.info(
        "Running fake-quant and magnitude-prune perturbations on top %d candidates.",
        len(eval_records),
    )

    record_map = {record["name"]: record for record in records}
    candidate_map = {candidate.name: candidate for candidate in candidates}
    for rank, record in enumerate(eval_records, start=1):
        candidate = candidate_map[record["name"]]
        LOGGER.info(
            "[%d/%d] %s taylor_sum=%.4f",
            rank,
            len(eval_records),
            candidate.name,
            record["taylor_sum"],
        )

        for bits in config.quant_bits:
            perturbed = evaluate_perturbation(
                model,
                candidate,
                batches,
                config,
                lambda x, bits=bits: fake_quantize(x, bits),
            )
            record_map[candidate.name][metric_key("quant", bits, "loss")] = perturbed["loss"]
            record_map[candidate.name][metric_key("quant", bits, "loss_delta")] = perturbed["loss"] - baseline["loss"]
            record_map[candidate.name][metric_key("quant", bits, "hidden_mse_delta")] = (
                perturbed["hidden_mse"] - baseline["hidden_mse"]
            )
            record_map[candidate.name][metric_key("quant", bits, "cosine_delta")] = (
                perturbed["cosine_loss"] - baseline["cosine_loss"]
            )

        for ratio in config.prune_ratios:
            perturbed = evaluate_perturbation(
                model,
                candidate,
                batches,
                config,
                lambda x, ratio=ratio: magnitude_prune(x, ratio),
            )
            record_map[candidate.name][metric_key("prune", ratio, "loss")] = perturbed["loss"]
            record_map[candidate.name][metric_key("prune", ratio, "loss_delta")] = perturbed["loss"] - baseline["loss"]
            record_map[candidate.name][metric_key("prune", ratio, "hidden_mse_delta")] = (
                perturbed["hidden_mse"] - baseline["hidden_mse"]
            )
            record_map[candidate.name][metric_key("prune", ratio, "cosine_delta")] = (
                perturbed["cosine_loss"] - baseline["cosine_loss"]
            )

    family_summary = summarize_by_family(records)

    summary = {
        "baseline": baseline,
        "top_by_taylor_sum": records[: min(20, len(records))],
        "family_summary": family_summary[: min(20, len(family_summary))],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_csv(output_dir / "candidate_scores.csv", records)
    write_csv(output_dir / "family_summary.csv", family_summary)

    LOGGER.info("Top sensitivity candidates:")
    for record in records[:10]:
        LOGGER.info(
            "  %s | taylor_sum=%.4f grad_norm=%.4f",
            record["name"],
            record["taylor_sum"],
            record["grad_norm"],
        )
    LOGGER.info("Sensitivity analysis written to %s", output_dir)


if __name__ == "__main__":
    main(tyro.cli(SensitivityConfig))
