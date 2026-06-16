from __future__ import annotations

import collections
import csv
import dataclasses
import json
import logging
import math
import os
import pathlib
import shutil
import sys
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

from flax import nnx
import jax
import numpy as np
from openpi_client import image_tools
import tyro

from openpi.models import model as _model
from openpi.policies import policy_config
import openpi.training.config as training_config
import openpi.training.data_loader as data_loader
import openpi.transforms as _transforms
from prune_distill import train_prefix_distill as distill

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset


LOGGER = logging.getLogger("pi05_benchmark")
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass(frozen=True)
class BenchmarkConfig:
    exp_name: str = "pi05_benchmark"
    output_dir: str = "/root/openpi-wr/checkpoints/prune_distill/benchmark"

    train_config_name: str = "pi05_libero"
    origin_checkpoint_dir: str | None = "/root/pi_train/pi05_libero"
    teacher_checkpoint: str = "/root/pi_train/pi05_libero/params"
    pruned_student_checkpoint: str | None = None

    dataset_path: str = "/root/flatten_fold_v2"
    norm_stats_assets_dir: str = "/root/pi_train/pi05_libero/assets"
    norm_stats_asset_id: str = "physical-intelligence/libero"
    max_examples: int | None = 50_000
    max_episodes: int | None = None
    max_eval_examples: int = 256
    batch_size: int = 4
    num_workers: int = 2

    run_dataset_benchmark: bool = True
    run_libero_rollout: bool = False

    sample_actions_num_steps: int = 10
    disable_policy_norm_stats: bool = False
    action_tolerance: float = 0.05

    libero_task_suite_names: tuple[str, ...] = ("libero_spatial",)
    libero_num_trials_per_task: int = 10
    libero_max_tasks: int | None = None
    libero_replan_steps: int = 5
    libero_num_steps_wait: int = 10
    libero_resize_size: int = 224
    libero_video_out_dir: str | None = None

    hidden_loss_weight: float = 1.0
    cosine_loss_weight: float = 0.1
    dtype: str = "bfloat16"
    seed: int = 7
    overwrite: bool = False


def init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _augment_oom_error(exc: Exception, context: str) -> RuntimeError:
    err = RuntimeError(
        f"{context} failed due to JAX/GPU OOM. "
        "Retry with CPU fallback, for example: `JAX_PLATFORMS=cpu .venv/bin/python ...`"
    )
    err.__cause__ = exc
    return err


def prepare_output_dir(config: BenchmarkConfig) -> pathlib.Path:
    output_dir = pathlib.Path(config.output_dir) / config.exp_name
    if output_dir.exists():
        if not config.overwrite:
            raise FileExistsError(f"{output_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def resolve_policy_checkpoint_dir(checkpoint_dir: str) -> pathlib.Path:
    path = pathlib.Path(checkpoint_dir).expanduser().resolve()
    if path.name == "params" and path.parent.exists():
        return path.parent
    if (path / "params").exists():
        return path
    raise FileNotFoundError(f"Expected a checkpoint directory containing params/: {checkpoint_dir}")


def resolve_teacher_checkpoint(path: str) -> str:
    resolved = pathlib.Path(path).expanduser().resolve()
    if resolved.name == "params":
        return str(resolved)
    if (resolved / "params").exists():
        return str((resolved / "params").resolve())
    raise FileNotFoundError(f"Expected teacher checkpoint params at {path}")


def resolve_student_checkpoint(path: str) -> pathlib.Path:
    resolved = pathlib.Path(path).expanduser().resolve()
    if resolved.name == "student":
        return resolved
    if (resolved / "student").exists():
        return (resolved / "student").resolve()
    raise FileNotFoundError(f"Expected pruned student checkpoint at {path}")


def to_distill_config(config: BenchmarkConfig) -> distill.DistillConfig:
    return distill.DistillConfig(
        exp_name=config.exp_name,
        teacher_checkpoint=resolve_teacher_checkpoint(config.teacher_checkpoint),
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


@dataclasses.dataclass(frozen=True)
class DatasetResources:
    train_config: training_config.TrainConfig
    data_config: training_config.DataConfig
    dataset: Any
    dataset_meta: lerobot_dataset.LeRobotDatasetMetadata


def create_dataset_resources(config: BenchmarkConfig) -> DatasetResources:
    distill_config = to_distill_config(config)
    train_config = distill.build_data_config(distill_config)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

    dataset_root = pathlib.Path(config.dataset_path).expanduser().resolve()
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(dataset_root.name, root=dataset_root)
    selected_episodes = distill.select_episode_subset(distill_config, dataset_meta)

    dataset = lerobot_dataset.LeRobotDataset(
        dataset_root.name,
        root=dataset_root,
        episodes=selected_episodes,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(train_config.model.action_horizon)]
            for key in data_config.action_sequence_keys
        },
    )
    data_config = distill.adapt_data_config_to_dataset(data_config, dataset)
    if data_config.prompt_from_task:
        dataset = data_loader.TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return DatasetResources(
        train_config=train_config,
        data_config=data_config,
        dataset=dataset,
        dataset_meta=dataset_meta,
    )


def choose_policy_norm_stats(
    data_config: training_config.DataConfig,
    sample: dict[str, Any],
    *,
    disable_policy_norm_stats: bool,
) -> dict[str, Any] | None:
    if disable_policy_norm_stats:
        LOGGER.warning("Policy norm stats disabled by config.")
        return {}

    norm_stats = data_config.norm_stats
    if norm_stats is None:
        LOGGER.warning("No norm stats available. Using no-op normalization for policy inference.")
        return {}

    state_stats = norm_stats.get("state")
    if state_stats is None:
        return norm_stats

    sample_state = np.asarray(sample["state"])
    sample_state_dim = int(sample_state.shape[-1])
    stats_state_dim = int(state_stats.mean.shape[-1])
    if sample_state_dim > stats_state_dim:
        LOGGER.warning(
            "Dataset state dim (%d) exceeds available norm stats dim (%d). Using no-op normalization for policy inference.",
            sample_state_dim,
            stats_state_dim,
        )
        return {}
    return norm_stats


def _safe_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if hasattr(value, "item"):
        try:
            return str(value.item())
        except Exception:
            pass
    return str(value)


def align_actions(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if pred.ndim == 1:
        pred = pred[None, :]
    if target.ndim == 1:
        target = target[None, :]
    horizon = min(pred.shape[0], target.shape[0])
    dims = min(pred.shape[1], target.shape[1])
    if horizon <= 0 or dims <= 0:
        raise ValueError(f"Cannot align action arrays with shapes {pred.shape} and {target.shape}")
    return pred[:horizon, :dims], target[:horizon, :dims]


def aggregate_dataset_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "examples_evaluated": 0,
        }
    return {
        "examples_evaluated": len(rows),
        "avg_first_action_l2": float(np.mean([row["first_action_l2"] for row in rows])),
        "avg_chunk_l2": float(np.mean([row["chunk_l2"] for row in rows])),
        "avg_chunk_l1": float(np.mean([row["chunk_l1"] for row in rows])),
        "first_action_match_rate": float(np.mean([row["first_action_match"] for row in rows])),
        "chunk_match_rate": float(np.mean([row["chunk_match"] for row in rows])),
        "avg_infer_ms": float(np.mean([row["infer_ms"] for row in rows])),
        "avg_compare_horizon": float(np.mean([row["compare_horizon"] for row in rows])),
        "avg_compare_dims": float(np.mean([row["compare_dims"] for row in rows])),
    }


def benchmark_origin_dataset(config: BenchmarkConfig, output_dir: pathlib.Path) -> dict[str, Any] | None:
    if config.origin_checkpoint_dir is None:
        LOGGER.info("Origin pi05 checkpoint not provided. Skipping offline dataset benchmark.")
        return None

    resources = create_dataset_resources(config)
    if len(resources.dataset) == 0:
        raise ValueError("Dataset is empty.")

    sample = resources.dataset[0]
    norm_stats = choose_policy_norm_stats(
        resources.data_config,
        sample,
        disable_policy_norm_stats=config.disable_policy_norm_stats,
    )
    try:
        policy = policy_config.create_trained_policy(
            resources.train_config,
            resolve_policy_checkpoint_dir(config.origin_checkpoint_dir),
            repack_transforms=resources.data_config.repack_transforms,
            norm_stats=norm_stats,
            sample_kwargs={"num_steps": config.sample_actions_num_steps},
        )
    except Exception as e:
        if "RESOURCE_EXHAUSTED" in str(e) or "Out of memory" in str(e):
            raise _augment_oom_error(e, "Origin pi05 offline dataset benchmark")
        raise

    num_examples = min(len(resources.dataset), config.max_eval_examples)
    rows: list[dict[str, Any]] = []
    LOGGER.info("Running offline origin pi05 dataset benchmark on %d examples.", num_examples)
    for idx in range(num_examples):
        item = resources.dataset[idx]
        result = policy.infer(item)
        pred_actions, target_actions = align_actions(result["actions"], item["actions"])
        diff = pred_actions - target_actions
        first_action_l2 = float(np.linalg.norm(diff[0]))
        chunk_l2 = float(np.sqrt(np.mean(np.square(diff))))
        chunk_l1 = float(np.mean(np.abs(diff)))
        rows.append(
            {
                "index": idx,
                "task": _safe_string(item.get("task")),
                "prompt": _safe_string(item.get("prompt")),
                "compare_horizon": int(pred_actions.shape[0]),
                "compare_dims": int(pred_actions.shape[1]),
                "first_action_l2": first_action_l2,
                "chunk_l2": chunk_l2,
                "chunk_l1": chunk_l1,
                "first_action_match": float(first_action_l2 <= config.action_tolerance),
                "chunk_match": float(chunk_l2 <= config.action_tolerance),
                "infer_ms": float(result["policy_timing"]["infer_ms"]),
            }
        )

    summary = aggregate_dataset_rows(rows)
    write_csv(output_dir / "origin_pi05_dataset_metrics.csv", rows)
    return summary


def maybe_load_pruned_student(model: distill.PrefixDistillModel, checkpoint: pathlib.Path) -> None:
    student_params = _model.restore_params(checkpoint, restore_type=np.ndarray)
    student_state = nnx.state(model.student)
    student_state.replace_by_pure_dict(student_params)
    nnx.update(model.student, student_state)


def benchmark_pruned_dataset(config: BenchmarkConfig, output_dir: pathlib.Path) -> dict[str, Any] | None:
    if config.pruned_student_checkpoint is None:
        LOGGER.info("Pruned student checkpoint not provided. Skipping pruned offline benchmark.")
        return None

    distill_config = to_distill_config(config)
    train_config = distill.build_data_config(distill_config)
    loader = distill.create_distill_data_loader(train_config, distill_config)

    try:
        state = distill.init_state(distill_config)
    except Exception as e:
        if "RESOURCE_EXHAUSTED" in str(e) or "Out of memory" in str(e):
            raise _augment_oom_error(e, "Pruned prefix offline benchmark")
        raise
    model = nnx.merge(state.model_def, state.params)
    maybe_load_pruned_student(model, resolve_student_checkpoint(config.pruned_student_checkpoint))

    rows: list[dict[str, Any]] = []
    LOGGER.info("Running pruned prefix offline benchmark on up to %d batches.", config.batch_size)
    for batch_idx, (observation, _) in enumerate(loader):
        if batch_idx >= max(1, math.ceil(config.max_eval_examples / max(config.batch_size, 1))):
            break
        loss, metrics = model.compute_loss(
            jax.random.fold_in(jax.random.key(config.seed + 101), batch_idx),
            observation,
            hidden_loss_weight=config.hidden_loss_weight,
            cosine_loss_weight=config.cosine_loss_weight,
            train=False,
        )
        rows.append(
            {
                "batch_idx": batch_idx,
                "loss": float(loss),
                "hidden_mse": float(metrics["hidden_mse"]),
                "cosine_loss": float(metrics["cosine_loss"]),
                "valid_tokens": float(metrics["valid_tokens"]),
            }
        )

    if not rows:
        return {"batches_evaluated": 0}

    summary = {
        "batches_evaluated": len(rows),
        "avg_loss": float(np.mean([row["loss"] for row in rows])),
        "avg_hidden_mse": float(np.mean([row["hidden_mse"] for row in rows])),
        "avg_cosine_loss": float(np.mean([row["cosine_loss"] for row in rows])),
        "avg_valid_tokens": float(np.mean([row["valid_tokens"] for row in rows])),
    }
    write_csv(output_dir / "pruned_prefix_dataset_metrics.csv", rows)
    return summary


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _libero_max_steps(task_suite_name: str) -> int:
    suite_to_steps = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    if task_suite_name not in suite_to_steps:
        raise ValueError(f"Unknown LIBERO task suite: {task_suite_name}")
    return suite_to_steps[task_suite_name]


def benchmark_origin_libero(config: BenchmarkConfig, output_dir: pathlib.Path) -> dict[str, Any] | None:
    if not config.run_libero_rollout:
        return None
    if config.origin_checkpoint_dir is None:
        LOGGER.info("Origin pi05 checkpoint not provided. Skipping LIBERO rollout benchmark.")
        return None

    try:
        import imageio
        from libero.libero import benchmark
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError as e:
        raise ImportError("LIBERO rollout benchmark requires the LIBERO env dependencies to be installed.") from e

    train_config = training_config.get_config(config.train_config_name)
    try:
        policy = policy_config.create_trained_policy(
            train_config,
            resolve_policy_checkpoint_dir(config.origin_checkpoint_dir),
            sample_kwargs={"num_steps": config.sample_actions_num_steps},
        )
    except Exception as e:
        if "RESOURCE_EXHAUSTED" in str(e) or "Out of memory" in str(e):
            raise _augment_oom_error(e, "Origin pi05 LIBERO rollout benchmark")
        raise

    suite_summaries: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    video_dir = pathlib.Path(config.libero_video_out_dir).resolve() if config.libero_video_out_dir else None
    if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)

    benchmark_dict = benchmark.get_benchmark_dict()
    np.random.seed(config.seed)
    for suite_name in config.libero_task_suite_names:
        task_suite = benchmark_dict[suite_name]()
        num_tasks = task_suite.n_tasks
        if config.libero_max_tasks is not None:
            num_tasks = min(num_tasks, config.libero_max_tasks)
        max_steps = _libero_max_steps(suite_name)

        total_episodes = 0
        total_successes = 0
        for task_id in range(num_tasks):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            task_description = task.language
            task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            env = OffScreenRenderEnv(
                bddl_file_name=task_bddl_file,
                camera_heights=LIBERO_ENV_RESOLUTION,
                camera_widths=LIBERO_ENV_RESOLUTION,
            )
            env.seed(config.seed)

            task_episodes = 0
            task_successes = 0
            for episode_idx in range(config.libero_num_trials_per_task):
                env.reset()
                obs = env.set_init_state(initial_states[episode_idx])
                action_plan = collections.deque()
                replay_images = []
                done = False
                t = 0

                while t < max_steps + config.libero_num_steps_wait:
                    if t < config.libero_num_steps_wait:
                        obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, config.libero_resize_size, config.libero_resize_size))
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, config.libero_resize_size, config.libero_resize_size)
                    )
                    if video_dir is not None:
                        replay_images.append(img)

                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"].copy()),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }
                        action_chunk = policy.infer(element)["actions"]
                        if len(action_chunk) < config.libero_replan_steps:
                            raise ValueError(
                                f"Policy only predicted {len(action_chunk)} actions, expected at least {config.libero_replan_steps}."
                            )
                        action_plan.extend(action_chunk[: config.libero_replan_steps])

                    action = action_plan.popleft()
                    obs, _, done, _ = env.step(action.tolist())
                    t += 1
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break

                if video_dir is not None and replay_images:
                    suffix = "success" if done else "failure"
                    suite_segment = suite_name.replace(" ", "_")
                    task_segment = task_description.replace(" ", "_")
                    imageio.mimwrite(
                        video_dir / f"{suite_segment}_task{task_id:02d}_ep{episode_idx:03d}_{task_segment}_{suffix}.mp4",
                        [np.asarray(x) for x in replay_images],
                        fps=10,
                    )

                task_episodes += 1
                total_episodes += 1
                episode_rows.append(
                    {
                        "suite": suite_name,
                        "task_id": task_id,
                        "task": task_description,
                        "episode_idx": episode_idx,
                        "success": float(done),
                        "steps": t,
                    }
                )

            suite_summaries.append(
                {
                    "suite": suite_name,
                    "episodes": total_episodes,
                    "successes": total_successes,
                    "success_rate": float(total_successes / total_episodes) if total_episodes else 0.0,
                }
            )

        LOGGER.info("LIBERO suite %s success rate: %.4f", suite_name, suite_summaries[-1]["success_rate"])

    write_csv(output_dir / "origin_pi05_libero_rollout.csv", episode_rows)
    return {
        "suites": suite_summaries,
    }


def main(config: BenchmarkConfig) -> None:
    init_logging()
    output_dir = prepare_output_dir(config)
    (output_dir / "config.json").write_text(json.dumps(dataclasses.asdict(config), indent=2))

    summary: dict[str, Any] = {
        "origin_pi05_dataset": None,
        "pruned_prefix_dataset": None,
        "origin_pi05_libero": None,
        "notes": [],
    }

    if config.run_dataset_benchmark:
        LOGGER.info("Starting offline dataset benchmark.")
        summary["origin_pi05_dataset"] = benchmark_origin_dataset(config, output_dir)
        summary["pruned_prefix_dataset"] = benchmark_pruned_dataset(config, output_dir)
        if config.pruned_student_checkpoint is None:
            summary["notes"].append("Pruned prefix checkpoint not provided, so only origin dataset metrics were run.")
    else:
        summary["notes"].append("Offline dataset benchmark disabled.")

    if config.run_libero_rollout:
        LOGGER.info("Starting LIBERO rollout benchmark.")
        summary["origin_pi05_libero"] = benchmark_origin_libero(config, output_dir)
        if config.pruned_student_checkpoint is not None:
            summary["notes"].append(
                "Pruned student rollout success is not supported yet because the saved student checkpoint is only a distilled prefix model, not a full pi05 action policy."
            )
    else:
        summary["notes"].append("LIBERO rollout benchmark disabled.")

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    LOGGER.info("Benchmark results written to %s", output_dir)


if __name__ == "__main__":
    main(tyro.cli(BenchmarkConfig))
