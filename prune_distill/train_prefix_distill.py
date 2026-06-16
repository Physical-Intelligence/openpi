from __future__ import annotations

import dataclasses
import functools
import json
import logging
import pathlib
import shutil
from typing import Any

from flax import nnx
from flax import struct
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import optax
import orbax.checkpoint as ocp
from torch.utils.tensorboard import SummaryWriter
import tyro

from openpi.models import gemma as gemma_teacher
from openpi.models import gemma_pruning
from openpi.models import model as _model
from openpi.models import pi0 as pi0_model
from openpi.models import pi0_config
from openpi.models import siglip
from openpi.shared import nnx_utils
import openpi.training.config as training_config
import openpi.training.data_loader as data_loader
import openpi.transforms as _transforms


LOGGER = logging.getLogger("prune_distill")


TRAINABLE_FILTER = nnx.All(nnx.Param, nnx_utils.PathRegex("student/.*"))
FROZEN_FILTER = nnx.All(nnx.Param, nnx_utils.PathRegex("(vision|teacher)/.*"))


def teacher_module_kwargs(embed_dtype: str) -> dict[str, Any]:
    teacher_cfg = gemma_teacher.get_config("gemma_2b")
    return {
        "variant": "gemma_2b",
        "width": teacher_cfg.width,
        "depth": teacher_cfg.depth,
        "mlp_dim": teacher_cfg.mlp_dim,
        "num_heads": teacher_cfg.num_heads,
        "num_kv_heads": teacher_cfg.num_kv_heads,
        "head_dim": teacher_cfg.head_dim,
        "norm_eps": 1e-6,
        "vocab_size": gemma_teacher.PALIGEMMA_VOCAB_SIZE,
        "embed_dtype": embed_dtype,
        "scan": True,
        "remat_policy": "nothing_saveable",
    }


@dataclasses.dataclass(frozen=True)
class DistillConfig:
    exp_name: str = "gemma_prefix_distill"
    teacher_checkpoint: str = "/root/pi_train/pi05_libero/params"
    output_dir: str = "/root/openpi-wr/checkpoints/prune_distill"
    train_config_name: str = "pi05_libero"
    dataset_path: str = "/root/flatten_fold_v2"
    norm_stats_assets_dir: str = "/root/pi_train/pi05_libero/assets"
    norm_stats_asset_id: str = "physical-intelligence/libero"
    max_examples: int | None = 50_000
    max_episodes: int | None = None
    batch_size: int = 8
    num_workers: int = 2
    num_train_steps: int = 10_000
    log_interval: int = 20
    save_interval: int = 500
    seed: int = 42
    learning_rate: float = 1e-4
    warmup_steps: int = 200
    weight_decay: float = 1e-2
    hidden_loss_weight: float = 1.0
    cosine_loss_weight: float = 0.1
    dtype: str = "bfloat16"
    resume: bool = False
    overwrite: bool = False


@struct.dataclass
class DistillState:
    step: jax.Array
    params: nnx.State
    model_def: nnx.GraphDef[Any]
    opt_state: optax.OptState
    tx: optax.GradientTransformation = struct.field(pytree_node=False)


class PrefixDistillModel(nnx.Module):
    def __init__(self, config: DistillConfig, rngs: nnx.Rngs):
        student_cfg = gemma_pruning.get_config("gemma_prune")

        self.vision = nnx_bridge.ToNNX(
            siglip.Module(
                num_classes=teacher_module_kwargs(config.dtype)["width"],
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        self.teacher = nnx_bridge.ToNNX(
            gemma_pruning.Module(**teacher_module_kwargs(config.dtype))
        )
        self.student = nnx_bridge.ToNNX(
            gemma_pruning.Module(**student_cfg, embed_dtype=config.dtype)
        )

        fake_image = next(iter(pi0_config.Pi0Config(pi05=True).fake_obs().images.values()))
        self.vision.lazy_init(fake_image, train=False, rngs=rngs)
        self.teacher.lazy_init(rngs=rngs, method="init")
        self.student.lazy_init(rngs=rngs, method="init")

    def _embed_images(
        self, observation: _model.Observation
    ) -> tuple[jax.Array, jax.Array]:
        tokens = []
        masks = []
        for name in _model.IMAGE_KEYS:
            image_tokens, _ = self.vision(observation.images[name], train=False)
            image_tokens = jax.lax.stop_gradient(image_tokens)
            tokens.append(image_tokens)
            masks.append(
                jnp.repeat(
                    observation.image_masks[name][:, None],
                    image_tokens.shape[1],
                    axis=1,
                )
            )
        return jnp.concatenate(tokens, axis=1), jnp.concatenate(masks, axis=1)

    def compute_loss(
        self,
        rng: jax.Array,
        observation: _model.Observation,
        *,
        hidden_loss_weight: float,
        cosine_loss_weight: float,
        train: bool,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        observation = _model.preprocess_observation(rng, observation, train=train)
        image_tokens, image_mask = self._embed_images(observation)

        if observation.tokenized_prompt is None or observation.tokenized_prompt_mask is None:
            raise ValueError("Prefix distillation requires tokenized prompts.")

        input_mask = jnp.concatenate([image_mask, observation.tokenized_prompt_mask], axis=1)
        ar_mask = jnp.zeros((input_mask.shape[1],), dtype=jnp.bool_)
        attn_mask = pi0_model.make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        teacher_hidden, _, _ = self.teacher(
            tokens=observation.tokenized_prompt,
            embedded_prefix=image_tokens,
            positions=positions,
            mask=attn_mask,
            return_prelogits=True,
            deterministic=True,
        )
        teacher_hidden = jax.lax.stop_gradient(teacher_hidden.astype(jnp.float32))

        student_hidden, _, _ = self.student(
            tokens=observation.tokenized_prompt,
            embedded_prefix=image_tokens,
            positions=positions,
            mask=attn_mask,
            return_prelogits=True,
            deterministic=not train,
        )
        student_hidden = student_hidden.astype(jnp.float32)

        valid = input_mask.astype(jnp.float32)
        denom = jnp.maximum(jnp.sum(valid), 1.0)

        sq_error = jnp.square(student_hidden - teacher_hidden)
        hidden_mse = jnp.sum(sq_error * valid[..., None]) / denom

        teacher_norm = teacher_hidden / jnp.maximum(jnp.linalg.norm(teacher_hidden, axis=-1, keepdims=True), 1e-6)
        student_norm = student_hidden / jnp.maximum(jnp.linalg.norm(student_hidden, axis=-1, keepdims=True), 1e-6)
        cosine_dist = 1.0 - jnp.sum(student_norm * teacher_norm, axis=-1)
        cosine_loss = jnp.sum(cosine_dist * valid) / denom

        loss = hidden_loss_weight * hidden_mse + cosine_loss_weight * cosine_loss
        return loss, {
            "hidden_mse": hidden_mse,
            "cosine_loss": cosine_loss,
            "valid_tokens": denom,
        }


def init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def build_data_config(config: DistillConfig) -> training_config.TrainConfig:
    repo_root = _repo_root()
    base = training_config.get_config(config.train_config_name)
    custom_data = dataclasses.replace(
        base.data,
        repo_id=config.dataset_path,
        assets=training_config.AssetsConfig(
            assets_dir=config.norm_stats_assets_dir,
            asset_id=config.norm_stats_asset_id,
        ),
    )
    return dataclasses.replace(
        base,
        data=custom_data,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        assets_base_dir=str(repo_root / "assets"),
        checkpoint_base_dir=str(repo_root / "checkpoints"),
    )


def _dataset_root(config: DistillConfig) -> pathlib.Path:
    return pathlib.Path(config.dataset_path).expanduser().resolve()


def _dataset_repo_id(dataset_root: pathlib.Path) -> str:
    return dataset_root.name


def select_episode_subset(
    config: DistillConfig,
    dataset_meta: lerobot_dataset.LeRobotDatasetMetadata,
) -> list[int] | None:
    if config.max_examples is None and config.max_episodes is None:
        return None

    selected: list[int] = []
    total_examples = 0
    episodes = sorted(dataset_meta.episodes.items())
    for episode_idx, episode in episodes:
        if config.max_episodes is not None and len(selected) >= config.max_episodes:
            break

        length = int(episode["length"])
        if config.max_examples is not None and selected and total_examples + length > config.max_examples:
            break

        selected.append(int(episode_idx))
        total_examples += length

        if config.max_examples is not None and total_examples >= config.max_examples:
            break

    if not selected and episodes:
        first_idx, first_episode = episodes[0]
        selected = [int(first_idx)]
        total_examples = int(first_episode["length"])

    LOGGER.info(
        "Using %d episodes with about %d examples from %s",
        len(selected),
        total_examples,
        _dataset_root(config),
    )
    return selected


def _make_repack_transform(mapping: dict[str, str]) -> _transforms.Group:
    return _transforms.Group(inputs=[_transforms.RepackTransform(mapping)])


def _first_present(keys: set[str], *candidates: str) -> str | None:
    for key in candidates:
        if key in keys:
            return key
    return None


def adapt_data_config_to_dataset(
    data_config: training_config.DataConfig,
    dataset: lerobot_dataset.LeRobotDataset,
) -> training_config.DataConfig:
    raw_keys = set(dataset.features)
    preview_keys = sorted(
        key
        for key in raw_keys
        if key in {"actions", "prompt", "state", "task_index"} or any(token in key for token in ("image", "state", "hand", "head", "camera"))
    )
    LOGGER.info("Detected raw dataset keys: %s", preview_keys[:16])

    base_image_key = _first_present(
        raw_keys,
        "observation/image",
        "observation.image",
        "observation/images/base_0_rgb",
        "observation.images.base_0_rgb",
        "front_head",
        "image",
    )
    wrist_image_key = _first_present(
        raw_keys,
        "observation/wrist_image",
        "observation.wrist_image",
        "observation/images/left_wrist_0_rgb",
        "observation.images.left_wrist_0_rgb",
        "left_hand",
        "right_hand",
        "wrist_image",
    )
    state_key = _first_present(
        raw_keys,
        "observation/state",
        "observation.state",
        "state",
    )

    repack_transforms = data_config.repack_transforms
    if (
        base_image_key == "observation/image"
        and wrist_image_key == "observation/wrist_image"
        and state_key == "observation/state"
    ):
        LOGGER.info("Dataset already uses observation/* LIBERO keys. Disabling repack transform.")
        repack_transforms = _transforms.Group()
    elif base_image_key is not None and wrist_image_key is not None and state_key is not None:
        LOGGER.info(
            "Adapting dataset schema to LIBERO keys: base=%s wrist=%s state=%s",
            base_image_key,
            wrist_image_key,
            state_key,
        )
        mapping = {
            "observation/image": base_image_key,
            "observation/wrist_image": wrist_image_key,
            "observation/state": state_key,
            "actions": "actions",
        }
        if "prompt" in raw_keys or data_config.prompt_from_task:
            mapping["prompt"] = "prompt"
        repack_transforms = _make_repack_transform(mapping)
    else:
        LOGGER.warning(
            "Could not infer a custom dataset image/state schema from keys %s. Keeping the configured repack transform.",
            preview_keys[:16],
        )

    prompt_from_task = data_config.prompt_from_task
    if "prompt" in raw_keys and prompt_from_task:
        LOGGER.info("Dataset already contains a prompt column. Disabling prompt_from_task.")
        prompt_from_task = False

    return dataclasses.replace(
        data_config,
        repack_transforms=repack_transforms,
        prompt_from_task=prompt_from_task,
    )


def create_distill_data_loader(
    train_config: training_config.TrainConfig,
    distill_config: DistillConfig,
):
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    dataset_root = _dataset_root(distill_config)
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(_dataset_repo_id(dataset_root), root=dataset_root)
    selected_episodes = select_episode_subset(distill_config, dataset_meta)

    dataset = lerobot_dataset.LeRobotDataset(
        _dataset_repo_id(dataset_root),
        root=dataset_root,
        episodes=selected_episodes,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(train_config.model.action_horizon)]
            for key in data_config.action_sequence_keys
        },
    )
    data_config = adapt_data_config_to_dataset(data_config, dataset)
    if data_config.prompt_from_task:
        dataset = data_loader.TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    # Prefix distillation only uses images and tokenized prompts, so we can skip dataset-specific
    # state/action normalization and avoid mismatches with custom robot state dimensions.
    dataset = data_loader.transform_dataset(dataset, data_config, skip_norm_stats=True)
    local_batch_size = train_config.batch_size // jax.process_count()
    torch_loader = data_loader.TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        seed=train_config.seed,
        framework="jax",
    )
    return data_loader.DataLoaderImpl(data_config, torch_loader)


def load_checkpoint_params(params_path: str) -> dict[str, Any]:
    LOGGER.info("Loading teacher checkpoint from %s", params_path)
    return _model.restore_params(params_path, restore_type=np.ndarray)


def extract_siglip_params(teacher_params: dict[str, Any]) -> dict[str, Any]:
    return teacher_params["PaliGemma"]["img"]


def extract_teacher_prefix_params(teacher_params: dict[str, Any]) -> dict[str, Any]:
    llm = teacher_params["PaliGemma"]["llm"]
    return {
        "embedder": llm["embedder"],
        "final_norm": llm["final_norm"],
        "layers": {
            "attn": {
                "attn_vec_einsum": llm["layers"]["attn"]["attn_vec_einsum"],
                "kv_einsum": llm["layers"]["attn"]["kv_einsum"],
                "q_einsum": llm["layers"]["attn"]["q_einsum"],
            },
            "mlp": llm["layers"]["mlp"],
            "pre_attention_norm": llm["layers"]["pre_attention_norm"],
            "pre_ffw_norm": llm["layers"]["pre_ffw_norm"],
        },
    }


def extract_student_init_params(teacher_params: dict[str, Any]) -> dict[str, Any]:
    llm = teacher_params["PaliGemma"]["llm"]
    student_cfg = gemma_pruning.get_config("gemma_prune")
    depth = student_cfg.depth
    mlp_dim = student_cfg.mlp_dim
    return {
        "embedder": llm["embedder"],
        "final_norm": llm["final_norm"],
        "layers": {
            "attn": {
                "attn_vec_einsum": {"w": llm["layers"]["attn"]["attn_vec_einsum"]["w"][:depth]},
                "kv_einsum": {"w": llm["layers"]["attn"]["kv_einsum"]["w"][:depth]},
                "q_einsum": {"w": llm["layers"]["attn"]["q_einsum"]["w"][:depth]},
            },
            "mlp": {
                "gating_einsum": llm["layers"]["mlp"]["gating_einsum"][:depth, :, :, :mlp_dim],
                "linear": llm["layers"]["mlp"]["linear"][:depth, :mlp_dim, :],
            },
            "pre_attention_norm": {"scale": llm["layers"]["pre_attention_norm"]["scale"][:depth]},
            "pre_ffw_norm": {"scale": llm["layers"]["pre_ffw_norm"]["scale"][:depth]},
        },
    }


def init_state(config: DistillConfig) -> DistillState:
    rng = jax.random.key(config.seed)
    model = PrefixDistillModel(config, rngs=nnx.Rngs(rng))
    graphdef, state = nnx.split(model)

    teacher_params = load_checkpoint_params(config.teacher_checkpoint)
    state.replace_by_pure_dict(
        {
            "vision": extract_siglip_params(teacher_params),
            "teacher": extract_teacher_prefix_params(teacher_params),
            "student": extract_student_init_params(teacher_params),
        }
    )
    del teacher_params

    state = nnx_utils.state_map(state, FROZEN_FILTER, lambda p: p.replace(p.value.astype(jnp.bfloat16)))
    model = nnx.merge(graphdef, state)
    params = nnx.state(model)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=config.warmup_steps,
        decay_steps=max(config.num_train_steps, config.warmup_steps + 1),
        end_value=config.learning_rate * 0.1,
    )
    tx = optax.adamw(schedule, weight_decay=config.weight_decay)

    return DistillState(
        step=jnp.asarray(0, dtype=jnp.int32),
        params=params,
        model_def=nnx.graphdef(model),
        opt_state=tx.init(params.filter(TRAINABLE_FILTER)),
        tx=tx,
    )


def train_step(
    state: DistillState,
    rng: jax.Array,
    batch: tuple[_model.Observation, jax.Array],
    *,
    hidden_loss_weight: float,
    cosine_loss_weight: float,
) -> tuple[DistillState, dict[str, jax.Array]]:
    model = nnx.merge(state.model_def, state.params)
    observation, _ = batch

    def loss_fn(module: PrefixDistillModel):
        return module.compute_loss(
            rng,
            observation,
            hidden_loss_weight=hidden_loss_weight,
            cosine_loss_weight=cosine_loss_weight,
            train=True,
        )

    diff_state = nnx.DiffState(0, TRAINABLE_FILTER)
    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True, argnums=diff_state)(model)

    trainable_params = state.params.filter(TRAINABLE_FILTER)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, trainable_params)
    new_trainable_params = optax.apply_updates(trainable_params, updates)
    nnx.update(model, new_trainable_params)
    new_params = nnx.state(model)

    info = {
        "loss": loss,
        "hidden_mse": metrics["hidden_mse"],
        "cosine_loss": metrics["cosine_loss"],
        "valid_tokens": metrics["valid_tokens"],
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(new_trainable_params),
    }
    return dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state), info


def count_params(state: nnx.State, filter_: nnx.filterlib.Filter | None = None) -> int:
    subset = state if filter_ is None else state.filter(filter_)
    total = 0
    for value in subset.flat_state().values():
        total += int(np.prod(value.value.shape))
    return total


def key_param_stats(state: nnx.State) -> dict[str, float]:
    wanted = {
        "embed": "student/embedder/input_embedding",
        "q": "student/layers/attn/q_einsum/w",
        "mlp_gate": "student/layers/mlp/gating_einsum",
        "mlp_out": "student/layers/mlp/linear",
        "final_norm": "student/final_norm/scale",
    }
    flat = state.flat_state()
    stats = {}
    for label, path in wanted.items():
        for key, value in flat.items():
            joined = "/".join(str(part) for part in key)
            if joined == path:
                stats[label] = float(jnp.linalg.norm(value.value.astype(jnp.float32)))
                break
    return stats


def extract_student_params(state: DistillState) -> dict[str, Any]:
    model = nnx.merge(state.model_def, state.params)
    return nnx.state(model.student).to_pure_dict()


def save_student_checkpoint(output_dir: pathlib.Path, student_params: dict[str, Any]) -> None:
    ckpt = output_dir / "student"
    if ckpt.exists():
        shutil.rmtree(ckpt)
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(ckpt, {"params": student_params})


def save_resume_checkpoint(
    output_dir: pathlib.Path,
    *,
    step: int,
) -> None:
    (output_dir / "resume_state.json").write_text(json.dumps({"step": step}, indent=2))


def save_step_checkpoint(output_dir: pathlib.Path, state: DistillState) -> None:
    student_params = extract_student_params(state)
    step = int(state.step)
    save_student_checkpoint(output_dir, student_params)
    save_resume_checkpoint(output_dir, step=step)


def _parse_step_dir(step_dir: pathlib.Path) -> int:
    return int(step_dir.name.split("_", 1)[1])


def _latest_step_dir(output_dir: pathlib.Path) -> pathlib.Path | None:
    step_dirs = sorted((p for p in output_dir.glob("step_*") if p.is_dir()), key=_parse_step_dir)
    if not step_dirs:
        return None
    return step_dirs[-1]


def _has_count_field(node: Any) -> bool:
    if hasattr(node, "_fields") and "count" in node._fields:
        return True
    if dataclasses.is_dataclass(node):
        return any(field.name == "count" for field in dataclasses.fields(node))
    return False


def _set_opt_state_step(opt_state: optax.OptState, step: int) -> optax.OptState:
    def replace_count(node: Any) -> Any:
        if hasattr(node, "_fields") and "count" in node._fields:
            return node._replace(count=jnp.asarray(step, dtype=jnp.asarray(node.count).dtype))
        if dataclasses.is_dataclass(node):
            return dataclasses.replace(node, count=jnp.asarray(step, dtype=jnp.asarray(node.count).dtype))
        return node

    return jax.tree.map(replace_count, opt_state, is_leaf=_has_count_field)


def maybe_resume_state(
    config: DistillConfig,
    output_dir: pathlib.Path,
    state: DistillState,
) -> DistillState:
    if not config.resume:
        return state

    step_dir = _latest_step_dir(output_dir)
    if step_dir is None:
        raise FileNotFoundError(f"No step_* checkpoints found under {output_dir} to resume from.")

    student_params = _model.restore_params(step_dir / "student", restore_type=np.ndarray)
    resume_json = step_dir / "resume_state.json"
    if resume_json.exists():
        resumed_step = int(json.loads(resume_json.read_text())["step"])
    else:
        resumed_step = _parse_step_dir(step_dir)
    state.params.replace_by_pure_dict({"student": student_params})
    resumed_state = dataclasses.replace(
        state,
        step=jnp.asarray(resumed_step, dtype=jnp.int32),
        opt_state=_set_opt_state_step(state.opt_state, resumed_step),
    )
    LOGGER.warning(
        "Resumed from %s at step=%d. "
        "Student weights were restored, but optimizer moments were reinitialized.",
        step_dir,
        resumed_step,
    )
    return resumed_state


def prepare_output_dir(config: DistillConfig) -> pathlib.Path:
    output_dir = pathlib.Path(config.output_dir) / config.exp_name
    if output_dir.exists():
        if config.resume and config.overwrite:
            raise ValueError("Cannot use resume and overwrite at the same time.")
        if not config.overwrite:
            if not config.resume:
                raise FileExistsError(f"{output_dir} already exists. Pass --overwrite or --resume.")
        else:
            shutil.rmtree(output_dir)
    elif config.resume:
        raise FileNotFoundError(f"{output_dir} does not exist, so there is nothing to resume.")

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    if not config.resume or not config_path.exists():
        config_path.write_text(json.dumps(dataclasses.asdict(config), indent=2))
    return output_dir


def log_tensorboard_scalars(
    writer: SummaryWriter,
    step: int,
    info: dict[str, float],
    *,
    key_params: dict[str, float] | None = None,
) -> None:
    writer.add_scalar("train/loss", info["loss"], step)
    writer.add_scalar("train/hidden_mse", info["hidden_mse"], step)
    writer.add_scalar("train/cosine_loss", info["cosine_loss"], step)
    writer.add_scalar("train/grad_norm", info["grad_norm"], step)
    writer.add_scalar("train/param_norm", info["param_norm"], step)
    writer.add_scalar("train/valid_tokens", info["valid_tokens"], step)
    if key_params is not None:
        for name, value in key_params.items():
            writer.add_scalar(f"key_params/{name}", value, step)


def main(config: DistillConfig) -> None:
    init_logging()
    LOGGER.info("Starting prefix distillation with config: %s", config)

    output_dir = prepare_output_dir(config)
    tensorboard_dir = output_dir / "tensorboard"
    base_data_config = build_data_config(config)
    train_loader = create_distill_data_loader(base_data_config, config)

    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    try:
        state = init_state(config)
        state = maybe_resume_state(config, output_dir, state)
        start_step = int(state.step)
        batch_iter = iter(train_loader)
        total_params = count_params(state.params)
        frozen_params = count_params(state.params, FROZEN_FILTER)
        trainable_params = count_params(state.params, TRAINABLE_FILTER)
        current_key_params = key_param_stats(state.params)

        LOGGER.info(
            "Params: total=%d frozen=%d trainable=%d",
            total_params,
            frozen_params,
            trainable_params,
        )
        LOGGER.info("Current key params at step=%d: %s", start_step, current_key_params)
        writer.add_text(
            "run/config",
            json.dumps(dataclasses.asdict(config), indent=2),
            start_step,
        )
        writer.add_scalar("params/total", total_params, start_step)
        writer.add_scalar("params/frozen", frozen_params, start_step)
        writer.add_scalar("params/trainable", trainable_params, start_step)
        for name, value in current_key_params.items():
            writer.add_scalar(f"key_params/{name}", value, start_step)

        if start_step >= config.num_train_steps:
            LOGGER.info("Current step %d is already at or beyond num_train_steps=%d.", start_step, config.num_train_steps)
            writer.flush()
            return

        ptrain_step = jax.jit(
            functools.partial(
                train_step,
                hidden_loss_weight=config.hidden_loss_weight,
                cosine_loss_weight=config.cosine_loss_weight,
            ),
            donate_argnums=(0,),
        )

        base_rng = jax.random.key(config.seed + 1)
        for _ in range(start_step, config.num_train_steps):
            batch = next(batch_iter)
            step_rng = jax.random.fold_in(base_rng, int(state.step))
            state, info = ptrain_step(state, step_rng, batch)

            step_num = int(state.step)
            host_info = jax.tree.map(lambda x: float(x), info)
            log_tensorboard_scalars(writer, step_num, host_info)

            if step_num % config.log_interval == 0 or step_num == 1:
                key_params = key_param_stats(state.params)
                for name, value in key_params.items():
                    writer.add_scalar(f"key_params/{name}", value, step_num)
                LOGGER.info(
                    "step=%d loss=%.6f hidden_mse=%.6f cosine=%.6f grad_norm=%.6f param_norm=%.6f key_params=%s",
                    step_num,
                    host_info["loss"],
                    host_info["hidden_mse"],
                    host_info["cosine_loss"],
                    host_info["grad_norm"],
                    host_info["param_norm"],
                    key_params,
                )

            if step_num % config.save_interval == 0 or step_num == config.num_train_steps:
                step_dir = output_dir / f"step_{step_num:07d}"
                step_dir.mkdir(parents=True, exist_ok=True)
                save_step_checkpoint(step_dir, state)
                writer.flush()

        final_key_params = key_param_stats(state.params)
        LOGGER.info("Finished distillation. Final key params: %s", final_key_params)
        for name, value in final_key_params.items():
            writer.add_scalar(f"key_params_final/{name}", value, int(state.step))
        writer.flush()
    finally:
        writer.close()


if __name__ == "__main__":
    main(tyro.cli(DistillConfig))
