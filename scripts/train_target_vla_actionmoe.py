"""Training script for the Pi0TargetVLAActionMoe model (trace-free ablation).

Architectural mirror of ``scripts/train_trace_vla_actionmoe.py`` but with a
**2-stream** trunk (paligemma + action MoE) and no trace stream. The pi05_base
checkpoint provides the paligemma stream weights as-is, and we fan its dense
``mlp_1`` FFN out into the K=5 action MoE experts at
``moe_1/expert_{k}/w{1,2,3}/kernel``. There is no stream-2 weight to copy.

Weight remap from ``pi05_base``:

  - Stream 0 (PaliGemma 2B VLM): loaded as-is.
  - Stream 1 (action MoE): pi05_base's dense ``mlp_1`` FFN is fanned out into
    the K=5 hard-MoE experts. All experts start as identical copies of the
    pi05_base action FFN. The dense FFN uses GELU and the MoE expert uses
    SiLU; the same activation mismatch is accepted in the trace_vla family —
    finetuning bridges it quickly.

The remap operates on a flat-dict copy of the raw pi05_base checkpoint
(loaded via ``_model.restore_params`` to bypass ``loader.load``'s model-shape
filter, which would otherwise drop ``mlp_1`` before we can fan it out). After
the manipulations we filter against the model's reference state and rely on
``update_params`` to overlay onto the fresh model init.
"""
from __future__ import annotations

import dataclasses
import functools
import logging
import platform

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.models.target_observation as _target_obs
import openpi.shared.array_typing as at
import openpi.shared.download as _download
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return
    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(name=config.exp_name, config=dataclasses.asdict(config), project=config.project_name)
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)
    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


# ---------------------------------------------------------------------------
# Weight loading: pi05_base -> Pi0TargetVLAActionMoe
# ---------------------------------------------------------------------------

def _load_and_filter_weights_target_actionmoe(loader, params_shape, num_action_experts: int = 5):
    """Load pi05_base weights into the target_vla_actionmoe model's parameter tree.

    Strategy:
      1. Load pi05_base raw via ``_model.restore_params`` (bypass ``loader.load``'s
         model-shape filter so ``mlp_1`` survives the round-trip — the new model
         has *no* ``mlp_1`` key, so the standard filter would drop it before we
         could fan it out).
      2. Fan out the dense ``mlp_1`` FFN into K experts at
         ``moe_1/expert_{k}/w{1,2,3}`` (the action MoE). All experts start as
         identical copies of the pi05_base action FFN.
      3. Filter the resulting flat dict against the model's reference parameters:
         keep only keys that exist in the model; back-fill LoRA params (which are
         not in pi05_base) from the model's reference state.

    There is no second non-VLM stream, so unlike ``train_trace_vla_actionmoe.py``
    we don't replicate stream-1 attention/norm weights into a stream 2. The
    target Fourier MLP, the action time MLP, the action I/O projections, and
    the completion head are all randomly initialized (they don't exist in
    pi05_base, so the model's reference state — pure random init — survives
    the overlay below).
    """
    # 1. Load pi05_base raw (not model-shape-filtered).
    raw_path = loader.params_path
    raw_params = _model.restore_params(_download.maybe_download(raw_path), restore_type=np.ndarray)
    flat_loaded = dict(traverse_util.flatten_dict(raw_params))

    # 2. mlp_1 -> moe_1/expert_*/w{1,2,3}.
    # pi05_base's dense FFN ``mlp_1`` uses two params:
    #   - ``gating_einsum`` of shape (L, 2, in, hidden) for the GeGLU (gate, value)
    #   - ``linear`` of shape (L, hidden, in)
    # The action MoE expert is SwiGLU with three Dense kernels:
    #   - w1.kernel  (L, in, hidden)  ← gating_einsum[..., 0, :, :]
    #   - w3.kernel  (L, in, hidden)  ← gating_einsum[..., 1, :, :]
    #   - w2.kernel  (L, hidden, in)  ← linear
    # The dense FFN uses GELU on the gate path while the MoE expert uses SiLU.
    # The same activation mismatch is accepted in train_trace_vla*.py —
    # finetuning bridges it quickly.
    gating_keys = [k for k in flat_loaded if k[-2:] == ("mlp_1", "gating_einsum")]
    linear_keys = [k for k in flat_loaded if k[-2:] == ("mlp_1", "linear")]

    for k in gating_keys:
        gating = flat_loaded[k]
        w1 = gating[..., 0, :, :]
        w3 = gating[..., 1, :, :]
        prefix = k[:-1]  # path up to "mlp_1"
        for e in range(num_action_experts):
            flat_loaded[(*prefix[:-1], "moe_1", f"expert_{e}", "w1", "kernel")] = w1
            flat_loaded[(*prefix[:-1], "moe_1", f"expert_{e}", "w3", "kernel")] = w3

    for k in linear_keys:
        linear = flat_loaded[k]
        prefix = k[:-1]
        for e in range(num_action_experts):
            flat_loaded[(*prefix[:-1], "moe_1", f"expert_{e}", "w2", "kernel")] = linear

    # 3. Filter against model params; back-fill LoRA from the model's reference state
    # (everything not in pi05_base — LoRA adapters, completion head, time MLP,
    # target MLP, I/O projections — stays at random init by virtue of the
    # ``update_params`` overlay).
    flat_ref = traverse_util.flatten_dict(params_shape)
    result: dict = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            ref_dtype = flat_ref[k].dtype
            if hasattr(v, "dtype") and v.dtype != ref_dtype:
                result[k] = v.astype(ref_dtype)
            else:
                result[k] = v
    # LoRA back-fill (the only path-segments that won't be in pi05_base).
    for k in flat_ref:
        if k in result:
            continue
        if any("lora" in str(seg) for seg in k):
            result[k] = flat_ref[k]

    return traverse_util.unflatten_dict(result)


def update_params(orig_params, partial_params):
    for k, v in partial_params.items():
        if isinstance(v, dict):
            if k not in orig_params:
                orig_params[k] = {}
            orig_params[k] = update_params(orig_params.get(k, {}), v)
        else:
            orig_params[k] = v
    return orig_params


# ---------------------------------------------------------------------------
# Train init / step
# ---------------------------------------------------------------------------

@at.typecheck
def init_train_state(config, init_rng, mesh, *, resume: bool):
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng, partial_params=None):
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)

        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state_dict = state.to_pure_dict() if hasattr(state, "to_pure_dict") else dict(state)
            updated_state_dict = update_params(state_dict, partial_params)
            state.replace_by_pure_dict(updated_state_dict)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        params = nnx_utils.state_map(
            params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16))
        )
        opt_state = tx.init(params.filter(config.trainable_filter))
        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=opt_state,
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng, None)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    # Build fresh model to get its params shape, then apply weight remap.
    model = config.model.create(init_rng)
    params = nnx.state(model)
    params_dict = params.to_pure_dict() if hasattr(params, "to_pure_dict") else dict(params)
    num_action_experts = int(getattr(config.model, "num_action_experts", 5))
    partial_params = _load_and_filter_weights_target_actionmoe(
        config.weight_loader, params_dict, num_action_experts=num_action_experts
    )

    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    train_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)
    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_target_obs.TargetObservation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(model, rng, observation, actions):
        per_sample, info = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(per_sample), info

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, train_info), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params,
                new_params,
            ),
        )

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    info.update(jax.tree.map(jnp.mean, train_info))
    return new_state, info


# ---------------------------------------------------------------------------
# Target data loader (TargetObservation-aware)
# ---------------------------------------------------------------------------

def _create_target_data_loader(config: _config.TrainConfig, *, sharding: jax.sharding.Sharding | None, shuffle: bool, num_batches=None, seed: int = 0):
    """Build a TargetObservation-aware data loader using the existing TorchDataLoader."""
    from openpi.policies.libero_target_dataset import LiberoTargetDataset  # noqa: PLC0415

    if not isinstance(config.data, _config.LeRobotTargetVLAActionMoeDataConfig):
        raise TypeError(
            f"train_target_vla_actionmoe requires LeRobotTargetVLAActionMoeDataConfig, got "
            f"{type(config.data).__name__}"
        )
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    dataset = LiberoTargetDataset(data_config, action_horizon=config.model.action_horizon)
    # Standard norm-stats workflow: run
    #   python pace/openpi/scripts/compute_norm_stats.py --config-name target_vla_actionmoe
    # (and / or target_vla_actionmoe_lora) once before training. The factory writes
    # norm stats under assets/<repo_id>/.
    transformed = _data_loader.transform_dataset(dataset, data_config, skip_norm_stats=False)

    local_batch_size = config.batch_size // jax.process_count()
    torch_loader = _data_loader.TorchDataLoader(
        transformed,
        local_batch_size=local_batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=seed,
    )

    class _Wrapper:
        def __init__(self, loader, dc):
            self._loader = loader
            self._dc = dc

        def data_config(self):
            return self._dc

        def __iter__(self):
            for batch in self._loader:
                yield _target_obs.TargetObservation.from_dict(batch), batch["actions"]

    return _Wrapper(torch_loader, data_config)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _create_target_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
        seed=config.seed,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Train at step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)
        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
