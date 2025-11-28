import dataclasses
import functools
import json
import logging
import os
import platform
from typing import Any

import boto3
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
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
from openpi.training.experiment_tracker import TrackingBackend
from openpi.training.experiment_tracker import create_tracker
from openpi.training.mlflow_client import MLflowClient
from openpi.training.mlflow_client import MLflowConfig
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def upload_checkpoint_to_s3(local_dir: str, s3_uri: str):
    """
    Recursively uploads a local checkpoint directory to an S3 path.

    Args:
        local_dir: str, Local path, e.g. "/tmp/checkpoints/pi0_pikki/exp_1"
        s3_uri: str, S3 URI, e.g. "s3://robotic-platform-vla/checkpoints/pi0_pikki/exp_1"
    """
    if not s3_uri.startswith("s3://"):
        raise ValueError("Expected S3 URI to start with 's3://'")

    s3 = boto3.client("s3", region_name="us-east-1")

    bucket, *key_parts = s3_uri[len("s3://") :].split("/", 1)
    prefix = key_parts[0] if key_parts else ""

    for root, _, files in os.walk(local_dir):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, local_dir)
            s3_key = os.path.join(prefix, rel_path).replace("\\", "/")

            print(f"[UPLOAD] {full_path} -> s3://{bucket}/{s3_key}")
            s3.upload_file(full_path, bucket, s3_key)


def publish_training_status(model_id: str, status: str, detail: dict | None = None):
    """
    Publish training status to SNS topic.

    Args:
        model_id: str, Model ID
        status: str, Training status (e.g., "SUCCEEDED", "FAILED")
        detail: dict | None, Additional details to include in the message
    """
    sns = boto3.client("sns", region_name="us-east-1")
    SNS_TOPIC_ARN = os.environ.get(  # noqa: N806
        "SNS_TOPIC_ARN", "arn:aws:sns:us-east-1:891376920743:training-status"
    )

    payload = {"model_id": model_id, "status": status, "detail": detail or {}}
    sns.publish(
        TopicArn=SNS_TOPIC_ARN,
        Message=json.dumps(payload),
        MessageAttributes={
            "model_id": {"DataType": "String", "StringValue": model_id},
            "status": {"DataType": "String", "StringValue": status},
        },
        Subject="training-status",
    )


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {
        "DEBUG": "D",
        "INFO": "I",
        "WARNING": "W",
        "ERROR": "E",
        "CRITICAL": "C",
    }

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


def init_experiment_tracker(config: _config.TrainConfig, *, resuming: bool):
    """
    Initialize experiment tracking backend.

    Args:
        config: TrainConfig, Configuration for training.
        resuming: bool, Whether the training is being resumed from a checkpoint.

    Returns:
        ExperimentTracker, Initialized experiment tracker.
    """
    # Determine tracking backend
    if config.wandb_enabled and config.mlflow_enabled:
        logging.warning(
            "[TRACKER] Both wandb and mlflow are enabled. Using mlflow by default. "
            "Set one to False to use the other."
        )
        backend = TrackingBackend.MLFLOW
    elif config.wandb_enabled:
        backend = TrackingBackend.WANDB
    elif config.mlflow_enabled:
        backend = TrackingBackend.MLFLOW
    else:
        backend = TrackingBackend.NONE

    # Initialize wandb if needed
    if backend == TrackingBackend.WANDB:
        wandb_local_dir = epath.Path("./wandb_runs") / config.name / config.exp_name
        wandb_local_dir.mkdir(parents=True, exist_ok=True)
        wandb_id_file = wandb_local_dir / "wandb_id.txt"

        if resuming and wandb_id_file.exists():
            run_id = wandb_id_file.read_text().strip()
            wandb.init(id=run_id, resume="must", project=config.project_name)
        else:
            wandb.init(
                name=config.exp_name,
                config=dataclasses.asdict(config),
                project=config.project_name,
            )
            wandb_id_file.write_text(wandb.run.id)

    # Initialize MLflow client if needed
    mlflow_client = None
    if backend == TrackingBackend.MLFLOW:
        model_id = os.environ.get("MODEL_ID", config.exp_name)
        mlflow_config = MLflowConfig(
            tracking_uri=config.mlflow_tracking_uri,
            experiment_name=config.mlflow_experiment_name or config.project_name,
            run_name=model_id,
            checkpoint_dir=config.checkpoint_dir,
            hyperparameters={
                "seed": config.seed,
                "batch_size": config.batch_size,
                "num_train_steps": config.num_train_steps,
                "learning_rate": config.lr_schedule.peak_lr,
                "warmup_steps": config.lr_schedule.warmup_steps,
                "fsdp_devices": config.fsdp_devices,
                "num_workers": config.num_workers,
            },
            tags={
                "experiment_name": config.exp_name,
                "model_name": config.name,
                "model_id": model_id,
                "asset_id": model_id,
            },
        )
        mlflow_client = MLflowClient(mlflow_config, resuming=resuming)
        mlflow_client.start()

    # Create unified tracker
    return create_tracker(backend, mlflow_client=mlflow_client)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
    *,
    resume: bool,
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(
            params,
            config.freeze_filter,
            lambda p: p.replace(p.value.astype(jnp.bfloat16)),
        )

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
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

    # Filter out params that aren't kernels.
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
    return new_state, info


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
    tracker = init_experiment_tracker(config, resuming=resuming)

    success = False
    try:
        data_loader = _data_loader.create_data_loader(
            config,
            sharding=data_sharding,
            num_workers=config.num_workers,
            shuffle=True,
        )
        data_iter = iter(data_loader)
        batch = next(data_iter)
        logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

        images_to_log = [
            np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1)
            for i in range(min(5, len(next(iter(batch[0].images.values())))))
        ]
        tracker.log_images(images_to_log, "camera_views", step=0)

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
                pbar.write(f"Step {step}: {info_str}")
                tracker.log_metrics(reduced_info, step=step)
                infos = []
            batch = next(data_iter)
            if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
                _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

                latest_step_dir = config.checkpoint_dir / str(step)
                run_on_aws = os.environ.get("AWS_BATCH_JOB_ID")
                if run_on_aws:
                    checkpoint_manager.wait_until_finished()
                    model_id = os.environ.get("MODEL_ID", "model_id")
                    s3_path = f"s3://robotic-platform-vla/dev/models/{model_id}/{config.name}/{config.exp_name}/{step}"
                    upload_checkpoint_to_s3(
                        local_dir=str(latest_step_dir),
                        s3_uri=s3_path,
                    )
                else:
                    logging.warning("Running locally.")

        logging.info("Waiting for checkpoint manager to finish")
        checkpoint_manager.wait_until_finished()

        success = True
    finally:
        status = "FINISHED" if success else "FAILED"
        tracker.end_run(status=status)


if __name__ == "__main__":
    config = _config.cli()
    model_id = os.environ.get("MODEL_ID")
    try:
        main(config)
        publish_training_status(model_id, "SUCCEEDED")
    except Exception as e:
        logging.error(e)
        publish_training_status(model_id, "FAILED")
        raise
