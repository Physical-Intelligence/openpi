"""
PyTorch training entrypoint for PI0/PI05 with multi-GPU and multi-node (DDP) support.
This script mirrors the behavior of the JAX trainer (`scripts/train.py`) but runs
entirely in PyTorch using the `PI0Pytorch` model and your existing config/data
pipeline from `src/openpi/training/config.py` and `src/openpi/training/data_loader.py`.

Usage
Single GPU:
  python scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>
  Example:
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test --resume  # Resume from latest checkpoint
Multi-GPU (single node):
  torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>
  Example:
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume
Multi-Node Training:
	torchrun \
    --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> --node_rank=<rank_of_node> \
    --master_addr=<master_ip> --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>

"""

import dataclasses
import gc
import json
import logging
import math
import os
import platform
import shutil
import time

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
from torch.utils.tensorboard import SummaryWriter
import tqdm
import wandb

import openpi.models.pi0_config
import openpi.models_pytorch.lora as _lora
import openpi.models_pytorch.pi0_pytorch
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data
from openpi.training.reward_checkpoints import RewardCheckpointStore


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
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    """Initialize wandb logging."""
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
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")

        # Set up debugging environment variables for DDP issues
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def cleanup_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


def build_datasets(config: _config.TrainConfig):
    # Use the unified data loader with PyTorch framework
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    return data_loader, data_loader.data_config()


def build_reward_batches(config: _config.TrainConfig):
    """Materialize a deterministic, episode-spanning offline scoring set."""
    loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
    if not isinstance(loader, _data.DataLoaderImpl):
        raise TypeError("Offline reward checkpointing requires the finite PyTorch dataset loader")
    batches = loader.evenly_spaced_batches(
        num_batches=config.checkpoint_reward_batches,
        batch_size=config.batch_size,
    )
    if not batches:
        raise RuntimeError("Cannot checkpoint by reward without evaluation batches")
    return batches


def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


def trainable_parameters(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def evaluate_offline_reward(model, batches, device, *, seed: int) -> tuple[float, float]:
    """Return deterministic imitation reward and loss; this is not task reward."""
    target = unwrap_model(model)
    was_training = target.training
    target.eval()
    losses: list[float] = []
    fork_devices = [device] if device.type == "cuda" else []
    try:
        with torch.no_grad(), torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            for cpu_observation, cpu_actions in batches:
                observation = jax.tree.map(lambda value: value.to(device), cpu_observation)
                actions = cpu_actions.to(device=device, dtype=torch.float32)
                batch_losses = target(observation, actions)
                if isinstance(batch_losses, list | tuple):
                    batch_losses = torch.stack(batch_losses)
                losses.append(float(batch_losses.float().mean().item()))
    finally:
        target.train(was_training)
    mean_loss = sum(losses) / len(losses)
    return math.exp(-mean_loss), mean_loss


def write_checkpoint_contents(
    checkpoint_dir,
    *,
    model,
    optimizer,
    global_step,
    reward,
    reward_loss,
    config,
    data_config,
):
    target = unwrap_model(model)
    if config.pytorch_lora_rank is not None:
        state = _lora.trainable_state_dict(target)
        safetensors.torch.save_file(state, checkpoint_dir / "model.safetensors")
        checkpoint_kind = "pytorch_lora_adapter"
    else:
        safetensors.torch.save_model(target, checkpoint_dir / "model.safetensors")
        checkpoint_kind = "pytorch_full_model"

    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    metadata = {
        "global_step": global_step,
        "config": dataclasses.asdict(config),
        "timestamp": time.time(),
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_metric": "offline_imitation_reward",
        "checkpoint_reward": reward,
        "offline_imitation_loss": reward_loss,
    }
    torch.save(metadata, checkpoint_dir / "metadata.pt")

    norm_stats = data_config.norm_stats
    if norm_stats is not None and data_config.asset_id is not None:
        _normalize.save(checkpoint_dir / "assets" / data_config.asset_id, norm_stats)


def load_checkpoint(model, optimizer, checkpoint_store, device):
    """Resume from the highest-reward checkpoint."""
    ckpt_dir = checkpoint_store.best_checkpoint_dir()
    if ckpt_dir is None:
        raise FileNotFoundError(f"No reward checkpoints found in {checkpoint_store.root}")
    latest_step = int(ckpt_dir.name)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    try:
        logging.info("Loading highest-reward model state from %s", ckpt_dir)
        safetensors_path = ckpt_dir / "model.safetensors"
        if safetensors_path.exists():
            target = unwrap_model(model)
            state = safetensors.torch.load_file(safetensors_path, device=str(device))
            if any(name.endswith(("lora_a", "lora_b")) for name in state):
                _lora.load_trainable_state_dict(target, state)
            else:
                safetensors.torch.load_model(target, safetensors_path, device=str(device))
            logging.info("Loaded reward-selected model state")
        else:
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_model")

        # Load optimizer state with error handling
        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"

        if optimizer_path.exists():
            optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
            logging.info("Loaded optimizer state from pt format")
        else:
            raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_optimizer")

        # Load metadata
        logging.info("Loading metadata...")
        metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
        global_step = metadata.get("global_step", latest_step)
        del metadata
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_metadata")

        logging.info(f"Successfully loaded reward checkpoint components from step {latest_step}")
        return global_step

    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clear memory and provide detailed error message
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {e!s}")
            log_memory_usage(device, latest_step, "after_oom_error")
            raise RuntimeError(
                "Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            ) from e
        raise
def log_memory_usage(device, step, phase="unknown"):
    """Log detailed memory usage information."""
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    memory_free = memory_free / 1e9

    # Get more detailed memory info
    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    # Get DDP info if available
    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )


def train_loop(config: _config.TrainConfig):
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)

    # Initialize checkpoint directory and wandb.
    resuming = False
    if config.resume:
        exp_checkpoint_dir = config.checkpoint_dir
        if not exp_checkpoint_dir.exists():
            raise FileNotFoundError(f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume")
        resuming = True
    elif config.overwrite and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
        logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")

    # Create checkpoint directory with experiment name
    if not resuming:
        # For new runs, create experiment-specific checkpoint directory
        exp_checkpoint_dir = config.checkpoint_dir
        exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created experiment checkpoint directory: {exp_checkpoint_dir}")
    else:
        logging.info(f"Using existing experiment checkpoint directory: {config.checkpoint_dir}")

    checkpoint_store = RewardCheckpointStore(
        config.checkpoint_dir,
        metric_name="offline_imitation_reward",
        minimum_reward=config.checkpoint_reward_min,
        minimum_delta=config.checkpoint_reward_min_delta,
        minimum_free_bytes=int(config.checkpoint_min_free_disk_gib * 1024**3),
        minimum_free_fraction=config.checkpoint_min_free_disk_fraction,
    )
    if resuming:
        best = checkpoint_store.best_record()
        if best is None:
            raise FileNotFoundError(f"No reward checkpoint found in {config.checkpoint_dir}")
        logging.info(
            "Resuming from best reward checkpoint: step=%d reward=%.8f",
            int(best["step"]),
            float(best["reward"]),
        )

    # Initialize wandb (only on main process)
    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # Build data loader using the unified data loader
    # Calculate effective batch size per GPU for DDP
    # For N GPUs, each GPU should get batch_size/N samples, so total across all GPUs is batch_size
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    effective_batch_size = config.batch_size // world_size
    logging.info(
        f"Using batch size per GPU: {effective_batch_size} (total batch size across {world_size} GPUs: {config.batch_size})"
    )

    # Pass the original batch size to data loader - it will handle DDP splitting internally
    loader, data_config = build_datasets(config)
    reward_batches = build_reward_batches(config)

    # Log sample images to wandb on first batch
    if is_main and config.wandb_enabled and not resuming:
        # Create a separate data loader for sample batch to avoid consuming the main loader
        sample_data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
        sample_batch = next(iter(sample_data_loader))
        # Convert observation and actions to torch tensors
        observation, actions = sample_batch
        sample_batch = observation.to_dict()
        sample_batch["actions"] = actions

        # Create sample images for wandb
        images_to_log = []
        # Get batch size from the first image tensor
        batch_size = next(iter(sample_batch["image"].values())).shape[0]
        for i in range(min(5, batch_size)):
            # Concatenate all camera views horizontally for this batch item
            # Convert from NCHW to NHWC format for wandb
            img_concatenated = torch.cat([img[i].permute(1, 2, 0) for img in sample_batch["image"].values()], axis=1)
            img_concatenated = img_concatenated.cpu().numpy()
            images_to_log.append(wandb.Image(img_concatenated))

        wandb.log({"camera_views": images_to_log}, step=0)

        # Clear sample batch from memory aggressively
        del sample_batch, observation, actions, images_to_log, img_concatenated
        del sample_data_loader  # Also delete the sample data loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Cleared sample batch and data loader from memory")

    # Build model
    if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        # Convert dataclass to Pi0Config if needed
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = config.model
        # Update dtype to match pytorch_training_precision
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)

    # Always load the immutable base weights before inserting LoRA wrappers.
    if config.pytorch_weight_path is not None:
        logging.info(f"Loading base weights from: {config.pytorch_weight_path}")
        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        safetensors.torch.load_model(model, model_path)
        logging.info(f"Loaded PyTorch base weights from {config.pytorch_weight_path}")

    lora_modules: list[str] = []
    if config.pytorch_lora_rank is not None:
        action_expert_rank = config.pytorch_lora_action_expert_rank or config.pytorch_lora_rank
        lora_modules = _lora.apply_pi0_lora(
            model,
            paligemma_rank=config.pytorch_lora_rank,
            action_expert_rank=action_expert_rank,
            paligemma_alpha=config.pytorch_lora_alpha,
            action_expert_alpha=config.pytorch_lora_action_expert_alpha,
            dropout=config.pytorch_lora_dropout,
            train_action_heads=config.pytorch_lora_train_action_heads,
        )
        logging.info("Inserted LoRA into %d transformer linear layers", len(lora_modules))

    total_parameters, trainable_parameter_count = _lora.parameter_counts(model)
    if trainable_parameter_count == 0:
        raise RuntimeError("Training configuration froze every model parameter")
    logging.info(
        "Parameters: total=%d trainable=%d trainable_fraction=%.6f",
        total_parameters,
        trainable_parameter_count,
        trainable_parameter_count / total_parameters,
    )

    if hasattr(model, "gradient_checkpointing_enable"):
        enable_gradient_checkpointing = True
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing for memory optimization")
    else:
        enable_gradient_checkpointing = False
        logging.info("Gradient checkpointing is not supported for this model")

    # Log initial memory usage after model creation
    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    # Enable memory optimizations for large-scale training
    if world_size >= 8:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set memory allocation configuration
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        logging.info("Enabled memory optimizations for 8+ GPU training")

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=world_size >= 8,  # Enable for 8+ GPUs
        )

    # Optimizer + learning rate schedule from config
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    # Create optimizer with config parameters
    optimized_parameters = trainable_parameters(model)
    optim = torch.optim.AdamW(
        optimized_parameters,
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    # Load checkpoint if resuming
    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optim, checkpoint_store, device)
        logging.info(f"Resumed training from step {global_step}")

    def lr_schedule(step: int):
        if step < warmup_steps:
            # Match JAX behavior: start from peak_lr / (warmup_steps + 1)
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        # cosine decay
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    model.train()
    training_started_at = time.time()
    deadline_epoch = (
        training_started_at + config.max_train_seconds
        if config.max_train_seconds is not None
        else None
    )
    tensorboard_dir = config.checkpoint_dir / "tensorboard"
    tensorboard = SummaryWriter(log_dir=str(tensorboard_dir), purge_step=global_step) if is_main else None
    state_path = config.checkpoint_dir / "run_state.json"
    last_reward = None
    last_reward_loss = None
    last_reward_eval_step = None

    def write_run_state(state: str, **extra):
        if not is_main:
            return
        payload = {
            "state": state,
            "pid": os.getpid(),
            "global_step": global_step,
            "training_started_epoch": training_started_at,
            "deadline_epoch": deadline_epoch,
            "tensorboard_dir": str(tensorboard_dir),
            "checkpoint_metric": "offline_imitation_reward",
            "checkpoint_metric_is_task_reward": False,
            "last_offline_imitation_reward": last_reward,
            "last_offline_imitation_loss": last_reward_loss,
            "last_reward_eval_step": last_reward_eval_step,
            "best_reward_checkpoint": checkpoint_store.best_record(),
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameter_count,
            **extra,
        }
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(state_path)

    baseline_reward, baseline_reward_loss = evaluate_offline_reward(
        model,
        reward_batches,
        device,
        seed=config.seed + 10_000,
    )
    last_reward = baseline_reward
    last_reward_loss = baseline_reward_loss
    last_reward_eval_step = global_step
    if checkpoint_store.best_record() is None:
        checkpoint_store.minimum_reward = max(
            checkpoint_store.minimum_reward,
            baseline_reward + config.checkpoint_reward_min_delta,
        )
    if is_main:
        logging.info(
            "Offline checkpoint baseline: reward=%.8f loss=%.8f (not simulator task reward)",
            baseline_reward,
            baseline_reward_loss,
        )
        if tensorboard is not None:
            tensorboard.add_scalar("eval/offline_imitation_reward", baseline_reward, global_step)
            tensorboard.add_scalar("eval/offline_imitation_loss", baseline_reward_loss, global_step)
            tensorboard.flush()
    write_run_state("TRAINING_RUNNING")
    start_time = time.time()
    infos = []  # Collect stats over log interval
    if is_main:
        logging.info(
            f"Running on: {platform.node()} | world_size={torch.distributed.get_world_size() if use_ddp else 1}"
        )
        logging.info(
            f"Training config: batch_size={config.batch_size}, effective_batch_size={effective_batch_size}, num_train_steps={config.num_train_steps}"
        )
        logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
        logging.info(
            f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}"
        )
        logging.info(
            f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}"
        )
        logging.info("EMA is not supported for PyTorch training")
        logging.info(f"Training precision: {model_cfg.dtype}")
        logging.info(
            "LoRA: modules=%d trainable=%d/%d (%.4f%%)",
            len(lora_modules),
            trainable_parameter_count,
            total_parameters,
            100.0 * trainable_parameter_count / total_parameters,
        )

    def evaluate_and_maybe_checkpoint():
        nonlocal last_reward, last_reward_loss, last_reward_eval_step
        reward, reward_loss = evaluate_offline_reward(
            model,
            reward_batches,
            device,
            seed=config.seed + 10_000,
        )
        last_reward = reward
        last_reward_loss = reward_loss
        last_reward_eval_step = global_step
        if not is_main:
            return None
        result = checkpoint_store.maybe_save(
            step=global_step,
            reward=reward,
            write_checkpoint=lambda checkpoint_dir: write_checkpoint_contents(
                checkpoint_dir,
                model=model,
                optimizer=optim,
                global_step=global_step,
                reward=reward,
                reward_loss=reward_loss,
                config=config,
                data_config=data_config,
            ),
        )
        logging.info(
            "Offline checkpoint evaluation: step=%d reward=%.8f loss=%.8f saved=%s reason=%s",
            global_step,
            reward,
            reward_loss,
            result.saved,
            result.reason,
        )
        if tensorboard is not None:
            tensorboard.add_scalar("eval/offline_imitation_reward", reward, global_step)
            tensorboard.add_scalar("eval/offline_imitation_loss", reward_loss, global_step)
            tensorboard.add_scalar("checkpoint/saved", float(result.saved), global_step)
            if result.best_reward is not None:
                tensorboard.add_scalar("checkpoint/best_reward", result.best_reward, global_step)
            tensorboard.flush()
        if config.wandb_enabled:
            wandb.log(
                {
                    "offline_imitation_reward": reward,
                    "offline_imitation_loss": reward_loss,
                    "checkpoint_saved": int(result.saved),
                },
                step=global_step,
            )
        write_run_state("TRAINING_RUNNING")
        return result

    # Training loop - iterate until we reach num_train_steps
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
        if is_main
        else None
    )

    deadline_reached = False
    while global_step < config.num_train_steps and not deadline_reached:
        # Set epoch for distributed training
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))

        for observation, actions in loader:
            # Check if we've reached the target number of steps
            if global_step >= config.num_train_steps:
                break
            if deadline_epoch is not None and time.time() >= deadline_epoch:
                deadline_reached = True
                break

            # The unified data loader returns (observation, actions) tuple
            observation = jax.tree.map(lambda x: x.to(device), observation)  # noqa: PLW2901
            actions = actions.to(torch.float32)  # noqa: PLW2901
            actions = actions.to(device)  # noqa: PLW2901

            # Update LR
            for pg in optim.param_groups:
                pg["lr"] = lr_schedule(global_step)

            # Forward pass
            losses = model(observation, actions)
            # Ensure losses is a tensor and handle different return types
            if isinstance(losses, list | tuple):
                losses = torch.stack(losses)
            elif not isinstance(losses, torch.Tensor):
                losses = torch.tensor(losses, device=device, dtype=torch.float32)

            loss = losses.mean()

            # Backward pass
            loss.backward()

            # Log memory usage after backward pass
            if global_step < 5 and is_main and torch.cuda.is_available():
                log_memory_usage(device, global_step, "after_backward")

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                optimized_parameters, max_norm=config.optimizer.clip_gradient_norm
            )

            # Optimizer step
            optim.step()
            optim.zero_grad(set_to_none=True)

            # Clear gradients more aggressively
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad = None

            # Collect stats
            if is_main:
                infos.append(
                    {
                        "loss": loss.item(),
                        "learning_rate": optim.param_groups[0]["lr"],
                        "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    }
                )

            if is_main and (global_step % config.log_interval == 0):
                elapsed = time.time() - start_time

                # Average stats over log interval
                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)

                avg_grad_norm = None
                if any("grad_norm" in info for info in infos):
                    vals = [
                        info["grad_norm"] for info in infos if "grad_norm" in info and info["grad_norm"] is not None
                    ]
                    if len(vals) > 0:
                        avg_grad_norm = sum(vals) / len(vals)
                logging.info(
                    f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} grad_norm={avg_grad_norm:.2f} time={elapsed:.1f}s"
                    if avg_grad_norm is not None
                    else f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} time={elapsed:.1f}s"
                )

                # Log to wandb
                if config.wandb_enabled and len(infos) > 0:
                    log_payload = {
                        "loss": avg_loss,
                        "learning_rate": avg_lr,
                        "step": global_step,
                        "time_per_step": elapsed / config.log_interval,
                    }
                    if avg_grad_norm is not None:
                        log_payload["grad_norm"] = avg_grad_norm
                    wandb.log(log_payload, step=global_step)

                if tensorboard is not None:
                    tensorboard.add_scalar("train/loss", avg_loss, global_step)
                    tensorboard.add_scalar("train/learning_rate", avg_lr, global_step)
                    tensorboard.add_scalar("train/time_per_step_s", elapsed / max(1, len(infos)), global_step)
                    if avg_grad_norm is not None:
                        tensorboard.add_scalar("train/grad_norm", avg_grad_norm, global_step)
                    tensorboard.flush()
                write_run_state(
                    "TRAINING_RUNNING",
                    loss=avg_loss,
                    learning_rate=avg_lr,
                    grad_norm=avg_grad_norm,
                    uptime_seconds=time.time() - training_started_at,
                )

                start_time = time.time()
                infos = []  # Reset stats collection

            global_step += 1
            if global_step % config.save_interval == 0:
                evaluate_and_maybe_checkpoint()

            # Update progress bar
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "lr": f"{optim.param_groups[0]['lr']:.2e}", "step": global_step}
                )

    if global_step > 0 and last_reward_eval_step != global_step:
        evaluate_and_maybe_checkpoint()
    write_run_state(
        "TRAINING_DEADLINE_COMPLETE" if deadline_reached else "TRAINING_STEPS_COMPLETE",
        uptime_seconds=time.time() - training_started_at,
    )

    # Close progress bar
    if pbar is not None:
        pbar.close()

    # Finish wandb run
    if is_main and config.wandb_enabled:
        wandb.finish()
    if tensorboard is not None:
        tensorboard.close()

    cleanup_ddp()


def main():
    init_logging()
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()
