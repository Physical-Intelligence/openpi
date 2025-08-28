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

import argparse
import dataclasses
import gc
import logging
import os
import platform
import shutil
import time
from typing import Any, Dict

import jax
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import tqdm
import wandb
import safetensors.torch

import openpi.training.config as _config
import openpi.training.data_loader as _data
import openpi.models.model as _model
import openpi.models_pytorch.pi0_pytorch
import openpi.models.pi0_config


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


def batch_to_torch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
	# Memory-efficient conversion: convert to torch tensors and move to device in one step
	batch = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(device), batch)

	# Convert to float32 for memory efficiency (avoid float64)
	batch['state'] = batch['state'].to(dtype=torch.float32)
	batch['actions'] = batch['actions'].to(dtype=torch.float32)

	return batch


def get_model_state_dict(model):
	"""Get state dict from model, handling DDP wrapper."""
	return model.module.state_dict() if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.state_dict()


def get_model_parameters(model):
	"""Get parameters from model, handling DDP wrapper."""
	return model.module.parameters() if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.parameters()


def save_checkpoint(model, optimizer, global_step, config, is_main, ema_model=None):
	"""Save a checkpoint with model state, optimizer state, EMA state, and metadata."""
	if not is_main:
		return

	# Only save if it's time to save or if it's the final step
	if (global_step % config.save_interval == 0 and global_step > 0) or global_step == config.num_train_steps - 1:
		# Ensure checkpoint_dir is a Path object and create the step-specific directory
		ckpt_dir = config.checkpoint_dir / f"{global_step}"
		ckpt_dir.mkdir(parents=True, exist_ok=True)

		# Save model state
		state_dict = get_model_state_dict(model)
		torch.save(state_dict, ckpt_dir / "pytorch_model.pt")

		# Save optimizer state
		torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")

		# Save EMA state if available
		if ema_model is not None:
			torch.save(ema_model.state_dict(), ckpt_dir / "ema_model.pt")

		# Save training metadata (avoid saving full config to prevent JAX/Flax compatibility issues)
		metadata = {
			"global_step": global_step,
			"config": dataclasses.asdict(config),
			"timestamp": time.time(),
		}
		torch.save(metadata, ckpt_dir / "metadata.pt")

		logging.info(f"Saved checkpoint at step {global_step} -> {ckpt_dir}")

		# Log checkpoint to wandb
		if config.wandb_enabled:
			wandb.log({"checkpoint_step": global_step}, step=global_step)


def load_checkpoint(model, optimizer, checkpoint_dir, device, ema_model=None):
    """Load the latest checkpoint and return the global step."""
    checkpoint_steps = []
    for d in checkpoint_dir.iterdir():
        if d.is_dir() and d.name.isdigit():
            checkpoint_steps.append(int(d.name))
    
    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    
    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"
    
    # Clear memory before loading checkpoints
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    try:
        # Load model state with error handling
        logging.info("Loading model state...")
        model_state_dict = torch.load(ckpt_dir / "pytorch_model.pt", map_location=device, weights_only=False)
        (model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model).load_state_dict(model_state_dict)
        del model_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_model")
        
        # Load optimizer state with error handling
        logging.info("Loading optimizer state...")
        optimizer_state_dict = torch.load(ckpt_dir / "optimizer.pt", map_location=device, weights_only=False)
        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_optimizer")
        
        # Load EMA state if available
        if ema_model is not None and (ckpt_dir / "ema_model.pt").exists():
            logging.info("Loading EMA state...")
            # Clear as much memory as possible before loading EMA
            torch.cuda.empty_cache()
            gc.collect()
            
            ema_state_dict = torch.load(ckpt_dir / "ema_model.pt", map_location=device, weights_only=False)
            ema_model.load_state_dict(ema_state_dict)
            del ema_state_dict
            torch.cuda.empty_cache()
            gc.collect()
            log_memory_usage(device, latest_step, "after_loading_ema")
            logging.info(f"Successfully loaded EMA state from step {latest_step}")
        
        # Load metadata
        logging.info("Loading metadata...")
        metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
        global_step = metadata.get("global_step", latest_step)
        del metadata
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_metadata")
        
        logging.info(f"Successfully loaded all checkpoint components from step {latest_step}")
        return global_step
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clear memory and provide detailed error message
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {str(e)}")
            log_memory_usage(device, latest_step, "after_oom_error")
            raise RuntimeError(f"Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True") from e
        raise


def get_latest_checkpoint_step(checkpoint_dir):
	"""Get the latest checkpoint step number from a checkpoint directory."""
	checkpoint_steps = []
	for d in checkpoint_dir.iterdir():
		if d.is_dir() and d.name.isdigit():
			checkpoint_steps.append(int(d.name))

	return max(checkpoint_steps) if checkpoint_steps else None


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
	max_memory_allocated = memory_stats.get('allocated_bytes.all.peak', 0) / 1e9
	max_memory_reserved = memory_stats.get('reserved_bytes.all.peak', 0) / 1e9
	
	# Get DDP info if available
	ddp_info = ""
	if dist.is_initialized():
		ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"
	
	logging.info(f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}")


def train_loop(config: _config.TrainConfig):
	use_ddp, local_rank, device = setup_ddp()
	is_main = (not use_ddp) or (dist.get_rank() == 0)
	set_seed(config.seed, local_rank)

	# Initialize checkpoint directory and wandb
	resuming = False
	if config.resume:
		# Find checkpoint directory based on experiment name
		exp_checkpoint_dir = config.checkpoint_dir
		if exp_checkpoint_dir.exists():
			# Use validation to find the latest working checkpoint
			latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
			if latest_step is not None:
				resuming = True
				logging.info(f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}")
			else:
				raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir} for resume")
		else:
			raise FileNotFoundError(f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume")
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
		# For resume, checkpoint_dir is already set to the experiment directory
		logging.info(f"Using existing experiment checkpoint directory: {config.checkpoint_dir}")

	# Initialize wandb (only on main process)
	if is_main:
		init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

	# Build data loader using the unified data loader
	# Calculate effective batch size per GPU for DDP
	# For N GPUs, each GPU should get batch_size/N samples, so total across all GPUs is batch_size
	world_size = torch.distributed.get_world_size() if use_ddp else 1
	effective_batch_size = config.batch_size // world_size
	logging.info(f"Using batch size per GPU: {effective_batch_size} (total batch size across {world_size} GPUs: {config.batch_size})")
	
	# Pass the original batch size to data loader - it will handle DDP splitting internally
	loader, _ = build_datasets(config)

	# Log sample images to wandb on first batch
	if is_main and config.wandb_enabled and not resuming:
		# Create a separate data loader for sample batch to avoid consuming the main loader
		sample_data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
		sample_batch = next(iter(sample_data_loader))
		# Convert observation and actions to torch tensors
		observation, actions = sample_batch
		sample_batch = observation.to_dict()
		sample_batch["actions"] = actions
		sample_batch = batch_to_torch(sample_batch, device)

		# Create sample images for wandb
		images_to_log = []
		# Get batch size from the first image tensor
		batch_size = next(iter(sample_batch['image'].values())).shape[0]
		for i in range(min(5, batch_size)):
			# Concatenate all camera views horizontally for this batch item
			# Convert from NCHW to NHWC format for wandb
			img_concatenated = torch.cat([img[i].permute(1, 2, 0) for img in sample_batch['image'].values()], axis=1)
			img_concatenated = img_concatenated.cpu().numpy()
			images_to_log.append(wandb.Image(img_concatenated))

		wandb.log({"camera_views": images_to_log}, step=0)

		# Clear sample batch from memory
		torch.cuda.empty_cache() if torch.cuda.is_available() else None

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
	
	if hasattr(model, 'gradient_checkpointing_enable'):
		enable_gradient_checkpointing = True
		model.gradient_checkpointing_enable()
		logging.info("Enabled gradient checkpointing for memory optimization")
	else:
		enable_gradient_checkpointing = False
		logging.info("Gradient checkpointing is not supported for this model")
	
	# Log initial memory usage after model creation
	if is_main and torch.cuda.is_available():
		log_memory_usage(device, 0, "after_model_creation")
	
	if use_ddp:
		# Enable unused parameter detection to handle cases where some parameters don't participate in loss
		model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None, find_unused_parameters=True)

	# Load weights from weight_loader if specified (for fine-tuning)
	if config.pytorch_weight_path is not None:
		logging.info(f"Loading weights from: {config.pytorch_weight_path}")

		model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
		safetensors.torch.load_model((model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model), model_path)
		logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")

	# Optimizer + learning rate schedule from config
	warmup_steps = config.lr_schedule.warmup_steps
	peak_lr = config.lr_schedule.peak_lr
	decay_steps = config.lr_schedule.decay_steps
	end_lr = config.lr_schedule.decay_lr

	# Create optimizer with config parameters
	optim = torch.optim.AdamW(
		model.parameters(), 
		lr=peak_lr, 
		betas=(config.optimizer.b1, config.optimizer.b2), 
		eps=config.optimizer.eps,
		weight_decay=config.optimizer.weight_decay
	)

	# Initialize EMA if specified in config
	ema_model = None
	if config.ema_decay is not None:
		ema_model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)
		
		# Get the correct state dict from the main model
		main_model_state_dict = get_model_state_dict(model)
		
		# Load the state dict into EMA model
		ema_model.load_state_dict(main_model_state_dict)
		ema_model.eval()
		logging.info(f"Initialized EMA with decay {config.ema_decay}")

	# Load checkpoint if resuming
	global_step = 0
	if resuming:
		global_step = load_checkpoint(model, optim, config.checkpoint_dir, device, ema_model)
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
	start_time = time.time()
	infos = []  # Collect stats over log interval
	if is_main:
		logging.info(f"Running on: {platform.node()} | world_size={torch.distributed.get_world_size() if use_ddp else 1}")
		logging.info(f"Training config: batch_size={config.batch_size}, effective_batch_size={effective_batch_size}, num_train_steps={config.num_train_steps}")
		logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
		logging.info(f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}")
		logging.info(f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}")
		logging.info(f"Training precision: {model_cfg.dtype}")
		if config.ema_decay is not None:
			logging.info(f"EMA decay: {config.ema_decay}")

	# Training loop - iterate until we reach num_train_steps
	pbar = tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main) if is_main else None

	while global_step < config.num_train_steps:
		# Set epoch for distributed training
		if use_ddp and hasattr(loader, 'set_epoch'):
			loader.set_epoch(global_step // len(loader))

		for batch in loader:
			# Check if we've reached the target number of steps
			if global_step >= config.num_train_steps:
				break

			# The unified data loader returns (observation, actions) tuple
			observation, actions = batch
			
			# Convert observation and actions to torch tensors
			observation_dict = observation.to_dict()
			observation_dict["actions"] = actions
			batch_torch = batch_to_torch(observation_dict, device)
			actions = batch_torch["actions"]

			# Update LR
			for pg in optim.param_groups:
				pg["lr"] = lr_schedule(global_step)

			# Forward pass
			observation = _model.Observation.from_dict(batch_torch)
			losses = model(observation, actions)
			# Ensure losses is a tensor and handle different return types
			if isinstance(losses, (list, tuple)):
				losses = torch.stack(losses)
			elif not isinstance(losses, torch.Tensor):
				losses = torch.tensor(losses, device=device, dtype=torch.float32)
			
			loss = losses.mean()

			# Backward pass
			loss.backward()
			
			# Log memory usage after backward pass
			if global_step < 5 and is_main:
				if torch.cuda.is_available():
					log_memory_usage(device, global_step, "after_backward")

			# Gradient clipping
			grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optimizer.clip_gradient_norm)

			# Optimizer step
			optim.step()
			optim.zero_grad(set_to_none=True)
			
			# Clear gradients more aggressively
			for param in model.parameters():
				if param.grad is not None:
					param.grad.detach_()
					param.grad = None

			# Update EMA if enabled
			if ema_model is not None:
				try:
					with torch.no_grad():
						# Get parameters from the correct model structure
						main_model_params = get_model_parameters(model)
						for param, ema_param in zip(main_model_params, ema_model.parameters()):
							ema_param.data.mul_(config.ema_decay).add_(param.data, alpha=1 - config.ema_decay)
				except Exception as e:
					logging.warning(f"Failed to update EMA model: {e}")
					# Continue training without EMA update

			# Collect stats
			if is_main:
				infos.append({
					"loss": loss.item(),
					"learning_rate": optim.param_groups[0]['lr'],
					"grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
				})

			if is_main and (global_step % config.log_interval == 0):
				elapsed = time.time() - start_time

				# Average stats over log interval
				avg_loss = sum(info["loss"] for info in infos) / len(infos)
				avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)

				avg_grad_norm = None
				if any('grad_norm' in info for info in infos):
					vals = [info['grad_norm'] for info in infos if 'grad_norm' in info and info['grad_norm'] is not None]
					if len(vals) > 0:
						avg_grad_norm = sum(vals) / len(vals)
				logging.info(f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} grad_norm={avg_grad_norm:.2f} time={elapsed:.1f}s" if avg_grad_norm is not None else f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} time={elapsed:.1f}s")

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

				start_time = time.time()
				infos = []  # Reset stats collection

			# Save checkpoint using the new mechanism
			save_checkpoint(model, optim, global_step, config, is_main, ema_model)

			global_step += 1

			# Update progress bar
			if pbar is not None:
				pbar.update(1)
				pbar.set_postfix({
					'loss': f'{loss.item():.4f}',
					'lr': f'{optim.param_groups[0]["lr"]:.2e}',
					'step': global_step
				})

	# Close progress bar
	if pbar is not None:
		pbar.close()

	# Finish wandb run
	if is_main and config.wandb_enabled:
		wandb.finish()

	cleanup_ddp()


def main():
	init_logging()
	config = _config.cli()
	train_loop(config)


if __name__ == "__main__":
	main()
