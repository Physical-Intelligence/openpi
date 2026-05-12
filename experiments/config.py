# config.py: Builds TrainConfig for single right-arm pi0.5 fine-tuning experiments.
# Self-contained — does not modify the openpi codebase.

import sys
from pathlib import Path

import yaml

import openpi.models.pi0_config as pi0_config
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

# Import local transforms from experiments/
sys.path.insert(0, str(Path(__file__).parent))
from transforms import AlohaSingleArmInputs, AlohaSingleArmOutputs


def build_train_config(
    repo_id: str,
    exp_name: str,
    config_name: str,
    default_prompt: str,
    project_name: str,
    num_train_steps: int = 10_000,
    batch_size: int = 16,
    base_checkpoint: str = "gs://openpi-assets/checkpoints/pi05_base/params",
    asset_id: str = "trossen",
    assets_dir: str = "gs://openpi-assets/checkpoints/pi05_base/assets",
    peak_lr: float = 5e-5,
    warmup_steps: int = 1000,
    save_interval: int = 2000,
    resume: bool = False,
    wandb_enabled: bool = True,
    checkpoint_base_dir: str = "./checkpoints",
    pytorch_weight_path: str | None = None,
    use_lora: bool = True,
) -> _config.TrainConfig:
    """Build a TrainConfig for single right-arm pi0.5 LoRA fine-tuning."""
    repack = _transforms.Group(
        inputs=[
            _transforms.RepackTransform(
                {
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                }
            )
        ]
    )

    if use_lora:
        model_config = pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        )
        freeze_filter = model_config.get_freeze_filter()
        ema_decay = None
    else:
        model_config = pi0_config.Pi0Config(pi05=True)
        freeze_filter = _config.nnx.Nothing
        ema_decay = 0.99

    return _config.TrainConfig(
        name=config_name,
        project_name=project_name,
        exp_name=exp_name,
        model=model_config,
        data=_config.SimpleDataConfig(
            repo_id=repo_id,
            assets=_config.AssetsConfig(
                assets_dir=assets_dir,
                asset_id=asset_id,
            ),
            data_transforms=lambda _: _transforms.Group(
                inputs=[AlohaSingleArmInputs()],
                outputs=[AlohaSingleArmOutputs()],
            ),
            model_transforms=_config.ModelTransformFactory(
                default_prompt=default_prompt,
            ),
            base_config=_config.DataConfig(
                prompt_from_task=True,
                repack_transforms=repack,
                action_sequence_keys=("action",),
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(base_checkpoint),
        freeze_filter=freeze_filter,
        ema_decay=ema_decay,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=warmup_steps,
            peak_lr=peak_lr,
            decay_steps=num_train_steps,
            decay_lr=peak_lr * 0.1,
        ),
        num_train_steps=num_train_steps,
        batch_size=batch_size,
        save_interval=save_interval,
        resume=resume,
        wandb_enabled=wandb_enabled,
        checkpoint_base_dir=checkpoint_base_dir,
        pytorch_weight_path=pytorch_weight_path,
    )


def build_config_from_yaml(yaml_path: str) -> _config.TrainConfig:
    """Build TrainConfig from a YAML experiment config file."""
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    training = cfg.get("training", {})
    model = cfg.get("model", {})
    data = cfg.get("data", {})
    experiment = cfg.get("experiment", {})

    # Derive prompt and config name from experiment name
    exp_name = experiment["name"]
    task_name = exp_name.replace("_pi05_lora", "").replace("_pi05", "")
    default_prompt = data.get("default_prompt", f"pick {task_name.replace('_', ' ')}")

    return build_train_config(
        repo_id=data["merged_name"],
        exp_name=exp_name,
        config_name=f"pi05_aloha_{task_name}",
        default_prompt=default_prompt,
        project_name=experiment["project_name"],
        num_train_steps=training.get("num_train_steps", 10_000),
        batch_size=training.get("batch_size", 16),
        base_checkpoint=model.get("base_checkpoint", "gs://openpi-assets/checkpoints/pi05_base/params"),
        asset_id=model.get("asset_id", "trossen"),
        assets_dir=model.get("assets_dir", "gs://openpi-assets/checkpoints/pi05_base/assets"),
        peak_lr=training.get("peak_lr", 5e-5),
        warmup_steps=training.get("warmup_steps", 1000),
        save_interval=training.get("save_interval", 2000),
        resume=training.get("resume", False),
        wandb_enabled=training.get("wandb_enabled", True),
        checkpoint_base_dir=training.get("checkpoint_base_dir", "./checkpoints"),
        pytorch_weight_path=model.get("pytorch_weight_path"),
        use_lora=training.get("use_lora", True),
    )
