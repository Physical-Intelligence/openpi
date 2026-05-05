# config.py: Builds TrainConfig for single right-arm pi0.5 fine-tuning on lipbalm dataset.
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
    repo_id: str = "mobileai-lipbalm-all-right",
    exp_name: str = "lipbalm_pi05",
    num_train_steps: int = 20_000,
    batch_size: int = 64,
    base_checkpoint: str = "gs://openpi-assets/checkpoints/pi05_base/params",
    asset_id: str = "trossen",
    assets_dir: str = "gs://openpi-assets/checkpoints/pi05_base/assets",
    peak_lr: float = 5e-5,
    warmup_steps: int = 1000,
    save_interval: int = 2000,
    resume: bool = False,
    wandb_enabled: bool = True,
    project_name: str = "openpi-lipbalm",
    checkpoint_base_dir: str = "./checkpoints",
    pytorch_weight_path: str | None = None,
) -> _config.TrainConfig:
    """Build a TrainConfig for single right-arm pi0.5 fine-tuning."""
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

    return _config.TrainConfig(
        name="pi05_aloha_lipbalm",
        project_name=project_name,
        exp_name=exp_name,
        model=pi0_config.Pi0Config(pi05=True),
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
                default_prompt="pick lipbalm",
            ),
            base_config=_config.DataConfig(
                prompt_from_task=True,
                repack_transforms=repack,
                action_sequence_keys=("action",),
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(base_checkpoint),
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

    return build_train_config(
        repo_id=data.get("merged_name", "mobileai-lipbalm-all-right"),
        exp_name=experiment.get("name", "lipbalm_pi05"),
        num_train_steps=training.get("num_train_steps", 20_000),
        batch_size=training.get("batch_size", 64),
        base_checkpoint=model.get("base_checkpoint", "gs://openpi-assets/checkpoints/pi05_base/params"),
        asset_id=model.get("asset_id", "trossen"),
        assets_dir=model.get("assets_dir", "gs://openpi-assets/checkpoints/pi05_base/assets"),
        peak_lr=training.get("peak_lr", 5e-5),
        warmup_steps=training.get("warmup_steps", 1000),
        save_interval=training.get("save_interval", 2000),
        resume=training.get("resume", False),
        wandb_enabled=training.get("wandb_enabled", True),
        project_name=experiment.get("project_name", "openpi-lipbalm"),
        checkpoint_base_dir=training.get("checkpoint_base_dir", "./checkpoints"),
        pytorch_weight_path=model.get("pytorch_weight_path"),
    )
