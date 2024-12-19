from collections.abc import Sequence
import dataclasses
import difflib
import os
import pathlib
from typing import Annotated, Protocol, Union, runtime_checkable

import namer
import tyro

import openpi.models.common as common
import openpi.models.model as _model
import openpi.models.pi0 as pi0
import openpi.models.pi0_small as pi0_small
from openpi.policies import aloha_policy
import openpi.shared.normalize as _normalize
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms


def default_dataset_root() -> str | None:
    # TODO(ury): Temporary, remove this once the default works well.
    if os.path.exists("/mnt/weka"):  # noqa: PTH110
        return "/mnt/weka/lerobot"
    return None


@dataclasses.dataclass
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Contains precomputed normalization stats.
    norm_stats: dict[str, _transforms.NormStats] | None = None
    # Input transforms.
    input_transforms: Sequence[_transforms.DataTransformFn] = ()
    # Default prompt that will be used the model.
    default_prompt: str | None = None

    # Indicates where the cached dataset should be stored.
    # This can also be controlled by setting the LEROBOT_HOME environment variable.
    dataset_root: str | None = dataclasses.field(default_factory=default_dataset_root)


@runtime_checkable
class DataConfigFactory(Protocol):
    def create(self, metadata_dir: pathlib.Path, model: _model.Model) -> DataConfig:
        """Create a data config."""


class FakeDataConfig(DataConfigFactory):
    def create(self, metadata_dir: pathlib.Path, model: _model.Model) -> DataConfig:
        return DataConfig(repo_id="fake")


class LeRobotRepack(_transforms.DataTransformFn):
    def __call__(self, item) -> dict:
        return {
            "images": {"cam_high": item["observation.images.top"]},
            "state": item["observation.state"],
            "actions": item["action"],
        }


@dataclasses.dataclass(frozen=True)
class LeRobotDataConfig(DataConfigFactory):
    repo_id: str = "lerobot/aloha_sim_transfer_cube_human"
    delta_action_mask: Sequence[bool] | None = None
    default_prompt: str | None = None

    def create(self, metadata_dir: pathlib.Path, model: _model.Model) -> DataConfig:
        norm_stats_path = metadata_dir / self.repo_id / "norm_stats.json"
        norm_stats = _normalize.deserialize_json(norm_stats_path.read_text()) if norm_stats_path.exists() else None

        return DataConfig(
            repo_id=self.repo_id,
            norm_stats=norm_stats,
            default_prompt=self.default_prompt,
            input_transforms=[
                LeRobotRepack(),
                aloha_policy.AlohaInputs(action_dim=model.action_dim, delta_action_mask=self.delta_action_mask),
                _transforms.ResizeImages(224, 224),
            ],
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    name: str
    project_name: str = "openpi"
    exp_name: str = namer.generate(category=["food", "technology"], suffix_length=3)  # noqa: RUF009

    action_dim: int = 24
    action_horizon: int = 50
    max_token_len: int = 48

    module: common.BaseModule = dataclasses.field(default_factory=pi0.Module)
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = None

    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    metadata_base_dir: str = "./assets"
    checkpoint_base_dir: str = "./checkpoints"

    seed: int = 42
    batch_size: int = 16
    num_train_steps: int = 2_000_000

    log_interval: int = 100
    save_interval: int = 1000
    keep_interval: int = 5000

    overwrite: bool = False
    resume: bool = False

    @property
    def metadata_dir(self) -> pathlib.Path:
        """Get the metadata directory for this config."""
        return (pathlib.Path(self.metadata_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        return (pathlib.Path(self.checkpoint_base_dir) / self.exp_name).resolve()

    def create_model(self) -> _model.Model:
        return _model.Model(
            module=self.module,
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            max_token_len=self.max_token_len,
        )


_CONFIGS = [
    #
    # pi0 configs.
    #
    TrainConfig(
        name="pi0",
        data=LeRobotDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            delta_action_mask=None,
        ),
    ),
    TrainConfig(
        name="pi0_pretrained",
        data=LeRobotDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            delta_action_mask=None,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("checkpoints/pi0_base/model"),
    ),
    TrainConfig(
        name="pi0_paligemma",
        weight_loader=weight_loaders.PaliGemmaWeightLoader(),
    ),
    #
    # pi0_small configs.
    #
    TrainConfig(
        name="pi0_small",
        module=pi0_small.Module(),
        weight_loader=weight_loaders.GoogleViTWeightLoader(),
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        batch_size=2,
        module=pi0.Module(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
    ),
    TrainConfig(
        name="debug_restore",
        batch_size=2,
        module=pi0.Module(paligemma_variant="dummy", action_expert_variant="dummy"),
        resume=True,
        exp_name="debug",
        num_train_steps=10,
    ),
]

_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.cli(
        Union.__getitem__(  # type: ignore
            tuple(
                Annotated.__class_getitem__(  # type: ignore
                    (
                        Annotated.__class_getitem__((type(v), tyro.conf.AvoidSubcommands)),  # type: ignore
                        tyro.conf.subcommand(k, default=v),
                    )
                )
                for k, v in _CONFIGS_DICT.items()
            )
        ),
    )


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f"Did you mean '{closest[0]}'? " if closest else ""
        if closest:
            raise ValueError(f"Config '{config_name}' not found. Did you mean '{closest_str}'?")
        raise ValueError(f"Config '{config_name}' not found.")

    return _CONFIGS_DICT[config_name]
