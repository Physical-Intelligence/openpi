"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
import numpy as np
import openpi.shared.nnx_utils as nnx_utils
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.pi0_rl_config as pi0_rl_config
import openpi.models.pi05_config as pi05_config
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.arx5_multitask_policy as arx5_multitask_policy
import openpi.policies.bin_pack_policy as bin_pack_policy
import openpi.policies.block_tower_policy as block_tower_policy
import openpi.policies.arx_policy as arx_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.so101_policy as so101_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.libero_subtask_policy as libero_subtask_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None
    # If true, use per-timestep action normalization when available. If None, use defaults set by data configs.
    use_per_timestep_action_norm: bool | None = None
    # Per-timestep action normalization stats (actions only).
    per_timestep_action_norm_stats: _transforms.NormStats | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False
    # If true, will override the prompt with the per-episode subtask description (when available).
    prompt_from_subtask: bool = False
    # Optional deterministic episode-level train/validation split.
    episode_split: "EpisodeSplitConfig | None" = None

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class EpisodeSplitConfig:
    val_ratio: float = 0.1
    seed: int = 42


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                # Support both Pi05Config and Pi0Config with pi05=True
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeHighPrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class SubtaskModelTransformFactory(GroupFactory):
    """Creates model transforms for subtask-based hierarchical learning."""

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI05:
                # Use FAST tokens only if fast_token_loss_weight > 0
                use_fast_tokens = getattr(model_config, "fast_token_loss_weight", 0.0) > 0

                # Build tokenizer kwargs (with or without FAST tokenizer path)
                tokenizer_kwargs = {"max_len": model_config.max_token_len}
                if use_fast_tokens:
                    fast_tokenizer_path = getattr(model_config, "fast_tokenizer_path", "physical-intelligence/fast")
                    tokenizer_kwargs["fast_tokenizer_path"] = fast_tokenizer_path

                return _transforms.Group(
                    inputs=[
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeHighLowPrompt(
                            _tokenizer.PaligemmaTokenizer(**tokenizer_kwargs),
                            use_fast_tokens=use_fast_tokens,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _:
                raise ValueError(f"Subtask mode only supports PI05 model type, got {model_config.model_type}")


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            per_timestep_action_norm_stats=self._load_per_timestep_action_norm_stats(
                epath.Path(self.assets.assets_dir or assets_dirs), asset_id
            ),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            try:
                norm_stats = _normalize.load(_download.maybe_download(str(assets_dir)))
                logging.info(f"Loaded norm stats from {assets_dir}")
                return norm_stats
            except FileNotFoundError:
                logging.info(f"Norm stats not found in {assets_dir}, skipping.")
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None

    def _load_per_timestep_action_norm_stats(
        self, assets_dir: epath.Path, asset_id: str | None
    ) -> _transforms.NormStats | None:
        if asset_id is None:
            try:
                action_stats = _normalize.load_actions_per_timestep(_download.maybe_download(str(assets_dir)))
                logging.info(f"Loaded per-timestep action stats from {assets_dir}")
                return action_stats
            except FileNotFoundError:
                logging.info(f"Per-timestep action stats not found in {assets_dir}, skipping.")
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            action_stats = _normalize.load_actions_per_timestep(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded per-timestep action stats from {data_assets_dir}")
            return action_stats
        except FileNotFoundError:
            logging.info(f"Per-timestep action stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        base_config = self.create_base_config(assets_dirs, model_config)
        use_per_timestep_action_norm = base_config.use_per_timestep_action_norm
        if self.use_delta_joint_actions and use_per_timestep_action_norm is None:
            use_per_timestep_action_norm = True

        return dataclasses.replace(
            base_config,
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            use_per_timestep_action_norm=use_per_timestep_action_norm,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)
        base_config = self.create_base_config(assets_dirs, model_config)
        use_per_timestep_action_norm = base_config.use_per_timestep_action_norm
        if self.extra_delta_transform and use_per_timestep_action_norm is None:
            use_per_timestep_action_norm = True

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            base_config,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            use_per_timestep_action_norm=use_per_timestep_action_norm,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotBinPackDataConfig(DataConfigFactory):
    """Data config for the bin_pack_coffee_capsules LeRobot dataset."""

    default_prompt: str | None = "pack coffee capsules into the cardboard bin container"
    use_control_mode_advantage_prompt: bool = False
    advantage_prompt_mode: Literal["positive_only", "mixed"] = "mixed"
    advantage_dropout_rate: float = 0.0
    # If true, will convert actions to deltas relative to the current state. When mask is None,
    # all action dimensions shared with state will be treated as delta dimensions.
    use_delta_actions: bool = False
    # Optional mask for which action dimensions should be converted to deltas.
    delta_action_mask: Sequence[bool] | None = None
    # If true, keep model outputs as deltas at inference time (do not convert back to absolute).
    output_delta_actions: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # BinPackInputs converts EEF orientation from RPY (3) to rot6d (6),
        # so (pos(7) + eef(7)) becomes (pos(7) + eef(10)) = 17 dims.
        _ROT6D_SLICE = slice(10, 16)  # indices of rot6d inside the 17D state/action vector

        input_transforms: list[_transforms.DataTransformFn] = []
        if self.use_control_mode_advantage_prompt:
            input_transforms.append(
                _transforms.SetAdvantageLabelFromControlMode(
                    mode=self.advantage_prompt_mode,
                    dropout_rate=self.advantage_dropout_rate,
                )
            )
        input_transforms.append(bin_pack_policy.BinPackInputs())

        data_transforms = _transforms.Group(
            inputs=input_transforms,
            # Slice to the 17D "semantic" action, then decode rot6d back to RPY for downstream consumers.
            outputs=[bin_pack_policy.BinPackOutputs(action_dim=17, output_rpy=True)],
        )
        if self.use_delta_actions:
            # If mask is not provided, default to delta-ing joints + xyz + gripper, but keep rot6d absolute.
            # (Subtracting rot6d vectors is not a valid relative-rotation representation.)
            delta_action_mask = self.delta_action_mask
            if delta_action_mask is None:
                delta_action_mask = tuple([True] * 10 + [False] * 6 + [True])  # 17 dims
            output_transforms = []
            if not self.output_delta_actions:
                output_transforms.append(_transforms.AbsoluteActionsFromState(delta_action_mask))
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActionsFromState(delta_action_mask)],
                outputs=output_transforms,
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        base_config = self.create_base_config(assets_dirs, model_config)

        # Match TRI LBM guidance for 6D rotation: do not normalize rot6d components (they already live in [-1, 1]).
        # We achieve this by overriding the normalization stats so quantile normalization becomes identity
        # (q01=-1, q99=+1) and z-score normalization becomes identity (mean=0, std=1) for those dims.
        def _set_rot6d_identity(stats: _transforms.NormStats) -> _transforms.NormStats:
            mean = np.array(stats.mean, copy=True)
            std = np.array(stats.std, copy=True)
            if mean.shape[-1] < _ROT6D_SLICE.stop or std.shape[-1] < _ROT6D_SLICE.stop:
                raise ValueError(
                    "Bin-pack rot6d identity normalization expects at least 17D stats "
                    f"(got mean {mean.shape}, std {std.shape}). "
                    "Regenerate norm stats after enabling rot6d encoding."
                )
            mean[..., _ROT6D_SLICE] = 0.0
            std[..., _ROT6D_SLICE] = 1.0
            q01 = None if stats.q01 is None else np.array(stats.q01, copy=True)
            q99 = None if stats.q99 is None else np.array(stats.q99, copy=True)
            if q01 is not None:
                q01[..., _ROT6D_SLICE] = -1.0
            if q99 is not None:
                q99[..., _ROT6D_SLICE] = 1.0
            return _normalize.NormStats(mean=mean, std=std, q01=q01, q99=q99)

        patched_norm_stats = base_config.norm_stats
        if patched_norm_stats is not None:
            patched_norm_stats = dict(patched_norm_stats)
            if "state" in patched_norm_stats:
                patched_norm_stats["state"] = _set_rot6d_identity(patched_norm_stats["state"])
            if "actions" in patched_norm_stats:
                patched_norm_stats["actions"] = _set_rot6d_identity(patched_norm_stats["actions"])

        patched_per_ts = base_config.per_timestep_action_norm_stats
        if patched_per_ts is not None:
            patched_per_ts = _set_rot6d_identity(patched_per_ts)

        use_per_timestep_action_norm = base_config.use_per_timestep_action_norm
        if self.use_delta_actions and use_per_timestep_action_norm is None:
            use_per_timestep_action_norm = True
        return dataclasses.replace(
            base_config,
            norm_stats=patched_norm_stats,
            per_timestep_action_norm_stats=patched_per_ts,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action.pos", "action.eef_pose"),
            use_per_timestep_action_norm=use_per_timestep_action_norm,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotBlockTowerDataConfig(DataConfigFactory):
    """Data config for build_block_tower datasets (LeRobot v2.1 format).

    Raw state and actions are 7D joint-space, but the training pipeline maps
    them into the repo-standard 17D canonical action/state representation.
    """

    default_prompt: str | None = "build a block tower"
    use_control_mode_advantage_prompt: bool = False
    advantage_prompt_mode: Literal["positive_only", "mixed"] = "mixed"
    advantage_dropout_rate: float = 0.0
    use_delta_actions: bool = False
    delta_action_mask: Sequence[bool] | None = None
    output_delta_actions: bool = False
    # When True, restricts training to the first 7 (joint) dims of the 17D
    # canonical action vector. In the input pipeline (which is what training
    # and norm precompute see): EE channels of state/action are zeroed and
    # action_dim_mask is forced to joints-only so the flow-matching loss ignores
    # them. At inference time, BlockTowerOutputs slices the policy output to 7D.
    joints_only: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        _RAW_DIM = block_tower_policy._RAW_DIM  # 7 (joint dims)
        _CANONICAL_DIM = block_tower_policy._CANONICAL_DIM  # 17 (full canonical layout)
        _ROT6D_SLICE = slice(10, 16)  # indices of rot6d inside the 17D state/action vector

        input_transforms: list[_transforms.DataTransformFn] = []
        if self.use_control_mode_advantage_prompt:
            input_transforms.append(
                _transforms.SetAdvantageLabelFromControlMode(
                    mode=self.advantage_prompt_mode,
                    dropout_rate=self.advantage_dropout_rate,
                )
            )
        input_transforms.append(block_tower_policy.BlockTowerInputs(joints_only=self.joints_only))

        output_action_dim = _RAW_DIM if self.joints_only else _CANONICAL_DIM
        data_transforms = _transforms.Group(
            inputs=input_transforms,
            outputs=[block_tower_policy.BlockTowerOutputs(action_dim=output_action_dim)],
        )
        if self.use_delta_actions:
            delta_action_mask = self.delta_action_mask
            if delta_action_mask is None:
                # Follow the repo's 17D convention: only rot6d stays absolute.
                delta_action_mask = tuple([True] * 10 + [False] * 6 + [True])
            output_transforms = []
            if not self.output_delta_actions:
                output_transforms.append(_transforms.AbsoluteActionsFromState(delta_action_mask))
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActionsFromState(delta_action_mask)],
                outputs=output_transforms,
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        base_config = self.create_base_config(assets_dirs, model_config)

        # Pick the normalization-patching slice. In the standard 17D run we only
        # need to keep the rot6d channels identity-normalized so quantile/z-score
        # normalization doesn't distort the rotation representation. In the
        # joints-only run all EE channels (xyz + rot6d + gripper, indices 7..17)
        # are constant zeros, so a learned norm would divide by ~0; pin the
        # entire EE slice to identity so the model sees clean zeros instead.
        identity_slice = slice(_RAW_DIM, _CANONICAL_DIM) if self.joints_only else _ROT6D_SLICE

        def _set_identity(stats: _transforms.NormStats) -> _transforms.NormStats:
            mean = np.array(stats.mean, copy=True)
            std = np.array(stats.std, copy=True)
            if mean.shape[-1] < identity_slice.stop or std.shape[-1] < identity_slice.stop:
                raise ValueError(
                    f"Block-tower identity normalization expects at least {identity_slice.stop}D stats "
                    f"(got mean {mean.shape}, std {std.shape}). "
                    "Regenerate norm stats with the correct data config."
                )
            mean[..., identity_slice] = 0.0
            std[..., identity_slice] = 1.0
            q01 = None if stats.q01 is None else np.array(stats.q01, copy=True)
            q99 = None if stats.q99 is None else np.array(stats.q99, copy=True)
            if q01 is not None:
                q01[..., identity_slice] = -1.0
            if q99 is not None:
                q99[..., identity_slice] = 1.0
            return _normalize.NormStats(mean=mean, std=std, q01=q01, q99=q99)

        patched_norm_stats = base_config.norm_stats
        if patched_norm_stats is not None:
            patched_norm_stats = dict(patched_norm_stats)
            if "state" in patched_norm_stats:
                patched_norm_stats["state"] = _set_identity(patched_norm_stats["state"])
            if "actions" in patched_norm_stats:
                patched_norm_stats["actions"] = _set_identity(patched_norm_stats["actions"])

        patched_per_ts = base_config.per_timestep_action_norm_stats
        if patched_per_ts is not None:
            patched_per_ts = _set_identity(patched_per_ts)

        use_per_timestep_action_norm = base_config.use_per_timestep_action_norm
        if self.use_delta_actions and use_per_timestep_action_norm is None:
            use_per_timestep_action_norm = True
        return dataclasses.replace(
            base_config,
            norm_stats=patched_norm_stats,
            per_timestep_action_norm_stats=patched_per_ts,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
            use_per_timestep_action_norm=use_per_timestep_action_norm,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotBlockTowerSubtaskDataConfig(LeRobotBlockTowerDataConfig):
    """Hierarchical build_block_tower config with explicit subtask prompts."""

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        _ROT6D_SLICE = slice(10, 16)

        if self.joints_only:
            raise NotImplementedError(
                "joints_only=True is not yet wired through LeRobotBlockTowerSubtaskDataConfig. "
                "Use LeRobotBlockTowerDataConfig (flat) for the joints-only ablation, or "
                "extend this subtask create() to mirror the joints-only path."
            )

        input_transforms: list[_transforms.DataTransformFn] = []
        input_transforms.append(block_tower_policy.BlockTowerSubtaskInputs(default_prompt=self.default_prompt or ""))
        if self.use_control_mode_advantage_prompt:
            input_transforms.append(
                _transforms.SetAdvantageLabelFromControlMode(
                    mode=self.advantage_prompt_mode,
                    dropout_rate=self.advantage_dropout_rate,
                )
            )

        data_transforms = _transforms.Group(
            inputs=input_transforms,
            outputs=[block_tower_policy.BlockTowerOutputs(action_dim=17)],
        )
        if self.use_delta_actions:
            delta_action_mask = self.delta_action_mask
            if delta_action_mask is None:
                delta_action_mask = tuple([True] * 10 + [False] * 6 + [True])
            output_transforms = []
            if not self.output_delta_actions:
                output_transforms.append(_transforms.AbsoluteActionsFromState(delta_action_mask))
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActionsFromState(delta_action_mask)],
                outputs=output_transforms,
            )

        model_transforms = SubtaskModelTransformFactory()(model_config)
        base_config = self.create_base_config(assets_dirs, model_config)

        def _set_rot6d_identity(stats: _transforms.NormStats) -> _transforms.NormStats:
            mean = np.array(stats.mean, copy=True)
            std = np.array(stats.std, copy=True)
            if mean.shape[-1] < _ROT6D_SLICE.stop or std.shape[-1] < _ROT6D_SLICE.stop:
                raise ValueError(
                    "Block-tower rot6d identity normalization expects at least 17D stats "
                    f"(got mean {mean.shape}, std {std.shape}). "
                    "Regenerate norm stats after enabling rot6d encoding."
                )
            mean[..., _ROT6D_SLICE] = 0.0
            std[..., _ROT6D_SLICE] = 1.0
            q01 = None if stats.q01 is None else np.array(stats.q01, copy=True)
            q99 = None if stats.q99 is None else np.array(stats.q99, copy=True)
            if q01 is not None:
                q01[..., _ROT6D_SLICE] = -1.0
            if q99 is not None:
                q99[..., _ROT6D_SLICE] = 1.0
            return _normalize.NormStats(mean=mean, std=std, q01=q01, q99=q99)

        patched_norm_stats = base_config.norm_stats
        if patched_norm_stats is not None:
            patched_norm_stats = dict(patched_norm_stats)
            if "state" in patched_norm_stats:
                patched_norm_stats["state"] = _set_rot6d_identity(patched_norm_stats["state"])
            if "actions" in patched_norm_stats:
                patched_norm_stats["actions"] = _set_rot6d_identity(patched_norm_stats["actions"])

        patched_per_ts = base_config.per_timestep_action_norm_stats
        if patched_per_ts is not None:
            patched_per_ts = _set_rot6d_identity(patched_per_ts)

        use_per_timestep_action_norm = base_config.use_per_timestep_action_norm
        if self.use_delta_actions and use_per_timestep_action_norm is None:
            use_per_timestep_action_norm = True
        return dataclasses.replace(
            base_config,
            norm_stats=patched_norm_stats,
            per_timestep_action_norm_stats=patched_per_ts,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
            use_per_timestep_action_norm=use_per_timestep_action_norm,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotARX5MultiTaskDataConfig(DataConfigFactory):
    """Data config for multi-task training across ARX5 single-arm and bimanual datasets.

    Handles mixed robot configs: single-arm (7-dim) padded to bimanual (14-dim)
    with loss masking on padded dimensions. Agilex gripper values are rescaled
    from centimeters to meters in the data transform.
    """

    default_prompt: str | None = "Do something useful"
    use_delta_actions: bool = False
    delta_action_mask: Sequence[bool] | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Resolve repo_id relative to assets_dirs so all assets live together.
        if self.repo_id.endswith(".json") and not pathlib.Path(self.repo_id).is_absolute():
            resolved = assets_dirs / pathlib.Path(self.repo_id).name
            object.__setattr__(self, "repo_id", str(resolved))

        data_transforms = _transforms.Group(
            inputs=[arx5_multitask_policy.ARX5MultiTaskInputs()],
            outputs=[arx5_multitask_policy.ARX5MultiTaskOutputs()],
        )

        if self.use_delta_actions:
            delta_action_mask = self.delta_action_mask
            if delta_action_mask is None:
                # Delta joints, absolute grippers for bimanual: [J]*6 + [G] + [J]*6 + [G]
                delta_action_mask = tuple([True] * 6 + [False] + [True] * 6 + [False])
            # Use the FromState variants so the mask is clipped to
            # min(mask_dim, state_dim, action_dim). This is critical at
            # inference time: a single-arm robot sends 7-dim state while the
            # mask is 14-dim; the FromState variants handle this gracefully
            # instead of crashing on a shape mismatch.
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActionsFromState(delta_action_mask)],
                outputs=[_transforms.AbsoluteActionsFromState(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        base_config = self.create_base_config(assets_dirs, model_config)

        use_per_timestep_action_norm = base_config.use_per_timestep_action_norm
        if self.use_delta_actions and use_per_timestep_action_norm is None:
            use_per_timestep_action_norm = True

        return dataclasses.replace(
            base_config,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            # RoboCandyWrapper normalises action keys to "action" via key_rename_map
            action_sequence_keys=("action",),
            use_per_timestep_action_norm=use_per_timestep_action_norm,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.

    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotSO101DataConfig(DataConfigFactory):
    """Data config for SO101 single-arm robot (6D joint-space).

    Supports optional delta action conversion (5 joints delta, gripper absolute).
    """

    default_prompt: str | None = "stack the rings"
    use_delta_actions: bool = False

    # Repack transforms: map LeRobot v3 keys to canonical internal keys.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation.images.front": "observation.images.front",
                        "observation.images.wrist": "observation.images.wrist",
                        "observation.state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[so101_policy.SO101Inputs(default_prompt=self.default_prompt or "stack the rings")],
            outputs=[so101_policy.SO101Outputs()],
        )

        if self.use_delta_actions:
            # 5 joints as delta, gripper (dim 5) stays absolute.
            delta_action_mask = _transforms.make_bool_mask(5, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        base_config = self.create_base_config(assets_dirs, model_config)

        use_per_timestep_action_norm = base_config.use_per_timestep_action_norm
        if self.use_delta_actions and use_per_timestep_action_norm is None:
            use_per_timestep_action_norm = True

        return dataclasses.replace(
            base_config,
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            use_per_timestep_action_norm=use_per_timestep_action_norm,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoSubtaskDataConfig(DataConfigFactory):
    """
    Data config for Libero environment with subtask support.
    Assumes features stored as:
      - images.agentview_rgb: image (3, 256, 256) uint8
      - images.wrist_rgb: image (3, 256, 256) uint8
      - state: float32, shape (8,)
      - actions: float32, shape (horizon, 7)
      - task: string (high-level task)
      - subtask: string (low-level subtask)
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images.agentview_rgb": "images.agentview_rgb",
                        # Map dataset wrist image name to the expected key.
                        "images.wrist_rgb_left": "images.wrist_rgb",
                        "state": "state",
                        "actions": "actions",
                        "task": "task",
                        "subtask": "subtask",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[libero_subtask_policy.LiberoSubtaskInputs(model_type=model_config.model_type)],
            outputs=[libero_subtask_policy.LiberoSubtaskOutputs()],
        )

        model_transforms = SubtaskModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Optional fully-qualified assets directory (overrides assets_base_dir/name).
    assets_dir: str | None = None
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to run validation when a validation split is enabled.
    val_interval: int | None = None
    # Number of validation batches to average when validation is enabled.
    val_num_batches: int = 10
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        if self.assets_dir is not None:
            return pathlib.Path(self.assets_dir).resolve()
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir).expanduser() / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_BLOCK_TOWER_6MIX_REPO_ID = (
    "["
    "villekuosmanen/build_block_tower, "
    "villekuosmanen/dAgger_build_block_tower_1.0.0, "
    "villekuosmanen/dAgger_build_block_tower_1.1.0, "
    "villekuosmanen/dAgger_build_block_tower_1.2.0, "
    "villekuosmanen/dAgger_build_block_tower_1.3.0, "
    "villekuosmanen/dAgger_build_block_tower_1.4.0"
    "]"
)

_CONFIGS = [
    # ⭐ Libero Subtask Training Configurations - Three libero training modes
    
    # Mode 1: Subtask + Flow Matching (Original Pi05 style)
    TrainConfig(
        name="libero_pi05_subtask_flow",
        exp_name="libero_pi05_subtask_flow",
        model=pi05_config.Pi05Config(
            action_horizon=10,
            max_token_len=256,
            discrete_state_input=False,
            # ⭐ Only use subtask and flow matching loss
            subtask_loss_weight=1.0,
            fast_token_loss_weight=0.0,  # Disable FAST token loss
            flow_matching_loss_weight=1.0,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("weights/pi05_base/params"),
        data=LeRobotLiberoSubtaskDataConfig(
            repo_id="KeWangRobotics/libero_10_subtasks",
            base_config=DataConfig(
                asset_id="libero_subtask",
                use_quantile_norm=True,  # ⭐ Use quantile normalization for gripper actions
            ),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=3000,
            peak_lr=2.5e-5,
            decay_steps=150_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=100_000,
        save_interval=10000,
        batch_size=32,
        fsdp_devices=8,
        ema_decay=0.999,
        wandb_enabled=True,

    ),
    
    # Mode 2: Subtask + FAST Token (Discrete action tokens)
    TrainConfig(
        name="libero_pi05_subtask_fast",
        exp_name="libero_subtask_fast",
        model=pi05_config.Pi05Config(
            action_horizon=25,
            max_token_len=256,
            discrete_state_input=False,
            # ⭐ Only use subtask and FAST token loss
            subtask_loss_weight=10.0,
            fast_token_loss_weight=1.0,  # Enable FAST token loss weight
            flow_matching_loss_weight=0.0,  # Disable flow matching
            fast_tokenizer_path="weights/fast",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("weights/pi05_base/params"),
        
        data=LeRobotLiberoSubtaskDataConfig(
            repo_id="KeWangRobotics/libero_10_subtasks",
            base_config=DataConfig(
                asset_id="libero_subtask",
                use_quantile_norm=True,  # ⭐ Use quantile normalization for gripper actions
            ),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=3000,
            peak_lr=2.5e-5,
            decay_steps=150_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=20_000,
        save_interval=4000,
        batch_size=512,
        fsdp_devices=8,
        ema_decay=0.999,
        wandb_enabled=True,
    ),
        
    # Mode 3: Action Expert
    TrainConfig(
        name="libero_pi05_action_expert",
        exp_name="libero_action_expert",
        model=pi05_config.Pi05Config(
            action_horizon=25,
            max_token_len=256,
            discrete_state_input=False,
            # ⭐ Only use action expert loss
            subtask_loss_weight=0.0,
            fast_token_loss_weight=0.0,  
            flow_matching_loss_weight=1.0,  # Enable flow matching
            fast_tokenizer_path="weights/fast",
            stop_gradient_flow_to_prefix=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/kewang/.cache/openpi/openpi-checkpoints/libero_pi05_subtask_fast/my_experiment/12000/params"
        ),
        
        data=LeRobotLiberoSubtaskDataConfig(
            repo_id="KeWangRobotics/libero_10_subtasks",
            base_config=DataConfig(
                asset_id="libero_subtask",
                use_quantile_norm=True,  # ⭐ Use quantile normalization for gripper actions
            ),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=3000,
            peak_lr=2.5e-5,
            decay_steps=150_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=8_000,
        save_interval=2000,
        batch_size=512,
        fsdp_devices=8,
        ema_decay=0.999,
        wandb_enabled=True,

        freeze_filter=nnx.All(
         nnx.Param,
         nnx_utils.PathRegex(".*llm.*"),             # match all LLM layers
         nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*")) # exclude action expert branch
     )
    ),
    
    # Mode 3: Subtask + FAST + Flow (Hybrid - All three losses)
    TrainConfig(
        name="libero_pi05_subtask_hybrid",
        exp_name="libero_subtask_hybrid",
        model=pi05_config.Pi05Config(
            action_horizon=20,
            max_token_len=192,
            discrete_state_input=False,
            # ⭐ Use all three losses
            subtask_loss_weight=0.15,
            fast_token_loss_weight=0.15,  # Lower weight for FAST tokens
            flow_matching_loss_weight=1.0,  # Lower weight for flow matching
            fast_tokenizer_path="weights/fast",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("weights/pi05_base/params"),
        data=LeRobotLiberoSubtaskDataConfig(
            repo_id="KeWangRobotics/libero_10_subtasks",
            base_config=DataConfig(
                asset_id="libero_subtask",
                use_quantile_norm=True,  # ⭐ Use quantile normalization for gripper actions
            ),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=3000,
            peak_lr=2.5e-5,
            decay_steps=150_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=40_000,
        save_interval=5000,
        batch_size=64,
        fsdp_devices=1,
        ema_decay=0.999,
    ),

    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(prompt_from_task=True),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(prompt_from_task=True),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(prompt_from_task=True),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    # Reward recap configs for bin-pack.
    # Keep only the canonical positive_only and mixed variants, both starting from pi05 base weights.
    TrainConfig(
        name="pi05_bin_pack_coffee_capsules_recap_positive_only",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50),
        data=LeRobotBinPackDataConfig(
            repo_id=(
                "["
                "villekuosmanen/bin_pick_pack_coffee_capsules, "
                "villekuosmanen/dAgger_bin_pick_pack_coffee_capsules_1.0.0, "
                "villekuosmanen/dAgger_bin_pick_pack_coffee_capsules_1.1.0, "
                "villekuosmanen/dAgger_bin_pick_pack_coffee_capsules_1.2.0, "
                "villekuosmanen/dAgger_bin_pick_pack_coffee_capsules_1.3.1, "
                "villekuosmanen/dAgger_bin_pick_pack_coffee_capsules_1.4.0, "
                "villekuosmanen/dAgger_bin_pick_pack_coffee_capsules_1.5.0, "
                "villekuosmanen/dAgger_bin_pick_pack_coffee_capsules_1.5.1, "
                "villekuosmanen/dAgger_bin_pick_pack_coffee_capsules_1.7.0"
                "]"
            ),
            base_config=DataConfig(prompt_from_task=True),
            use_control_mode_advantage_prompt=True,
            advantage_prompt_mode="positive_only",
            advantage_dropout_rate=0.3,
            use_delta_actions=True,
            output_delta_actions=True,
        ),
        batch_size=36,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("weights/pi05_base/params"),
        num_train_steps=50_000,
    ),
    TrainConfig(
        name="pi05_bin_pack_coffee_capsules_recap_mixed",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50),
        data=LeRobotBinPackDataConfig(
            repo_id=(
                "["
                "villekuosmanen/bin_pick_pack_coffee_capsules, "
                "villekuosmanen/dAgger_bin_pick_pack_coffee_capsules_1.0.0, "
                "villekuosmanen/dAgger_bin_pick_pack_coffee_capsules_1.1.0, "
                "villekuosmanen/dAgger_bin_pick_pack_coffee_capsules_1.2.0, "
                "villekuosmanen/dAgger_bin_pick_pack_coffee_capsules_1.3.1, "
                "villekuosmanen/dAgger_bin_pick_pack_coffee_capsules_1.4.0, "
                "villekuosmanen/dAgger_bin_pick_pack_coffee_capsules_1.5.0, "
                "villekuosmanen/dAgger_bin_pick_pack_coffee_capsules_1.5.1, "
                "villekuosmanen/dAgger_bin_pick_pack_coffee_capsules_1.7.0"
                "]"
            ),
            base_config=DataConfig(prompt_from_task=True),
            use_control_mode_advantage_prompt=True,
            advantage_prompt_mode="mixed",
            advantage_dropout_rate=0.3,
            use_delta_actions=True,
            output_delta_actions=True,
        ),
        batch_size=36,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("weights/pi05_base/params"),
        num_train_steps=50_000,
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instructions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="/mnt/pi-data/kevin",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # SO101 stacking rings config.
    #
    TrainConfig(
        name="pi05_so101_stacking_rings",
        project_name="so101_stacking_rings",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30),
        data=LeRobotSO101DataConfig(
            repo_id="lorenzouttini/so101_stacking_rings",
            default_prompt="stack the rings",
            use_delta_actions=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("weights/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=100_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=50_000,
        save_interval=5000,
        batch_size=32,
        ema_decay=0.999,
        wandb_enabled=True,
    ),
    #
    # SO101 stacking big rings config.
    #
    TrainConfig(
        name="pi05_so101_stacking_big_rings",
        project_name="so101_stacking_big_rings",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30),
        data=LeRobotSO101DataConfig(
            repo_id="lorenzouttini/so101_stacking_big_rings",
            default_prompt="stack the big rings",
            use_delta_actions=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("weights/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=100_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=50_000,
        save_interval=5000,
        batch_size=32,
        ema_decay=0.999,
        wandb_enabled=True,
    ),
    #
    # ARX5 multi-task foundation model configs.
    #
    TrainConfig(
        name="pi05_arx5_multitask_v1",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50),
        data=LeRobotARX5MultiTaskDataConfig(
            # JSON listing 186 repo_ids; resolved against --assets-dir at runtime
            repo_id="training_mix_v1.json",
            base_config=DataConfig(prompt_from_task=True),
            use_delta_actions=True,
        ),
        batch_size=36,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("weights/pi05_base/params"),
        num_train_steps=100_000,
        wandb_enabled=True,
    ),
    #
    # ARX5 micro ablation configs (14-dataset subset).
    # Both share the same dataset mix; they differ only in valid-index filtering.
    # Norm stats are loaded from the baseline assets dir for both.
    #
    TrainConfig(
        name="pi05_arx5_multitask_micro_baseline",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50),
        data=LeRobotARX5MultiTaskDataConfig(
            repo_id="training_mix_micro.json",
            base_config=DataConfig(prompt_from_task=True),
            use_delta_actions=True,
        ),
        batch_size=36,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("weights/pi05_base/params"),
        num_train_steps=30_000,
        wandb_enabled=True,
    ),
    TrainConfig(
        name="pi05_arx5_multitask_micro_advantaged",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50),
        data=LeRobotARX5MultiTaskDataConfig(
            repo_id="training_mix_micro.json",
            base_config=DataConfig(prompt_from_task=True),
            use_delta_actions=True,
        ),
        batch_size=36,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("weights/pi05_base/params"),
        num_train_steps=30_000,
        wandb_enabled=True,
    ),
    #
    # Build-block-tower joints-only ablation: train only the first 7 (joint)
    # dims of the 17D canonical action vector. EE channels of state/action are
    # zeroed in the input pipeline and the action_dim_mask forces the
    # flow-matching loss to ignore EE dims. At inference, BlockTowerOutputs
    # slices the policy output to 7D. Mirrors the historical
    # pi05_build_block_tower_baseline schedule on the 6-dataset mix.
    #
    TrainConfig(
        name="pi05_build_block_tower_baseline_6mix_joints_only",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50),
        data=LeRobotBlockTowerDataConfig(
            repo_id=_BLOCK_TOWER_6MIX_REPO_ID,
            base_config=DataConfig(prompt_from_task=True),
            use_delta_actions=True,
            output_delta_actions=True,
            joints_only=True,
        ),
        batch_size=36,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("weights/pi05_base/params"),
        num_train_steps=50_000,
    ),
    #
    # Build-block-tower reward recap configs.
    #
    TrainConfig(
        name="pi05_build_block_tower_recap_positive_only",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50),
        data=LeRobotBlockTowerDataConfig(
            repo_id=_BLOCK_TOWER_6MIX_REPO_ID,
            base_config=DataConfig(prompt_from_task=True),
            use_control_mode_advantage_prompt=True,
            advantage_prompt_mode="positive_only",
            advantage_dropout_rate=0.3,
            use_delta_actions=True,
            output_delta_actions=True,
        ),
        batch_size=36,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("weights/pi05_base/params"),
        num_train_steps=50_000,
    ),
    TrainConfig(
        name="pi05_build_block_tower_recap_mixed",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50),
        data=LeRobotBlockTowerDataConfig(
            repo_id=_BLOCK_TOWER_6MIX_REPO_ID,
            base_config=DataConfig(prompt_from_task=True),
            use_control_mode_advantage_prompt=True,
            advantage_prompt_mode="mixed",
            advantage_dropout_rate=0.3,
            use_delta_actions=True,
            output_delta_actions=True,
        ),
        batch_size=36,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("weights/pi05_base/params"),
        num_train_steps=50_000,
    ),
    TrainConfig(
        name="pi05_build_block_tower_subtask_recap_positive_only",
        model=pi05_config.Pi05Config(
            action_horizon=50,
            subtask_loss_weight=1.0,
            fast_token_loss_weight=0.0,
            flow_matching_loss_weight=1.0,
        ),
        data=LeRobotBlockTowerSubtaskDataConfig(
            repo_id=_BLOCK_TOWER_6MIX_REPO_ID,
            base_config=DataConfig(prompt_from_task=True),
            use_control_mode_advantage_prompt=True,
            advantage_prompt_mode="positive_only",
            advantage_dropout_rate=0.3,
            use_delta_actions=True,
            output_delta_actions=True,
        ),
        batch_size=36,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("weights/pi05_base/params"),
        num_train_steps=50_000,
    ),
    TrainConfig(
        name="pi05_build_block_tower_subtask_recap_mixed",
        model=pi05_config.Pi05Config(
            action_horizon=50,
            subtask_loss_weight=1.0,
            fast_token_loss_weight=0.0,
            flow_matching_loss_weight=1.0,
        ),
        data=LeRobotBlockTowerSubtaskDataConfig(
            repo_id=_BLOCK_TOWER_6MIX_REPO_ID,
            base_config=DataConfig(prompt_from_task=True),
            use_control_mode_advantage_prompt=True,
            advantage_prompt_mode="mixed",
            advantage_dropout_rate=0.3,
            use_delta_actions=True,
            output_delta_actions=True,
        ),
        batch_size=36,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("weights/pi05_base/params"),
        num_train_steps=50_000,
    ),
    #
    # RL Token (RLT Stage 1) configs.
    #
    TrainConfig(
        name="pi05_rlt_build_block_tower_6mix",
        model=pi0_rl_config.Pi0RLConfig(pi05=True, action_horizon=50, rl_vla_loss_weight=0.0),
        data=LeRobotBlockTowerDataConfig(
            repo_id=_BLOCK_TOWER_6MIX_REPO_ID,
            base_config=DataConfig(
                prompt_from_task=True,
                episode_split=EpisodeSplitConfig(val_ratio=0.1, seed=42),
            ),
            use_delta_actions=True,
            output_delta_actions=True,
        ),
        batch_size=36,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=10_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        freeze_filter=pi0_rl_config.Pi0RLConfig(
            pi05=True, action_horizon=50, rl_vla_loss_weight=0.0
        ).get_rl_freeze_filter(),
        weight_loader=weight_loaders.RLTokenCheckpointWeightLoader(
            "checkpoints/pi05_build_block_tower_baseline_6mix/retain/step_49999/alpha_0.5/params"
        ),
        num_train_steps=50_000,
        val_interval=1000,
        val_num_batches=10,
    ),
    TrainConfig(
        name="pi05_rlt_build_block_tower_6mix_joints_only",
        model=pi0_rl_config.Pi0RLConfig(pi05=True, action_horizon=50, rl_vla_loss_weight=0.0),
        data=LeRobotBlockTowerDataConfig(
            repo_id=_BLOCK_TOWER_6MIX_REPO_ID,
            base_config=DataConfig(
                prompt_from_task=True,
                episode_split=EpisodeSplitConfig(val_ratio=0.1, seed=42),
            ),
            use_delta_actions=True,
            output_delta_actions=True,
            joints_only=True,
        ),
        batch_size=36,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=10_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        freeze_filter=pi0_rl_config.Pi0RLConfig(
            pi05=True, action_horizon=50, rl_vla_loss_weight=0.0
        ).get_rl_freeze_filter(),
        weight_loader=weight_loaders.RLTokenCheckpointWeightLoader(
            "checkpoints/pi05_build_block_tower_baseline_6mix_joints_only/joints_only/49999/params"
        ),
        num_train_steps=50_000,
        val_interval=1000,
        val_num_batches=10,
    ),
    TrainConfig(
        name="pi05_rl_token_bin_pack_coffee_capsules",
        model=pi0_rl_config.Pi0RLConfig(pi05=True, action_horizon=50),
        data=LeRobotBinPackDataConfig(
            repo_id=(
                "["
                "villekuosmanen/bin_pick_pack_coffee_capsules, "
                "villekuosmanen/bin_pick_pack_coffee_capsules_continuous, "
                "villekuosmanen/dAgger_bin_pick_pack_coffee_capsules_1.0.0, "
                "villekuosmanen/dAgger_bin_pick_pack_coffee_capsules_1.1.0, "
                "villekuosmanen/dAgger_bin_pick_pack_coffee_capsules_1.2.0, "
                "villekuosmanen/dAgger_bin_pick_pack_coffee_capsules_1.3.1, "
                "villekuosmanen/dAgger_bin_pick_pack_coffee_capsules_1.4.0, "
                "villekuosmanen/dAgger_bin_pick_pack_coffee_capsules_1.5.0, "
                "villekuosmanen/dAgger_bin_pick_pack_coffee_capsules_1.5.1, "
                "villekuosmanen/free_play_bin_pick_pack_coffee_capsules"
                "]"
            ),
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=1,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.RLTokenCheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=10_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug_pi0_rl",
        model=pi0_rl_config.Pi0RLConfig(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi0_rl",
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
