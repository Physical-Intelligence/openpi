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
from typing_extensions import override
import tyro

import os

import openpi
OPENPI_ROOT = pathlib.Path(openpi.__file__).parent.resolve()
REPO_ROOT = OPENPI_ROOT / '..' / '..'
import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.pi0_fuse as pi0_fuse
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
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
    # Local repo path, if we don't want to use huggingface cache
    repo_path: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

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

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()


@dataclasses.dataclass(frozen=True)
class AtomicDataConfig(DataConfig):
    """Extended DataConfig for Atomic datasets with reasoning support."""
    seed: int = 42
    use_reasoning: bool = True
    reasoning_json_path: str | None = None
    use_outdated_reasoning: bool = True


@dataclasses.dataclass(frozen=True)
class LiberoTraceDataConfig(DataConfig):
    """Extended DataConfig for the TraceVLA model on LIBERO.

    Carries paths to both annotation files (skill segments + trace targets/EE) and
    training-time hyperparameters specific to trace generation:

      - ``trace_horizon`` / ``trace_resample_method`` define the (N, 2) trace shape
        and how variable-length segments are downsampled to it.
      - ``h_train_max`` is the maximum anchor age used during training (matches the
        receding-horizon design; the deployment re-plan period F must be ≤ this).
      - ``scene_dropout_rate`` is the probability to mask out the scene image while
        keeping the trace overlay/anchors clean.
      - ``overlay_*`` control the polyline rendering style for the action-head input.
    """

    seed: int = 42
    skill_annotations_path: str = ""
    trace_annotations_path: str = ""
    use_wrist_image: bool = True
    is_computing_norm_stats: bool = False
    action_down_sample_steps: int = 1

    # Trace head shape and resampling.
    trace_horizon: int = 20
    trace_resample_method: str = "arc_length"  # or "time_uniform"

    # Receding-horizon training: max anchor-age (in control steps).
    h_train_max: int = 15

    # Scene dropout (planning + execution).
    scene_dropout_rate: float = 0.15

    # Overlay dropout: with probability ``overlay_dropout_rate`` the execution-mode
    # overlay image is replaced with the *clean* base image (no trace polyline),
    # forcing the action head to occasionally act without a trace cue. Dual of the
    # scene dropout above; targets the inference failure mode of over-relying on
    # the trace overlay. Independent draw from the scene dropout.
    overlay_dropout_rate: float = 0.10

    # Smooth low-frequency perturbation applied to the overlay trace (only — the
    # supervised trace target is left untouched). Sigma is drawn uniformly from
    # ``[0, trace_perturb_max_sigma]`` per sample, so a fraction of samples land
    # near-clean and the rest get progressively bent. Units are normalized [0, 1]
    # image-space; ~0.03 ≈ 7 px on a 224×224 image.
    trace_perturb_max_sigma: float = 0.03
    trace_perturb_num_freqs: int = 3

    # Overlay rendering.
    overlay_color: tuple[int, int, int] = (0, 255, 255)
    overlay_thickness: int = 2
    overlay_endpoint_radius: float = 2.5


@dataclasses.dataclass(frozen=True)
class LiberoTargetDataConfig(DataConfig):
    """Extended DataConfig for the TargetVLA family on LIBERO (trace-free).

    Carries paths to the same two annotation files as :class:`LiberoTraceDataConfig`
    (the trace-annotations file is the source of the per-segment ``semantic_target``
    point that this family conditions on; we ignore its end-effector trace polylines)
    but **omits** every trace-/overlay-specific hyperparameter:

      - no ``trace_horizon`` / ``trace_resample_method`` (no trace target)
      - no ``h_train_max`` (no anchor-age augmentation)
      - no ``scene_dropout_rate`` / ``overlay_dropout_rate`` /
        ``trace_perturb_*`` (no trace-related augmentations)
      - no ``overlay_*`` rendering options (no overlay image)
    """

    seed: int = 42
    skill_annotations_path: str = ""
    trace_annotations_path: str = ""
    use_wrist_image: bool = True
    is_computing_norm_stats: bool = False
    action_down_sample_steps: int = 1


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


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
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FUSE:
                return _transforms.Group(
                    inputs=[
                        _transforms.ResizeImages(224, 224),
                        _transforms.FuseTokenizePrompt(
                            _tokenizer.FusePaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=True,
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
class AtomicModelTransformFactory(GroupFactory):
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
                        _transforms.AtomicTokenizePrompt(
                            _tokenizer.AtomicPaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                    outputs=[
                        _transforms.ExtractThoughts(
                            _tokenizer.AtomicPaligemmaTokenizer(model_config.max_token_len),
                        ),
                    ]
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0AtomicConfig)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.AtomicTokenizePrompt(
                            _tokenizer.AtomicPaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                    outputs=[
                        _transforms.ExtractThoughts(
                            _tokenizer.AtomicPaligemmaTokenizer(model_config.max_token_len),
                        ),
                    ]
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
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
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

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )

@dataclasses.dataclass(frozen=True)
class CalvinDataConfig(DataConfig):
    """Extended data config for CALVIN with CALVIN task annotations."""
    action_down_sample_steps: int = 1
    getitem_type: str = "necessary"
    use_wrist_image: bool = True
    is_computing_norm_stats: bool = False
    use_val_dataset: bool = True
    val_ratio: float = 0.0
    create_train_val_split: bool = False
    seed: int = 42
    norm_stats_dir: str = ""


@dataclasses.dataclass(frozen=True)
class AtomicCalvinDataConfig(AtomicDataConfig):
    """Extended data config for AtomicVLA training on CALVIN annotations."""
    action_down_sample_steps: int = 1
    getitem_type: str = "necessary"
    use_wrist_image: bool = True
    is_computing_norm_stats: bool = False
    use_val_dataset: bool = True
    val_ratio: float = 0.0
    create_train_val_split: bool = False
    norm_stats_dir: str = ""


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
            inputs=[libero_policy.LiberoReasonInputs(model_type=model_config.model_type)],
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

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LiberoReasonDataConfig(DataConfig):
    """Extended data config for LIBERO with reasoning annotations."""
    action_down_sample_steps: int = 1
    getitem_type: str = "necessary"
    use_reasoning: bool = True
    use_wrist_image: bool = True
    use_history: bool = False
    use_outdated_reasoning: bool = True
    is_computing_norm_stats: bool = False
    reasoning_json_path: str | None = None
    use_val_dataset: bool = True
    val_ratio: float = 0.1
    create_train_val_split: bool = False
    seed: int = 42
    norm_stats_dir: str = ""

@dataclasses.dataclass(frozen=True)
class LiberoSkillReasonDataConfig (DataConfig):
    """Extended data config for LIBERO with reasoning annotations."""
    action_down_sample_steps: int = 1
    getitem_type: str = "necessary"
    use_reasoning: bool = True
    use_wrist_image: bool = True
    use_history: bool = False
    use_outdated_reasoning: bool = True
    is_computing_norm_stats: bool = False
    reasoning_json_path: str | None = None
    use_val_dataset: bool = True
    val_ratio: float = 0.1
    create_train_val_split: bool = False
    seed: int = 42
    norm_stats_dir: str = ""


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoReasonDataConfig(DataConfigFactory):
    """Data config factory for LIBERO with reasoning annotations (for Pi0Fuse)."""

    #base_config: tyro.conf.Suppress[LiberoReasonDataConfig | DataConfig | None] = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Why is this the reverse of the Libero config?
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoReasonInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        model_transforms = _transforms.Group(
            inputs=[
                _transforms.ResizeImages(224, 224),
                _transforms.FuseTokenizePrompt(
                    _tokenizer.FusePaligemmaTokenizer(model_config.max_token_len),
                    discrete_state_input=True,
                ),
                _transforms.PadStatesAndActions(model_config.action_dim),
            ],
        )

        base = self.create_base_config(assets_dirs, model_config)
        if base.norm_stats is None:
            base = dataclasses.replace(base, norm_stats={})
        return dataclasses.replace(
            base,
            repack_transforms=_transforms.Group(),
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotTraceVLADataConfig(DataConfigFactory):
    """Data factory for TraceVLA on LIBERO.

    Produces a ``LiberoTraceDataConfig`` with the right repack + data + model transforms
    for the TraceVLA model. The transforms pack the trace fields (``semantic_target_xy``,
    ``current_ee_xy``, ``future_trace_xy``, etc.) and the overlay image into the model input.
    """

    base_config: tyro.conf.Suppress[LiberoTraceDataConfig | None] = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Local imports to avoid a heavy import cycle at module load time.
        import openpi.policies.libero_trace_policy as libero_trace_policy  # noqa: PLC0415
        from openpi.models import pi0_trace_vla_config as pi0_trace_vla_config  # noqa: PLC0415

        if not isinstance(model_config, pi0_trace_vla_config.Pi0TraceVLAConfig):
            raise TypeError(
                f"LeRobotTraceVLADataConfig expects a Pi0TraceVLAConfig model_config, got {type(model_config).__name__}"
            )

        # Inputs: pack the dataset's dict into the model's expected shape.
        data_transforms = _transforms.Group(
            inputs=[libero_trace_policy.LiberoTraceInputs(model_type=model_config.model_type)],
            outputs=[libero_trace_policy.LiberoTraceOutputs()],
        )

        # Model transforms: resize images (incl. overlay), tokenize prompt, pad action dims.
        model_transforms = _transforms.Group(
            inputs=[
                libero_trace_policy.TraceResizeImages(224, 224),
                libero_trace_policy.TraceTokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                    discrete_state_input=False,
                ),
                _transforms.PadStatesAndActions(model_config.action_dim),
            ],
        )

        base = self.create_base_config(assets_dirs, model_config)
        # If norm stats absent, allow training (we can use zero-norm fallback when computing norm stats).
        return dataclasses.replace(
            base,
            repack_transforms=_transforms.Group(),
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotTraceVLAActionMoeDataConfig(DataConfigFactory):
    """Data factory for the TraceVLA-actionmoe variant on LIBERO.

    Mirror of :class:`LeRobotTraceVLADataConfig` — same dataset, same input/output
    transforms, same overlay rendering. The only reason this is a separate factory
    class is the runtime type check: it accepts a
    :class:`pi0_trace_vla_actionmoe_config.Pi0TraceVLAActionMoeConfig` rather than
    the original ``Pi0TraceVLAConfig``. The produced :class:`LiberoTraceDataConfig`
    is identical in shape, so ``create_torch_dataset`` and ``compute_norm_stats``
    work unchanged.
    """

    base_config: tyro.conf.Suppress[LiberoTraceDataConfig | None] = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        import openpi.policies.libero_trace_policy as libero_trace_policy  # noqa: PLC0415
        from openpi.models import pi0_trace_vla_actionmoe_config as pi0_trace_vla_actionmoe_config  # noqa: PLC0415

        if not isinstance(model_config, pi0_trace_vla_actionmoe_config.Pi0TraceVLAActionMoeConfig):
            raise TypeError(
                f"LeRobotTraceVLAActionMoeDataConfig expects a Pi0TraceVLAActionMoeConfig "
                f"model_config, got {type(model_config).__name__}"
            )

        data_transforms = _transforms.Group(
            inputs=[libero_trace_policy.LiberoTraceInputs(model_type=model_config.model_type)],
            outputs=[libero_trace_policy.LiberoTraceOutputs()],
        )

        model_transforms = _transforms.Group(
            inputs=[
                libero_trace_policy.TraceResizeImages(224, 224),
                libero_trace_policy.TraceTokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                    discrete_state_input=False,
                ),
                _transforms.PadStatesAndActions(model_config.action_dim),
            ],
        )

        base = self.create_base_config(assets_dirs, model_config)
        return dataclasses.replace(
            base,
            repack_transforms=_transforms.Group(),
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotTraceVLAMoeDataConfig(DataConfigFactory):
    """Data factory for the combined-MoE TraceVLA variant on LIBERO.

    Mirror of :class:`LeRobotTraceVLAActionMoeDataConfig`: same dataset, same
    transforms, same overlay rendering. Separate class only so the runtime type
    check binds to :class:`pi0_trace_vla_moe_config.Pi0TraceVLAMoeConfig`. The
    produced :class:`LiberoTraceDataConfig` is identical in shape, so
    ``create_torch_dataset`` and ``compute_norm_stats`` work unchanged.
    """

    base_config: tyro.conf.Suppress[LiberoTraceDataConfig | None] = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        import openpi.policies.libero_trace_policy as libero_trace_policy  # noqa: PLC0415
        from openpi.models import pi0_trace_vla_moe_config as pi0_trace_vla_moe_config  # noqa: PLC0415

        if not isinstance(model_config, pi0_trace_vla_moe_config.Pi0TraceVLAMoeConfig):
            raise TypeError(
                f"LeRobotTraceVLAMoeDataConfig expects a Pi0TraceVLAMoeConfig "
                f"model_config, got {type(model_config).__name__}"
            )

        data_transforms = _transforms.Group(
            inputs=[libero_trace_policy.LiberoTraceInputs(model_type=model_config.model_type)],
            outputs=[libero_trace_policy.LiberoTraceOutputs()],
        )

        model_transforms = _transforms.Group(
            inputs=[
                libero_trace_policy.TraceResizeImages(224, 224),
                libero_trace_policy.TraceTokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                    discrete_state_input=False,
                ),
                _transforms.PadStatesAndActions(model_config.action_dim),
            ],
        )

        base = self.create_base_config(assets_dirs, model_config)
        return dataclasses.replace(
            base,
            repack_transforms=_transforms.Group(),
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotTargetVLAActionMoeDataConfig(DataConfigFactory):
    """Data factory for the trace-free TargetVLA-ActionMoe variant on LIBERO.

    Same dataset (LIBERO + skill annotations) as the trace family, but:

      - wires the trace-free transforms from
        :mod:`openpi.policies.libero_target_policy` (no overlay-image resize,
        no trace fields in the input pack), and
      - type-checks against
        :class:`openpi.models.pi0_target_vla_actionmoe_config.Pi0TargetVLAActionMoeConfig`.

    The produced :class:`LiberoTargetDataConfig` is the slimmed-down sibling of
    :class:`LiberoTraceDataConfig`: only the fields needed for skill + target +
    completion annotation join remain (no trace-related hyperparameters).
    """

    base_config: tyro.conf.Suppress[LiberoTargetDataConfig | None] = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        import openpi.policies.libero_target_policy as libero_target_policy  # noqa: PLC0415
        from openpi.models import pi0_target_vla_actionmoe_config as pi0_target_vla_actionmoe_config  # noqa: PLC0415

        if not isinstance(model_config, pi0_target_vla_actionmoe_config.Pi0TargetVLAActionMoeConfig):
            raise TypeError(
                f"LeRobotTargetVLAActionMoeDataConfig expects a Pi0TargetVLAActionMoeConfig "
                f"model_config, got {type(model_config).__name__}"
            )

        data_transforms = _transforms.Group(
            inputs=[libero_target_policy.LiberoTargetInputs(model_type=model_config.model_type)],
            outputs=[libero_target_policy.LiberoTargetOutputs()],
        )

        model_transforms = _transforms.Group(
            inputs=[
                libero_target_policy.TargetResizeImages(224, 224),
                libero_target_policy.TargetTokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                    discrete_state_input=False,
                ),
                _transforms.PadStatesAndActions(model_config.action_dim),
            ],
        )

        base = self.create_base_config(assets_dirs, model_config)
        return dataclasses.replace(
            base,
            repack_transforms=_transforms.Group(),
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAtomicDataConfig(DataConfigFactory):
    extra_delta_transform: bool = False
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Prepare data for policy training
        # Convert images to uint8 numpy arrays, add masks
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                        "thought": "thought",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoReasonInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )
        # Use delta actions (not for gripper)

        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = AtomicModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
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
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
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
            base_config=DataConfig(
                prompt_from_task=True,
            ),
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
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=15, discrete_state_input=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
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
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
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
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=True),
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
    TrainConfig(
        name="pi05_real_lora",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                repo_path=REPO_ROOT/"data/orange_plate",
                prompt_from_task=True
            ),
            extra_delta_transform=False,
        ),
        batch_size=64,
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
        num_train_steps=1_000,
    ),
    # Libero 100 (libero_10 + libero_90), yilin wu edition
    TrainConfig(
        name="pi05_libero_100",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=True),
        data=LeRobotLiberoDataConfig(
            repo_id="yilin-wu/libero-100",
            base_config=DataConfig(
                repo_path=REPO_ROOT/"data/libero-100",
                # repo_path="/work/nvme/bgtb/zhong2/.cache/huggingface/hub/datasets--yilin-wu--libero-100/snapshots/1384872f07707d6aa361588292068eba7698facd",
                prompt_from_task=True
            ),
            extra_delta_transform=False,
        ),
        batch_size=64,
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
        num_train_steps=100_000,        # match AtomicVLA
        keep_period=10_000,
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
    # Debugging configs.
    #
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
    # RoboArena & PolaRiS configs.
    *roboarena_config.get_roboarena_configs(),
    *polaris_config.get_polaris_configs(),
    #
    # Pi05 Fuse (LoRA) fine-tuning with reasoning (Do What You Say).
    #
    # Pi05 LoRA finetuning with reasoning (Do What You Say) - LIBERO-100
    # Uses the reasoning-annotated LIBERO dataset from "Do What You Say" paper.
    # Before running, ensure:
    #   1. Download dataset: set repo_id to your LeRobot LIBERO repo (e.g., "yilin-wu/libero-100")
    #   2. Download cot_simple.json from nvidia/libero-r-datasets/libero-100-r/cot_simple.json
    #   3. Set reasoning_json_path to "<dir>/<repo_id>/cot_simple.json"
    TrainConfig(
        name="pi05_libero_reason_lora",
        model=pi0_fuse.Pi0FuseConfig(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            action_dim=32,
            action_horizon=16,
            max_token_len=415,
            diffusion_loss_coeff=1.0,
        ),
        data=LeRobotLiberoReasonDataConfig(
            repo_id="yilin-wu/libero-100",
            base_config=LiberoReasonDataConfig(
                prompt_from_task=False,
                use_reasoning=True,
                use_wrist_image=True,
                use_history=False,
                use_outdated_reasoning=True,
                action_down_sample_steps=1,
                reasoning_json_path=REPO_ROOT/'data/libero-100/cot_simple.json',
                use_val_dataset=False,
                is_computing_norm_stats=False,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        freeze_filter=pi0_fuse.Pi0FuseConfig(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=8_000,     # 10_600 steps with 64 bs = 1 full pass
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=8_000,
            decay_lr=5e-6,
        ),
        wandb_enabled=True,
        save_interval=1_000,
        keep_period=2_000,
    ),
    # Pi05 FULL finetuning with reasoning (Do What You Say) - LIBERO-100.
    # Variant of `pi05_libero_reason_lora` that fine-tunes the entire pi05 base
    # weights instead of adapting via LoRA. Drops the `_lora` Gemma variants and
    # the freeze filter so PaliGemma backbone + action expert both train. All
    # data / loss / reasoning settings match the LoRA variant.
    # NOTE: batch_size=256 may be too large for full FT on a single node — tune
    # batch_size (and possibly peak_lr) for your hardware.
    TrainConfig(
        name="pi05_libero_reason",
        model=pi0_fuse.Pi0FuseConfig(
            pi05=True,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            action_dim=32,
            action_horizon=10,
            max_token_len=415,
            diffusion_loss_coeff=1.0,
        ),
        data=LeRobotLiberoReasonDataConfig(
            repo_id="yilin-wu/libero-100",
            base_config=LiberoReasonDataConfig(
                prompt_from_task=False,
                use_reasoning=True,
                use_wrist_image=True,
                use_history=False,
                use_outdated_reasoning=True,
                action_down_sample_steps=1,
                reasoning_json_path=REPO_ROOT/'data/libero-100/cot_simple.json',
                use_val_dataset=False,
                is_computing_norm_stats=False,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        ema_decay=0.999,
        num_train_steps=100_000,
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        wandb_enabled=True,
        save_interval=5_000,
        keep_period=10_000,
    ),
    # Pi05 LoRA finetuning with reasoning - LIBERO-10 (smaller, for testing)
    TrainConfig(
        name="pi05_libero_10_reason_lora",
        model=pi0_fuse.Pi0FuseConfig(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            action_dim=32,
            action_horizon=16,
            max_token_len=415,
            diffusion_loss_coeff=1.0,
        ),
        data=LeRobotLiberoReasonDataConfig(
            repo_id="yilin-wu/libero-10",
            base_config=LiberoReasonDataConfig(
                prompt_from_task=False,
                use_reasoning=True,
                use_wrist_image=True,
                use_history=False,
                use_outdated_reasoning=True,
                action_down_sample_steps=1,
                reasoning_json_path=REPO_ROOT/'data/libero-10/cot_simple.json',
                use_val_dataset=False,
                is_computing_norm_stats=False,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        freeze_filter=pi0_fuse.Pi0FuseConfig(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=2000,
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=100,
            peak_lr=5e-5,
            decay_steps=2000,
            decay_lr=5e-6,
        ),
        wandb_enabled=True,
        save_interval=500,
        keep_period=1000,
    ),
    # New training config for skill conditioned reasoning
    TrainConfig(
        name="pi05_libero_skill_reason_lora",
        model=pi0_fuse.Pi0FuseConfig(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            action_dim=32,
            action_horizon=16,
            max_token_len=415,
            diffusion_loss_coeff=1.0,
        ),
        data=LeRobotLiberoReasonDataConfig(
            repo_id="yilin-wu/libero-100",
            base_config=LiberoSkillReasonDataConfig(
                prompt_from_task=False,
                use_reasoning=True,
                use_wrist_image=True,
                use_history=False,
                use_outdated_reasoning=True,
                action_down_sample_steps=1,
                reasoning_json_path=REPO_ROOT/'data/libero-100/cot_skill_fixed.json',
                use_val_dataset=False,
                is_computing_norm_stats=False,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        freeze_filter=pi0_fuse.Pi0FuseConfig(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=100_000,     # 10_600 steps with 64 bs = 1 full pass
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-6,
        ),
        wandb_enabled=True,
        save_interval=5000,
        keep_period=10_000,
    ),

    # New training config for skill conditioned reasoning - in v2 we use simpler reasoning format
    TrainConfig(
        name="pi05_libero_skill_reason_lora_v2",
        model=pi0_fuse.Pi0FuseConfig(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            action_dim=32,
            action_horizon=16,
            max_token_len=415,
            diffusion_loss_coeff=1.0,
        ),
        data=LeRobotLiberoReasonDataConfig(
            repo_id="yilin-wu/libero-100",
            base_config=LiberoSkillReasonDataConfig(
                prompt_from_task=False,
                use_reasoning=True,
                use_wrist_image=True,
                use_history=False,
                use_outdated_reasoning=True,
                action_down_sample_steps=1,
                reasoning_json_path=REPO_ROOT/'data/libero-100/cot_skill_fixed2.json',
                use_val_dataset=False,
                is_computing_norm_stats=False,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        freeze_filter=pi0_fuse.Pi0FuseConfig(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=30_000,     # 10_600 steps with 64 bs = 1 full pass
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        wandb_enabled=True,
        save_interval=5000,
        keep_period=10_000,
    ),

    # Skill condition reasoning with reannotated skills
    TrainConfig(
        name="pi05_libero_skill_reason_fixed",
        model=pi0_fuse.Pi0FuseConfig(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            action_dim=32,
            action_horizon=16,
            max_token_len=415,
            diffusion_loss_coeff=1.0,
        ),
        data=LeRobotLiberoReasonDataConfig(
            repo_id="yilin-wu/libero-100",
            base_config=LiberoSkillReasonDataConfig(
                prompt_from_task=False,
                use_reasoning=True,
                use_wrist_image=True,
                use_history=False,
                use_outdated_reasoning=True,
                action_down_sample_steps=1,
                #reasoning_json_path=REPO_ROOT/'data/libero-100/cot_skill.json',
                reasoning_json_path=REPO_ROOT/'data/libero-100/skill_annotations.json',
                use_val_dataset=False,
                is_computing_norm_stats=False,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        freeze_filter=pi0_fuse.Pi0FuseConfig(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=30_000,     # 10_600 steps with 64 bs = 1 full pass
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        wandb_enabled=True,
        save_interval=5000,
        keep_period=10_000,
    ),
    # Skill conditioned reasoning with reannotated skills, full finetuning (no LoRA adapters).
    TrainConfig(
        name="pi05_libero_skill_reason_full_finetune",
        model=pi0_fuse.Pi0FuseConfig(
            pi05=True,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            action_dim=32,
            action_horizon=16,
            max_token_len=415,
            diffusion_loss_coeff=1.0,
        ),
        data=LeRobotLiberoReasonDataConfig(
            repo_id="yilin-wu/libero-100",
            base_config=LiberoSkillReasonDataConfig(
                prompt_from_task=False,
                use_reasoning=True,
                use_wrist_image=True,
                use_history=False,
                use_outdated_reasoning=True,
                action_down_sample_steps=1,
                reasoning_json_path=REPO_ROOT/'data/libero-100/skill_annotations.json',
                use_val_dataset=False,
                is_computing_norm_stats=False,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        assets_base_dir=str(REPO_ROOT / "assets"),
        ema_decay=None,
        num_train_steps=100_000,     # 10_600 steps with 64 bs = 1 full pass
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-6,
        ),
        wandb_enabled=True,
        save_interval=5000,
        keep_period=10_000,
    ),
    # Skill condition reasoning with reannotated skills
    TrainConfig(
        name="pi05_libero_skill_reason_target",
        model=pi0_fuse.Pi0FuseConfig(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            action_dim=32,
            action_horizon=16,
            max_token_len=415,
            diffusion_loss_coeff=1.0,
        ),
        data=LeRobotLiberoReasonDataConfig(
            repo_id="yilin-wu/libero-100",
            base_config=LiberoSkillReasonDataConfig(
                prompt_from_task=False,
                use_reasoning=True,
                use_wrist_image=True,
                use_history=False,
                use_outdated_reasoning=True,
                action_down_sample_steps=1,
                #reasoning_json_path=REPO_ROOT/'data/libero-100/cot_skill.json',
                reasoning_json_path=REPO_ROOT/'data/libero-100/skills_with_targets.json',
                use_val_dataset=False,
                is_computing_norm_stats=False,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        freeze_filter=pi0_fuse.Pi0FuseConfig(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=30_000,     # 10_600 steps with 64 bs = 1 full pass
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        wandb_enabled=True,
        save_interval=5_000,
        keep_period=10_000,
    ),
    # Skill condition reasoning with reannotated skills, action head only
    TrainConfig(
        name="pi05_libero_skill_reason_target_action",
        model=pi0_fuse.Pi0FuseConfig(
            pi05=True,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m_lora",
            action_dim=32,
            action_horizon=16,
            max_token_len=415,
            diffusion_loss_coeff=1.0,
        ),
        data=LeRobotLiberoReasonDataConfig(
            repo_id="yilin-wu/libero-100",
            base_config=LiberoSkillReasonDataConfig(
                prompt_from_task=False,
                use_reasoning=True,
                use_wrist_image=True,
                use_history=False,
                use_outdated_reasoning=True,
                action_down_sample_steps=1,
                #reasoning_json_path=REPO_ROOT/'data/libero-100/cot_skill.json',
                reasoning_json_path=REPO_ROOT/'data/libero-100/skills_with_targets.json',
                use_val_dataset=False,
                is_computing_norm_stats=False,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        freeze_filter=pi0_fuse.Pi0FuseConfig(
            pi05=True,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(freeze_llm=True),
        ema_decay=None,
        num_train_steps=30_000,     # 10_600 steps with 64 bs = 1 full pass
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        wandb_enabled=True,
        save_interval=5_000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="Atomic_libero",
        model=pi0_config.Pi0AtomicConfig(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotAtomicDataConfig(
            repo_id="yilin-wu/libero-100",
            base_config=AtomicDataConfig(
                prompt_from_task=False,
                repo_path="/work/nvme/bgtb/zhong2/.cache/huggingface/hub/datasets--yilin-wu--libero-100",
                use_reasoning=True,
                reasoning_json_path=REPO_ROOT / "data/libero-100/skill_annotations.json",
            ),
        ),
        assets_base_dir=str(REPO_ROOT / "assets"),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=140_000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="atomic_calvin",
        model=pi0_config.Pi0AtomicConfig(
            pi05=True,
            action_expert_variant="moe_gemma_8",
            action_horizon=10,
            discrete_state_input=False,
        ),
        data=LeRobotAtomicDataConfig(
            repo_id="fywang/calvin-task-ABC-D-lerobot",
            base_config=AtomicCalvinDataConfig(
                prompt_from_task=False,
                repo_path="/work/nvme/bgtb/zhong2/.cache/huggingface/hub/datasets--fywang--calvin-task-ABC-D-lerobot/snapshots/b3d4ef71226a5fb359f05eeb7036c3caafc3a3c1",
                use_reasoning=True,
                reasoning_json_path=REPO_ROOT / "data/calvin-task-ABC-D-lerobot/skill_annotations_calvin.json",
            ),
        ),
        assets_base_dir=str(REPO_ROOT / "assets"),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=130_000,
        keep_period=10_000,
    ),
    # AtomicVLA on the physical-robot table-tasks dataset (n5zhong/table_tasks).
    # Three skills (PICKUP_FROM, PLACE_ON, PLACE_IN) routed onto 2 atomic-skill
    # experts + 1 shared expert (see embed_sigma() in models/tokenizer.py).
    TrainConfig(
        name="atomic_table_tasks",
        model=pi0_config.Pi0AtomicConfig(
            pi05=True,
            action_expert_variant="moe_gemma_2",
            action_horizon=10,
            discrete_state_input=False,
        ),
        data=LeRobotAtomicDataConfig(
            repo_id="n5zhong/table_tasks",
            base_config=AtomicDataConfig(
                prompt_from_task=False,
                repo_path="~/.cache/huggingface/hub/datasets--n5zhong--table_tasks",
                use_reasoning=True,
                reasoning_json_path=REPO_ROOT / "data/table-tasks/tabletask_skill_annotations.json",
            ),
        ),
        assets_base_dir=str(REPO_ROOT / "assets"),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-4,
            decay_steps=40_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=50_000,
        keep_period=2_000,
    ),
    TrainConfig(
        name="pi05_calvin",
        model=pi0_fuse.Pi0FuseConfig(
            pi05=True,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            action_dim=32,
            action_horizon=10,
            max_token_len=256,
            diffusion_loss_coeff=1.0,
        ),
        data=LeRobotLiberoReasonDataConfig(
            repo_id="fywang/calvin-task-ABC-D-lerobot",
            base_config=CalvinDataConfig(
                prompt_from_task=False,
                repo_path=REPO_ROOT/"data/calvin-task-ABC-D-lerobot"
            ),
        ),
        freeze_filter=pi0_fuse.Pi0FuseConfig(
            pi05=True,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
        ).get_freeze_filter(),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        wandb_enabled=True,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        #num_train_steps=30_000,
        num_train_steps=100_000,     # Bumping it up to atomicVLA setting. We will train everything at 100k
        keep_period=10_000
    ),
    # ============================================================
    # Trace-augmented VLA (Pi0TraceVLA) - full finetune and LoRA.
    # ============================================================
    TrainConfig(
        name="trace_vla",
        # Full finetune: gemma_2b for the VLM, gemma_300m for the action head, full FT trace MoE.
        model=__import__(
            "openpi.models.pi0_trace_vla_config", fromlist=["Pi0TraceVLAConfig"]
        ).Pi0TraceVLAConfig(
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            trace_expert_variant="trace_moe_gemma_300m",
            action_horizon=10,
            pi05=True,
            discrete_state_input=False,
            max_token_len=200,
            trace_horizon=20,
            num_trace_experts=5,
        ),
        data=LeRobotTraceVLADataConfig(
            repo_id="yilin-wu/libero-100",
            base_config=LiberoTraceDataConfig(
                repo_path="/work/nvme/bgtb/zhong2/.cache/huggingface/hub/datasets--yilin-wu--libero-100/snapshots/1384872f07707d6aa361588292068eba7698facd",
                prompt_from_task=True,
                skill_annotations_path=str(REPO_ROOT / "data/libero-100/skill_annotations.json"),
                trace_annotations_path=str(REPO_ROOT / "data/libero-100/skill_target_traces.json"),
                use_wrist_image=True,
                is_computing_norm_stats=False,
            ),
        ),
        assets_base_dir=str(REPO_ROOT / "assets"),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=110_000,
        save_interval=5_000,
        keep_period=10_000,
        log_interval=100,
        wandb_enabled=True,
    ),
    TrainConfig(
        name="trace_vla_lora",
        # LoRA finetune: gemma_2b_lora + gemma_300m_lora; trace expert and completion head are full FT.
        model=__import__(
            "openpi.models.pi0_trace_vla_config", fromlist=["Pi0TraceVLAConfig"]
        ).Pi0TraceVLAConfig(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            trace_expert_variant="trace_moe_gemma_300m",
            action_horizon=10,
            pi05=True,
            discrete_state_input=False,
            max_token_len=200,
            trace_horizon=20,
            num_trace_experts=5,
        ),
        data=LeRobotTraceVLADataConfig(
            repo_id="yilin-wu/libero-100",
            base_config=LiberoTraceDataConfig(
                repo_path="/work/nvme/bgtb/zhong2/.cache/huggingface/hub/datasets--yilin-wu--libero-100/snapshots/1384872f07707d6aa361588292068eba7698facd",
                prompt_from_task=True,
                skill_annotations_path=str(REPO_ROOT / "data/libero-100/skill_annotations.json"),
                trace_annotations_path=str(REPO_ROOT / "data/libero-100/skill_target_traces.json"),
                use_wrist_image=True,
                is_computing_norm_stats=False,
            ),
        ),
        assets_base_dir=str(REPO_ROOT / "assets"),
        # Compute the freeze filter from the matching model config.
        freeze_filter=__import__(
            "openpi.models.pi0_trace_vla_config", fromlist=["Pi0TraceVLAConfig"]
        ).Pi0TraceVLAConfig(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            trace_expert_variant="trace_moe_gemma_300m",
            action_horizon=10,
            pi05=True,
            discrete_state_input=False,
            max_token_len=200,
            trace_horizon=20,
            num_trace_experts=5,
        ).get_freeze_filter(),
        ema_decay=None,
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2e-4,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        save_interval=5_000,
        keep_period=10_000,
        log_interval=100,
        wandb_enabled=True,
    ),
    # ============================================================
    # TraceVLA-actionmoe (Pi0TraceVLAActionMoe) - full FT and LoRA.
    # ============================================================
    # Architectural swap vs. trace_vla: action expert is the 5-expert hard-routed
    # MoE; trace expert is a single dense gemma_300m FFN. Everything else
    # (conditioning, training tricks, dataset, transforms, completion head) is
    # the same as the latest trace_vla.
    TrainConfig(
        name="trace_vla_actionmoe",
        model=__import__(
            "openpi.models.pi0_trace_vla_actionmoe_config", fromlist=["Pi0TraceVLAActionMoeConfig"]
        ).Pi0TraceVLAActionMoeConfig(
            paligemma_variant="gemma_2b",
            action_expert_variant="trace_moe_gemma_300m",   # 5-expert MoE for actions
            trace_expert_variant="gemma_300m",              # single dense FFN for traces
            action_horizon=10,
            pi05=True,
            discrete_state_input=False,
            max_token_len=200,
            trace_horizon=20,
            num_action_experts=5,
        ),
        data=LeRobotTraceVLAActionMoeDataConfig(
            repo_id="yilin-wu/libero-100",
            base_config=LiberoTraceDataConfig(
                repo_path=str(REPO_ROOT / "data/libero-100"),
                prompt_from_task=True,
                skill_annotations_path=str(REPO_ROOT / "data/libero-100/skill_annotations.json"),
                trace_annotations_path=str(REPO_ROOT / "data/libero-100/skill_target_traces.json"),
                use_wrist_image=True,
                is_computing_norm_stats=False,
            ),
        ),
        assets_base_dir=str(REPO_ROOT / "assets"),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=200_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=100_000,
        save_interval=5_000,
        keep_period=10_000,
        log_interval=100,
        wandb_enabled=True,
    ),
    TrainConfig(
        name="trace_vla_actionmoe_lora",
        # LoRA on paligemma 2B only. Action MoE + trace single FFN + completion head are full FT.
        model=__import__(
            "openpi.models.pi0_trace_vla_actionmoe_config", fromlist=["Pi0TraceVLAActionMoeConfig"]
        ).Pi0TraceVLAActionMoeConfig(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="trace_moe_gemma_300m",
            trace_expert_variant="gemma_300m",
            action_horizon=10,
            pi05=True,
            discrete_state_input=False,
            max_token_len=200,
            trace_horizon=20,
            num_action_experts=5,
        ),
        data=LeRobotTraceVLAActionMoeDataConfig(
            repo_id="yilin-wu/libero-100",
            base_config=LiberoTraceDataConfig(
                repo_path=str(REPO_ROOT / "data/libero-100"),
                prompt_from_task=True,
                skill_annotations_path=str(REPO_ROOT / "data/libero-100/skill_annotations.json"),
                trace_annotations_path=str(REPO_ROOT / "data/libero-100/skill_target_traces.json"),
                use_wrist_image=True,
                is_computing_norm_stats=False,
            ),
        ),
        assets_base_dir=str(REPO_ROOT / "assets"),
        freeze_filter=__import__(
            "openpi.models.pi0_trace_vla_actionmoe_config", fromlist=["Pi0TraceVLAActionMoeConfig"]
        ).Pi0TraceVLAActionMoeConfig(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="trace_moe_gemma_300m",
            trace_expert_variant="gemma_300m",
            action_horizon=10,
            pi05=True,
            discrete_state_input=False,
            max_token_len=200,
            trace_horizon=20,
            num_action_experts=5,
        ).get_freeze_filter(),
        ema_decay=None,
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2e-4,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        save_interval=5_000,
        keep_period=10_000,
        log_interval=100,
        wandb_enabled=True,
    ),
    # ============================================================
    # TraceVLA combined-MoE (Pi0TraceVLAMoe) - full FT and LoRA.
    # ============================================================
    # Both the action and the trace stream carry a 5-expert hard-routed MoE.
    # Action MoE keeps the size of `trace_moe_gemma_300m` (matches actionmoe);
    # trace MoE is the shrunk Recipe-C `trace_moe_small` (width=512, mlp_dim=2048).
    # The trace MoE is randomly initialized — its shape no longer matches pi05_base's
    # action expert, so warm-start from pi05 is infeasible for stream 2.
    # All conditioning, training tricks, dataset, transforms, completion head, and
    # losses are identical to trace_vla / trace_vla_actionmoe.
    TrainConfig(
        name="trace_vla_moe",
        model=__import__(
            "openpi.models.pi0_trace_vla_moe_config", fromlist=["Pi0TraceVLAMoeConfig"]
        ).Pi0TraceVLAMoeConfig(
            paligemma_variant="gemma_2b",
            action_expert_variant="trace_moe_gemma_300m",   # 5-expert MoE for actions (full size)
            trace_expert_variant="trace_moe_small",         # 5-expert MoE for traces (shrunk)
            action_horizon=10,
            pi05=True,
            discrete_state_input=False,
            max_token_len=200,
            trace_horizon=20,
            num_action_experts=5,
            num_trace_experts=5,
        ),
        data=LeRobotTraceVLAMoeDataConfig(
            repo_id="yilin-wu/libero-100",
            base_config=LiberoTraceDataConfig(
                repo_path=str(REPO_ROOT / "data/libero-100"),
                prompt_from_task=True,
                skill_annotations_path=str(REPO_ROOT / "data/libero-100/skill_annotations.json"),
                trace_annotations_path=str(REPO_ROOT / "data/libero-100/skill_target_traces.json"),
                use_wrist_image=True,
                is_computing_norm_stats=False,
            ),
        ),
        assets_base_dir=str(REPO_ROOT / "assets"),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=120_000,
        save_interval=5_000,
        keep_period=10_000,
        log_interval=100,
        wandb_enabled=True,
    ),
    TrainConfig(
        name="trace_vla_moe_lora",
        # LoRA on paligemma 2B only. Both expert MoEs + completion head are full FT.
        model=__import__(
            "openpi.models.pi0_trace_vla_moe_config", fromlist=["Pi0TraceVLAMoeConfig"]
        ).Pi0TraceVLAMoeConfig(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="trace_moe_gemma_300m",
            trace_expert_variant="trace_moe_small",
            action_horizon=10,
            pi05=True,
            discrete_state_input=False,
            max_token_len=200,
            trace_horizon=20,
            num_action_experts=5,
            num_trace_experts=5,
        ),
        data=LeRobotTraceVLAMoeDataConfig(
            repo_id="yilin-wu/libero-100",
            base_config=LiberoTraceDataConfig(
                repo_path=str(REPO_ROOT / "data/libero-100"),
                prompt_from_task=True,
                skill_annotations_path=str(REPO_ROOT / "data/libero-100/skill_annotations.json"),
                trace_annotations_path=str(REPO_ROOT / "data/libero-100/skill_target_traces.json"),
                use_wrist_image=True,
                is_computing_norm_stats=False,
            ),
        ),
        assets_base_dir=str(REPO_ROOT / "assets"),
        freeze_filter=__import__(
            "openpi.models.pi0_trace_vla_moe_config", fromlist=["Pi0TraceVLAMoeConfig"]
        ).Pi0TraceVLAMoeConfig(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="trace_moe_gemma_300m",
            trace_expert_variant="trace_moe_small",
            action_horizon=10,
            pi05=True,
            discrete_state_input=False,
            max_token_len=200,
            trace_horizon=20,
            num_action_experts=5,
            num_trace_experts=5,
        ).get_freeze_filter(),
        ema_decay=None,
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2e-4,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        save_interval=5_000,
        keep_period=10_000,
        log_interval=100,
        wandb_enabled=True,
    ),
    # ============================================================
    # TraceVLA combined-MoE on the physical-robot table-tasks dataset (full FT).
    # ============================================================
    # Mirror of ``trace_vla_moe`` (full FT, both action + trace streams are hard-routed
    # MoE), retargeted from LIBERO to the physical-robot ``n5zhong/table_tasks`` dataset.
    # Only the dataset-driven essentials change vs ``trace_vla_moe``:
    #   - K=2 skill experts instead of 5 (the 3 table-task skills PICKUP_FROM / PLACE_ON /
    #     PLACE_IN route onto 2 experts per ``embed_sigma`` / ``trace_utils.skill_to_expert_id``:
    #     PICKUP_FROM -> 0, PLACE_ON/PLACE_IN -> 1). Both MoE streams use the new 2-expert
    #     variants (``trace_moe_gemma_300m_2e`` / ``trace_moe_small_2e``).
    #   - dataset = ``n5zhong/table_tasks`` + the table-task skill/trace annotations.
    # The table_tasks images are stored as MP4 ``video`` features (vs LIBERO's in-parquet
    # ``image`` features); ``LiberoTraceDataset`` decodes them transparently (guarded by
    # ``self.meta.video_keys``). Trace coords live in the per-episode 640x480 space recorded
    # in the trace annotations, and are normalized to [0, 1] before overlay/supervision —
    # so the 480x640 camera resolution needs no special handling. All conditioning, training
    # tricks (anchor-age augmentation, scene/overlay dropout, trace perturbation, image
    # augmentation), completion head, and losses are identical to ``trace_vla_moe``.
    TrainConfig(
        name="trace_vla_moe_table_tasks",
        model=__import__(
            "openpi.models.pi0_trace_vla_moe_config", fromlist=["Pi0TraceVLAMoeConfig"]
        ).Pi0TraceVLAMoeConfig(
            paligemma_variant="gemma_2b",
            action_expert_variant="trace_moe_gemma_300m_2e",   # 2-expert MoE for actions (full size)
            trace_expert_variant="trace_moe_small_2e",         # 2-expert MoE for traces (shrunk)
            action_horizon=10,
            pi05=True,
            discrete_state_input=False,
            max_token_len=200,
            trace_horizon=20,
            num_action_experts=2,
            num_trace_experts=2,
            # table-tasks camera frames are 480x640 (H x W); resize_with_pad letterboxes
            # them into 224x224. This keeps train-time geometric augmentation of the
            # trace/keypoint targets aligned with the letterboxed image content.
            image_source_hw=(480, 640),
        ),
        data=LeRobotTraceVLAMoeDataConfig(
            repo_id="n5zhong/table_tasks",
            base_config=LiberoTraceDataConfig(
                repo_path=str(REPO_ROOT / "data/table-tasks"),
                prompt_from_task=True,
                skill_annotations_path=str(REPO_ROOT / "data/table-tasks/tabletask_skill_annotations.json"),
                trace_annotations_path=str(REPO_ROOT / "data/table-tasks/tabletask_skill_target_traces.json"),
                use_wrist_image=True,
                is_computing_norm_stats=False,
            ),
        ),
        assets_base_dir=str(REPO_ROOT / "assets"),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=50_000,
        save_interval=5_000,
        keep_period=5_000,
        log_interval=100,
        wandb_enabled=True,
    ),
    # ============================================================
    # TargetVLA-ActionMoe (Pi0TargetVLAActionMoe) - full FT and LoRA.
    # ============================================================
    # Trace-free ablation of the TraceVLA family. Architecture: 2-stream Gemma
    # trunk (paligemma + action MoE) — no trace stream, no trace generation,
    # no image overlay, no trace-related data augmentation. The semantic-target
    # point (the same conditioning input the trace family used) is injected
    # into the **action MoE's AdaRMS** via the same Fourier(2-D, 8 freqs) +
    # 2-layer MLP encoder used by the trace family. Loss = action FM + per-skill
    # completion regression (masked by has_skill). Skill plan + step number +
    # parameterized skill expression all flow into the VLM prompt the same way
    # as the trace_vla variants.
    TrainConfig(
        name="target_vla_actionmoe",
        model=__import__(
            "openpi.models.pi0_target_vla_actionmoe_config", fromlist=["Pi0TargetVLAActionMoeConfig"]
        ).Pi0TargetVLAActionMoeConfig(
            paligemma_variant="gemma_2b",
            action_expert_variant="trace_moe_gemma_300m",   # 5-expert MoE for actions
            action_horizon=10,
            pi05=True,
            discrete_state_input=False,
            max_token_len=200,
            num_action_experts=5,
        ),
        data=LeRobotTargetVLAActionMoeDataConfig(
            repo_id="yilin-wu/libero-100",
            base_config=LiberoTargetDataConfig(
                repo_path=str(REPO_ROOT / "data/libero-100"),
                prompt_from_task=True,
                skill_annotations_path=str(REPO_ROOT / "data/libero-100/skill_annotations.json"),
                trace_annotations_path=str(REPO_ROOT / "data/libero-100/skill_target_traces.json"),
                use_wrist_image=True,
                is_computing_norm_stats=False,
            ),
        ),
        assets_base_dir=str(REPO_ROOT / "assets"),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=200_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=100_000,
        save_interval=5_000,
        keep_period=10_000,
        log_interval=100,
        wandb_enabled=True,
    ),
    TrainConfig(
        name="target_vla_actionmoe_lora",
        # LoRA on paligemma 2B only. Action MoE + completion head are full FT.
        model=__import__(
            "openpi.models.pi0_target_vla_actionmoe_config", fromlist=["Pi0TargetVLAActionMoeConfig"]
        ).Pi0TargetVLAActionMoeConfig(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="trace_moe_gemma_300m",
            action_horizon=10,
            pi05=True,
            discrete_state_input=False,
            max_token_len=200,
            num_action_experts=5,
        ),
        data=LeRobotTargetVLAActionMoeDataConfig(
            repo_id="yilin-wu/libero-100",
            base_config=LiberoTargetDataConfig(
                repo_path=str(REPO_ROOT / "data/libero-100"),
                prompt_from_task=True,
                skill_annotations_path=str(REPO_ROOT / "data/libero-100/skill_annotations.json"),
                trace_annotations_path=str(REPO_ROOT / "data/libero-100/skill_target_traces.json"),
                use_wrist_image=True,
                is_computing_norm_stats=False,
            ),
        ),
        assets_base_dir=str(REPO_ROOT / "assets"),
        freeze_filter=__import__(
            "openpi.models.pi0_target_vla_actionmoe_config", fromlist=["Pi0TargetVLAActionMoeConfig"]
        ).Pi0TargetVLAActionMoeConfig(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="trace_moe_gemma_300m",
            action_horizon=10,
            pi05=True,
            discrete_state_input=False,
            max_token_len=200,
            num_action_experts=5,
        ).get_freeze_filter(),
        ema_decay=None,
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2e-4,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        save_interval=5_000,
        keep_period=10_000,
        log_interval=100,
        wandb_enabled=True,
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
