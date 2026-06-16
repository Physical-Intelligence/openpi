# config.py: Builds TrainConfig for single right-arm pi0.5 fine-tuning experiments.
# Self-contained — does not modify the openpi codebase.

import sys
from pathlib import Path

import yaml
from flax import nnx

import openpi.models.pi0_config as pi0_config
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

# Import local transforms from experiments/
sys.path.insert(0, str(Path(__file__).parent))
from transforms import AlohaSingleArmInputs, AlohaSingleArmOutputs

# ---------------------------------------------------------------------------
# freeze filter
# ---------------------------------------------------------------------------
def vlm_only_freeze_filter() -> nnx.filterlib.Filter:
    """training: train ONLY the PaliGemma-LLM LoRA adapters.

    Frozen:
      - PaliGemma base weights (non-LoRA, non-action-expert under .llm.)
      - The ENTIRE action expert (every param with `_1` suffix — both base
        and any LoRA), so the flow-matching action quality is anchored
        against the loc-CE term and can't drift
      - SigLIP image encoder (.img.)
      - Action heads (action_in_proj, action_out_proj)
      - Time MLPs (time_mlp_in/out under pi05, action_time_mlp under pi0)
      - state_proj (only present under pi0)

    Trainable:
      - LoRA adapters on the PaliGemma backbone (matched by `.lora.` AND under .llm. AND NOT under `_1`)

    Returns an nnx filter that resolves to the union (Any) of all "frozen"
    sets — the trainer takes its complement to derive what to update.
    """
    return nnx.Any(
        # PaliGemma base: under .llm. but NOT lora and NOT action-expert (_1)
        nnx.All(
            nnx_utils.PathRegex(".*llm.*"),
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            nnx.Not(nnx_utils.PathRegex(".*_1.*")),
        ),
        # Entire action expert (including any LoRA on it)
        nnx_utils.PathRegex(".*_1.*"),
        # SigLIP
        nnx_utils.PathRegex(".*img.*"),
        # Action / time projections (outside .llm.)
        nnx_utils.PathRegex(".*action_in_proj.*"),
        nnx_utils.PathRegex(".*action_out_proj.*"),
        nnx_utils.PathRegex(".*action_time_mlp.*"),
        nnx_utils.PathRegex(".*time_mlp.*"),
        nnx_utils.PathRegex(".*state_proj.*"),
    )


# ---------------------------------------------------------------------------
# data + model transform factories
# ---------------------------------------------------------------------------
def _loc_ce_data_transforms_factory(bbox_sidecar_parquet: str, default_query: str):
    """Returns a callable `(model_config) -> Group` for the data_transforms slot.

    Inserts LocTargetsBuilder BEFORE the embodiment transform (so it has
    episode_index/frame_index to join on) and PreserveAuxFields-wraps the
    embodiment transform so the new aux fields survive its fresh-dict return.
    """
    sidecar_path = Path(bbox_sidecar_parquet).expanduser().resolve()
    if not sidecar_path.exists():
        raise FileNotFoundError(f"bbox_sidecar_parquet not found: {sidecar_path}")

    def factory(_model_config):
        from transforms_loc import LocTargetsBuilder, PreserveAuxFields

        return _transforms.Group(
            inputs=[
                LocTargetsBuilder(sidecar_parquet=sidecar_path, default_query=default_query),
                PreserveAuxFields(wrapped=AlohaSingleArmInputs()),
            ],
            outputs=[AlohaSingleArmOutputs()],
        )

    return factory


def _loc_ce_model_transforms_factory(default_prompt: str):
    """Returns a callable `(model_config) -> Group` for the model_transforms slot.

    Mirrors openpi's ModelTransformFactory PI05 case but REPLACES the stock
    TokenizePrompt with DualPromptTokenize, which emits both the π0.5
    template (for MSE) and the PaliGemma-native detect prompt (for CE).
    """

    def factory(model_config):
        import sentencepiece

        import openpi.shared.download as download
        from openpi.models.tokenizer import PaligemmaTokenizer

        from transforms_loc import DualPromptTokenize

        spm_path = download.maybe_download(
            "gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"}
        )
        spm = sentencepiece.SentencePieceProcessor(model_proto=spm_path.open("rb").read())
        pi05_tokenizer = PaligemmaTokenizer(model_config.max_token_len)

        return _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(default_prompt),
                _transforms.ResizeImages(224, 224),
                DualPromptTokenize(
                    pi05_tokenizer=pi05_tokenizer,
                    spm=spm,
                    detect_max_len=32,
                ),
                _transforms.PadStatesAndActions(model_config.action_dim),
            ],
        )

    return factory


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
    log_interval: int = 100,
    resume: bool = False,
    wandb_enabled: bool = True,
    checkpoint_base_dir: str = "./checkpoints",
    pytorch_weight_path: str | None = None,
    use_lora: bool = True,
    # ---------- extensions (default off for backward-compat) ----------
    # When `model_variant == "pi05_loc_ce"`, swap in Pi0WithLocCEConfig,
    # vlm_only_freeze_filter, LocTargetsBuilder + DualPromptTokenize transforms.
    model_variant: str = "pi05",
    freeze_mode: str = "default",        # "default" | "vlm_only"
    loc_loss_weight: float = 1.0,        # α — only used when model_variant == "pi05_loc_ce"
    mse_loss_weight: float = 10.0,       # β
    bbox_sidecar_parquet: str | None = None,
    bbox_query: str = "box",
) -> _config.TrainConfig:
    """Build a TrainConfig for single right-arm pi0.5 LoRA fine-tuning.

    path: set `model_variant="pi05_loc_ce"`, `freeze_mode="vlm_only"`,
    and provide `bbox_sidecar_parquet`. Otherwise behaves identically to the
    previous version.
    """
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
                    # needs these surviving the repack so LocTargetsBuilder
                    # can do its (episode_index, frame_index) lookup. RepackTransform
                    # accepts identity mappings — these are no-ops on standard data
                    # but ensure the keys propagate.
                    "episode_index": "episode_index",
                    "frame_index": "frame_index",
                }
            )
        ]
    )

    # -------- Model config + freeze filter dispatch --------
    is_loc_ce = (model_variant == "pi05_loc_ce")

    if is_loc_ce:
        from model_pi05_loc import Pi0WithLocCEConfig

        if bbox_sidecar_parquet is None:
            raise ValueError(
                "model_variant='pi05_loc_ce' requires bbox_sidecar_parquet (the merged "
                "sidecar produced by experiments/data/merge_bbox_sidecars.py)."
            )
        model_config = Pi0WithLocCEConfig(
            pi05=True,
            paligemma_variant="gemma_2b_lora" if use_lora else "gemma_2b",
            action_expert_variant="gemma_300m",        # NO LoRA on action expert — frozen entirely
            loc_loss_weight=loc_loss_weight,
            mse_loss_weight=mse_loss_weight,
        )
    else:
        if use_lora:
            model_config = pi0_config.Pi0Config(
                pi05=True,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            )
        else:
            model_config = pi0_config.Pi0Config(pi05=True)

    # Freeze filter
    if freeze_mode == "vlm_only":
        freeze_filter = vlm_only_freeze_filter()
        ema_decay = None
    elif use_lora:
        freeze_filter = model_config.get_freeze_filter()
        ema_decay = None
    else:
        freeze_filter = nnx.Nothing
        ema_decay = 0.99

    # -------- Transform group factories --------
    if is_loc_ce:
        data_transforms_fn = _loc_ce_data_transforms_factory(
            bbox_sidecar_parquet=bbox_sidecar_parquet,
            default_query=bbox_query,
        )
        model_transforms_fn = _loc_ce_model_transforms_factory(default_prompt=default_prompt)
    else:
        data_transforms_fn = lambda _: _transforms.Group(
            inputs=[AlohaSingleArmInputs()],
            outputs=[AlohaSingleArmOutputs()],
        )
        model_transforms_fn = _config.ModelTransformFactory(default_prompt=default_prompt)

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
            data_transforms=data_transforms_fn,
            model_transforms=model_transforms_fn,
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
        log_interval=log_interval,
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
        log_interval=training.get("log_interval", 100),
        resume=training.get("resume", False),
        wandb_enabled=training.get("wandb_enabled", True),
        checkpoint_base_dir=training.get("checkpoint_base_dir", "./checkpoints"),
        pytorch_weight_path=model.get("pytorch_weight_path"),
        use_lora=training.get("use_lora", True),
        # extensions (all default to backward-compat values when absent)
        model_variant=training.get("model_variant", "pi05"),
        freeze_mode=training.get("freeze_mode", "default"),
        loc_loss_weight=training.get("loc_loss_weight", 1.0),
        mse_loss_weight=training.get("mse_loss_weight", 10.0),
        bbox_sidecar_parquet=data.get("bbox_sidecar_parquet"),
        bbox_query=data.get("bbox_query", "box"),
    )
