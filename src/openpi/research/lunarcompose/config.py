"""LunarCompose training configurations.

Follows the polaris_config.py pattern:
  get_lunarcompose_configs() → list[TrainConfig]
  Splat into _CONFIGS in src/openpi/training/config.py

Config namespace: lunarcompose_*

Taxonomy (15 configs total):
  12 per-cell: lunarcompose_{task}_{env}  (4 tasks x 3 envs)
   2 architecture: lunarcompose_factorized, lunarcompose_monolithic
   1 debug: lunarcompose_debug
"""

# Task and environment axes for the compositional grid.
_TASKS = ("payload", "latch", "clean", "connector")
_ENVS = ("nominal", "shadow", "contamination")

# Re-entrancy guard: when training/config.py builds _CONFIGS it calls
# get_lunarcompose_configs() which would import rm75_policy → training/config
# (partially loaded) → get_lunarcompose_configs() again. The guard returns []
# on the nested call; the outer call populates _cache which training/config
# then splats into _CONFIGS.
_building = False
_cache: list | None = None


def get_lunarcompose_configs():
    """Return LunarCompose training configs for registration in openpi's config system."""
    global _building, _cache  # noqa: PLW0603

    # Return cached result if available.
    if _cache is not None:
        return list(_cache)

    # Detect re-entrant call during _CONFIGS construction.
    if _building:
        return []

    _building = True
    try:
        # ALL imports deferred inside function to avoid circular imports.
        import openpi.models.pi0_config as pi0_config
        from openpi.research.shared.rm75_policy import LeRobotRM75DataConfig
        from openpi.training.config import DataConfig
        from openpi.training.config import FakeDataConfig
        from openpi.training.config import TrainConfig
        import openpi.training.weight_loaders as weight_loaders

        # Shared freeze filter for LoRA fine-tuning on pi0.5.
        _lora_freeze = pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter()

        # --- 12 per-cell configs: one per task x env combination ---
        configs = [
            TrainConfig(
                name=f"lunarcompose_{task}_{env}",
                model=pi0_config.Pi0Config(
                    pi05=True,
                    paligemma_variant="gemma_2b_lora",
                    action_expert_variant="gemma_300m_lora",
                ),
                data=LeRobotRM75DataConfig(
                    repo_id=f"placeholder/lunarcompose_{task}_{env}",
                    base_config=DataConfig(prompt_from_task=True),
                ),
                weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
                freeze_filter=_lora_freeze,
                ema_decay=None,
                num_train_steps=10_000,
                batch_size=32,
                wandb_enabled=True,
            )
            for task in _TASKS
            for env in _ENVS
        ]

        # --- 2 architecture-level configs ---
        configs.append(
            TrainConfig(
                name="lunarcompose_factorized",
                model=pi0_config.Pi0Config(
                    pi05=True,
                    paligemma_variant="gemma_2b_lora",
                    action_expert_variant="gemma_300m_lora",
                ),
                data=LeRobotRM75DataConfig(
                    repo_id="placeholder/lunarcompose_factorized",
                    base_config=DataConfig(prompt_from_task=True),
                ),
                weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
                freeze_filter=_lora_freeze,
                ema_decay=None,
                num_train_steps=20_000,
                batch_size=32,
                wandb_enabled=True,
            )
        )

        configs.append(
            TrainConfig(
                name="lunarcompose_monolithic",
                model=pi0_config.Pi0Config(
                    pi05=True,
                    paligemma_variant="gemma_2b_lora",
                    action_expert_variant="gemma_300m_lora",
                ),
                data=LeRobotRM75DataConfig(
                    repo_id="placeholder/lunarcompose_monolithic",
                    base_config=DataConfig(prompt_from_task=True),
                ),
                weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
                freeze_filter=_lora_freeze,
                ema_decay=None,
                num_train_steps=20_000,
                batch_size=32,
                wandb_enabled=True,
            )
        )

        # --- 1 debug config ---
        configs.append(
            TrainConfig(
                name="lunarcompose_debug",
                model=pi0_config.Pi0Config(
                    pi05=True,
                    paligemma_variant="dummy",
                    action_expert_variant="dummy",
                ),
                data=FakeDataConfig(),
                batch_size=2,
                num_train_steps=10,
                overwrite=True,
                exp_name="lunarcompose_debug",
                wandb_enabled=False,
            )
        )

        _cache = configs
        return list(_cache)
    finally:
        _building = False
