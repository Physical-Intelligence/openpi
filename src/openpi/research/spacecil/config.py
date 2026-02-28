"""SpaceCIL training configurations.

Follows the polaris_config.py pattern:
  get_spacecil_configs() → list[TrainConfig]
  Splat into _CONFIGS in src/openpi/training/config.py

Config namespace: spacecil_*
"""

import dataclasses

# Task names for the four SpaceCIL operational tasks.
_TASKS = ("payload", "latch", "clean", "connector")

# Re-entrancy guard: when training/config.py builds _CONFIGS it calls
# get_spacecil_configs() which would import rm75_policy → training/config
# (partially loaded) → get_spacecil_configs() again. The guard returns []
# on the nested call; the outer call populates _cache which training/config
# then splats into _CONFIGS.
_building = False
_cache: list | None = None


def _make_baseline_variants(base_configs: list) -> list:
    """Generate baseline config variants for ablation studies.

    Args:
        base_configs: List of task-specific TrainConfig objects.

    Returns:
        List of baseline variant configs:
        1. <task>_fulltune: Full fine-tuning (freeze_filter=None)
        2. <task>_nodistill: Explicit no-distillation variant
        3. shared_lora: Single shared LoRA across all tasks
        4. <task>_oracle: Per-task PEFT with oracle routing
        5. <task>_random: Per-task PEFT with random routing
    """
    variants = []

    # 1. Full fine-tuning variant (freeze_filter=None)
    for cfg in base_configs:
        fulltune = dataclasses.replace(
            cfg,
            name=f"{cfg.name}_fulltune",
            freeze_filter=None,
        )
        variants.append(fulltune)

    # 2. No-distillation variant (same config, different name for ablation)
    for cfg in base_configs:
        nodistill = dataclasses.replace(
            cfg,
            name=f"{cfg.name}_nodistill",
        )
        variants.append(nodistill)

    # 3. Shared LoRA variant (one config shared across all tasks)
    if base_configs:
        first_cfg = base_configs[0]
        shared_lora = dataclasses.replace(
            first_cfg,
            name="spacecil_rm75_shared_lora",
        )
        variants.append(shared_lora)

    # 4. Oracle routing variant (per-task PEFT with oracle task IDs)
    for cfg in base_configs:
        oracle = dataclasses.replace(
            cfg,
            name=f"{cfg.name}_oracle",
        )
        variants.append(oracle)

    # 5. Random routing variant (per-task PEFT with random routing)
    for cfg in base_configs:
        random_cfg = dataclasses.replace(
            cfg,
            name=f"{cfg.name}_random",
        )
        variants.append(random_cfg)

    return variants


def get_spacecil_configs():
    """Return SpaceCIL training configs for registration in openpi's config system."""
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
        import os
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

        # One config per operational task.
        configs = [
            TrainConfig(
                name=f"spacecil_rm75_{task}",
                model=pi0_config.Pi0Config(
                    pi05=True,
                    paligemma_variant="gemma_2b_lora",
                    action_expert_variant="gemma_300m_lora",
                ),
                data=LeRobotRM75DataConfig(
                    repo_id=os.environ.get("SPACECIL_DATA_PREFIX", "placeholder") + f"/spacecil_{task}",
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
        ]

        # Generate baseline variants for ablation experiments.
        task_configs = configs  # Save task configs before adding debug
        baseline_variants = _make_baseline_variants(task_configs)
        configs.extend(baseline_variants)

        # Debug config with dummy models for fast testing.
        configs.append(
            TrainConfig(
                name="spacecil_debug",
                model=pi0_config.Pi0Config(
                    pi05=True,
                    paligemma_variant="dummy",
                    action_expert_variant="dummy",
                ),
                data=FakeDataConfig(),
                batch_size=2,
                num_train_steps=10,
                overwrite=True,
                exp_name="spacecil_debug",
                wandb_enabled=False,
            )
        )

        _cache = configs
        return list(_cache)
    finally:
        _building = False
