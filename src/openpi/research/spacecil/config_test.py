"""Tests for SpaceCIL training configurations."""

# Import training.config first to trigger _CONFIGS construction via the normal
# (non-re-entrant) path. This ensures spacecil configs are registered in the
# global config registry before we call get_spacecil_configs() directly.
from openpi.research.spacecil.config import get_spacecil_configs
import openpi.training.config  # noqa: F401


def test_returns_nonempty_list():
    configs = get_spacecil_configs()
    assert len(configs) > 0


def test_all_names_start_with_spacecil():
    configs = get_spacecil_configs()
    for cfg in configs:
        assert cfg.name.startswith("spacecil_"), f"Config name {cfg.name!r} must start with 'spacecil_'"


def test_config_names_are_unique():
    configs = get_spacecil_configs()
    names = [cfg.name for cfg in configs]
    assert len(names) == len(set(names)), f"Duplicate config names: {names}"


def test_expected_config_names():
    """Test that all expected base and baseline variant configs are present."""
    configs = get_spacecil_configs()
    names = {cfg.name for cfg in configs}
    # Base task configs
    base_names = {
        "spacecil_rm75_payload",
        "spacecil_rm75_latch",
        "spacecil_rm75_clean",
        "spacecil_rm75_connector",
        "spacecil_debug",
    }
    # Baseline variants (5 per task: fulltune, nodistill, oracle, random, plus 1 shared_lora)
    tasks = ["payload", "latch", "clean", "connector"]
    variant_suffixes = ["_fulltune", "_nodistill", "_oracle", "_random"]
    variant_names = {
        f"spacecil_rm75_{task}{suffix}"
        for task in tasks
        for suffix in variant_suffixes
    }
    variant_names.add("spacecil_rm75_shared_lora")  # Single shared LoRA variant
    expected = base_names | variant_names
    assert names == expected, f"Expected {expected}, got {names}"


def test_debug_config_uses_dummy_variants():
    configs = get_spacecil_configs()
    debug = [c for c in configs if c.name == "spacecil_debug"]
    assert len(debug) == 1
    cfg = debug[0]
    assert cfg.model.paligemma_variant == "dummy"
    assert cfg.model.action_expert_variant == "dummy"
    assert cfg.batch_size == 2
    assert cfg.num_train_steps == 10
    assert cfg.wandb_enabled is False
    assert cfg.overwrite is True


def test_task_configs_use_lora():
    configs = get_spacecil_configs()
    task_configs = [c for c in configs if c.name != "spacecil_debug"]
    for cfg in task_configs:
        assert cfg.model.paligemma_variant == "gemma_2b_lora"
        assert cfg.model.action_expert_variant == "gemma_300m_lora"
        assert cfg.model.pi05 is True
        assert cfg.ema_decay is None
        assert cfg.num_train_steps == 10_000
        assert cfg.batch_size == 32


def test_get_config_spacecil_debug():
    """Test that spacecil_debug is discoverable via the global config registry."""
    from openpi.training.config import get_config

    cfg = get_config("spacecil_debug")
    assert cfg.name == "spacecil_debug"
    assert cfg.model.paligemma_variant == "dummy"


def test_baseline_configs_resolve():
    """Test that baseline variant configs resolve via the global config registry."""
    from openpi.training.config import get_config

    # Test baseline config names resolve
    baseline_names = [
        "spacecil_rm75_payload_fulltune",
        "spacecil_rm75_payload_nodistill",
        "spacecil_rm75_shared_lora",
        "spacecil_rm75_payload_oracle",
        "spacecil_rm75_payload_random",
    ]

    for name in baseline_names:
        cfg = get_config(name)
        assert cfg.name == name, f"Config name mismatch: expected {name}, got {cfg.name}"

    # Verify fulltune variant has freeze_filter=None
    fulltune_cfg = get_config("spacecil_rm75_payload_fulltune")
    assert fulltune_cfg.freeze_filter is None, f"fulltune config should have freeze_filter=None, got {fulltune_cfg.freeze_filter}"

