"""Tests for LunarCompose training configurations."""

# Import training.config first to trigger _CONFIGS construction via the normal
# (non-re-entrant) path. This ensures lunarcompose configs are registered in the
# global config registry before we call get_lunarcompose_configs() directly.
from openpi.research.lunarcompose.config import _ENVS
from openpi.research.lunarcompose.config import _TASKS
from openpi.research.lunarcompose.config import get_lunarcompose_configs
import openpi.training.config  # noqa: F401


def test_config_count():
    """get_lunarcompose_configs() must return exactly 15 configs."""
    configs = get_lunarcompose_configs()
    assert len(configs) == 15, f"Expected 15 configs, got {len(configs)}"


def test_config_names_unique():
    """All 15 config names must be unique."""
    configs = get_lunarcompose_configs()
    names = [cfg.name for cfg in configs]
    assert len(names) == len(set(names)), f"Duplicate config names: {names}"


def test_config_namespace():
    """Every config name must start with 'lunarcompose_'."""
    configs = get_lunarcompose_configs()
    for cfg in configs:
        assert cfg.name.startswith("lunarcompose_"), f"Config name {cfg.name!r} must start with 'lunarcompose_'"


def test_debug_config_present():
    """A debug config with dummy variants must exist."""
    configs = get_lunarcompose_configs()
    debug = [c for c in configs if c.name == "lunarcompose_debug"]
    assert len(debug) == 1, "Expected exactly one lunarcompose_debug config"
    cfg = debug[0]
    assert cfg.model.paligemma_variant == "dummy"
    assert cfg.model.action_expert_variant == "dummy"
    assert cfg.batch_size == 2
    assert cfg.num_train_steps == 10
    assert cfg.wandb_enabled is False
    assert cfg.overwrite is True


def test_per_cell_configs_complete():
    """All 12 per-cell configs (4 tasks x 3 envs) must exist."""
    configs = get_lunarcompose_configs()
    names = {cfg.name for cfg in configs}
    for task in _TASKS:
        for env in _ENVS:
            expected = f"lunarcompose_{task}_{env}"
            assert expected in names, f"Missing per-cell config: {expected}"


def test_architecture_configs_present():
    """lunarcompose_factorized and lunarcompose_monolithic must exist."""
    configs = get_lunarcompose_configs()
    names = {cfg.name for cfg in configs}
    assert "lunarcompose_factorized" in names, "Missing lunarcompose_factorized"
    assert "lunarcompose_monolithic" in names, "Missing lunarcompose_monolithic"


def test_per_cell_configs_use_lora():
    """Per-cell configs must use LoRA variants and standard hyperparams."""
    configs = get_lunarcompose_configs()
    cell_configs = [
        c
        for c in configs
        if c.name not in {"lunarcompose_debug", "lunarcompose_factorized", "lunarcompose_monolithic"}
    ]
    assert len(cell_configs) == 12
    for cfg in cell_configs:
        assert cfg.model.paligemma_variant == "gemma_2b_lora"
        assert cfg.model.action_expert_variant == "gemma_300m_lora"
        assert cfg.model.pi05 is True
        assert cfg.ema_decay is None
        assert cfg.num_train_steps == 10_000
        assert cfg.batch_size == 32


def test_architecture_configs_hyperparams():
    """Architecture configs must use 20k steps."""
    configs = get_lunarcompose_configs()
    arch = {c.name: c for c in configs if c.name in ("lunarcompose_factorized", "lunarcompose_monolithic")}
    assert len(arch) == 2
    for name, cfg in arch.items():
        assert cfg.num_train_steps == 20_000, f"{name} should have 20k steps"
        assert cfg.model.paligemma_variant == "gemma_2b_lora"
        assert cfg.model.action_expert_variant == "gemma_300m_lora"
        assert cfg.batch_size == 32


def test_get_config_lunarcompose_debug():
    """lunarcompose_debug must be discoverable via the global config registry."""
    from openpi.training.config import get_config

    cfg = get_config("lunarcompose_debug")
    assert cfg.name == "lunarcompose_debug"
    assert cfg.model.paligemma_variant == "dummy"
