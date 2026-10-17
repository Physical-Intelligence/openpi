import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.weight_loaders as weight_loaders


def test_paligemma_configs_load_paligemma_weights():
    """All RoboArena baseline configs are documented as "Trained from PaliGemma".

    Regression test for https://github.com/Physical-Intelligence/openpi/issues/856:
    they must not silently default to NoOpWeightLoader (random initialization).
    """
    configs = roboarena_config.get_roboarena_configs()
    assert len(configs) == 5
    for config in configs:
        assert isinstance(
            config.weight_loader, weight_loaders.PaliGemmaWeightLoader
        ), f"{config.name} does not load PaliGemma weights (got {type(config.weight_loader).__name__})"
