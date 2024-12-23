from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


def test_make_bool_mask():
    assert _policy_config.make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
    assert _policy_config.make_bool_mask(2, 0, 2) == (True, True, True, True)


def test_create_trained_policy():
    policy = _policy_config.create_trained_policy(
        _config.get_config("debug"),
        "s3://openpi-assets/checkpoints/pi0_base",
        # The base checkpoint doesn't have norm stats.
        norm_stats={},
    )
    assert policy is not None
