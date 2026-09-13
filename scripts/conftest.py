import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    # Must run before any test module in this directory is collected/imported, since some of them
    # (e.g. serve_policy_test.py) transitively import jax through openpi.training.config and jax reads
    # this env var at first initialization. A pytest fixture would run too late for that.
    os.environ["JAX_PLATFORMS"] = "cpu"
