from jaxtyping import config
import pytest

from openpi.shared import array_typing as at


def test_disable_typechecking_restores_flag_on_exception():
    initial = config.jaxtyping_disable

    with pytest.raises(RuntimeError, match="boom"), at.disable_typechecking():
        raise RuntimeError("boom")

    assert config.jaxtyping_disable == initial
