import dataclasses

import jaxtyping
import numpy as np
import pytest

from openpi.shared import array_typing as at


@at.typecheck
@dataclasses.dataclass(frozen=True)
class _Sample:
    image: dict[str, at.Float[np.ndarray, "h w c"]]
    state: at.Float[np.ndarray, "..."]


def _valid() -> dict:
    return {"image": {"cam": np.zeros((4, 4, 3), np.float32)}, "state": np.zeros(6, np.float32)}


def _invalid() -> dict:
    return {"image": {"cam": np.zeros((4, 4), np.float32)}, "state": np.zeros(6, np.float32)}


def test_dataclass_checker_is_built_once(monkeypatch):
    # Regression: jaxtyping rebuilt and re-decorated the checker on every instantiation, ~1.4 ms per construct.
    monkeypatch.setattr(at, "_dataclass_checkers", {})
    builds = []
    original = at._build_dataclass_checker  # noqa: SLF001
    monkeypatch.setattr(
        at, "_build_dataclass_checker", lambda cls, typechecker: builds.append(cls) or original(cls, typechecker)
    )

    for _ in range(5):
        _Sample(**_valid())

    assert builds == [_Sample]


def test_dataclass_annotations_are_still_checked():
    _Sample(**_valid())

    with pytest.raises(jaxtyping.TypeCheckError):
        _Sample(**_invalid())


def test_cached_checker_still_honours_disable_typechecking():
    _Sample(**_valid())  # warm the cache

    with at.disable_typechecking():
        _Sample(**_invalid())  # would raise if the cached checker bypassed the global flag

    with pytest.raises(jaxtyping.TypeCheckError):
        _Sample(**_invalid())
