"""Transform registry for managing per-embodiment transform pipelines."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable

import openpi.transforms as _transforms

# Type alias for a factory that produces a transform Group.
TransformFactory = Callable[[], _transforms.Group]


class TransformRegistry:
    """Registry for embodiment-specific transform factories.

    Instead of a massive if-else block in config.py, new robots register their
    transform pipelines here. This makes it easy to add new robots without
    modifying existing code.

    Usage:
        registry = TransformRegistry()
        registry.register("rby1_gripper", data_transforms=lambda: Group(inputs=[...]))
        transforms = registry.get_data_transforms("rby1_gripper")
    """

    def __init__(self) -> None:
        self._data_transforms: dict[str, TransformFactory] = {}
        self._repack_transforms: dict[str, TransformFactory] = {}

    def register(
        self,
        name: str,
        *,
        data_transforms: TransformFactory | None = None,
        repack_transforms: TransformFactory | None = None,
    ) -> None:
        """Register transform factories for an embodiment.

        Args:
            name: Unique embodiment name.
            data_transforms: Factory that returns the data transforms Group.
            repack_transforms: Factory that returns the repack transforms Group.
        """
        if data_transforms is not None:
            self._data_transforms[name] = data_transforms
        if repack_transforms is not None:
            self._repack_transforms[name] = repack_transforms

    def get_data_transforms(self, name: str) -> _transforms.Group:
        """Get data transforms for the given embodiment name."""
        factory = self._data_transforms.get(name)
        if factory is None:
            logging.warning(f"No data transforms registered for embodiment '{name}', using empty Group.")
            return _transforms.Group()
        return factory()

    def get_repack_transforms(self, name: str) -> _transforms.Group:
        """Get repack transforms for the given embodiment name."""
        factory = self._repack_transforms.get(name)
        if factory is None:
            logging.warning(f"No repack transforms registered for embodiment '{name}', using empty Group.")
            return _transforms.Group()
        return factory()

    def registered_names(self) -> list[str]:
        """Return all registered embodiment names."""
        return sorted(set(self._data_transforms) | set(self._repack_transforms))


# Global registry instance.
_GLOBAL_REGISTRY = TransformRegistry()


def get_global_registry() -> TransformRegistry:
    """Return the global transform registry."""
    return _GLOBAL_REGISTRY


def register_embodiment(
    name: str,
    *,
    data_transforms: TransformFactory | None = None,
    repack_transforms: TransformFactory | None = None,
) -> None:
    """Register transform factories for an embodiment in the global registry."""
    _GLOBAL_REGISTRY.register(
        name,
        data_transforms=data_transforms,
        repack_transforms=repack_transforms,
    )
