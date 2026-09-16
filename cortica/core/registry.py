"""A registry of available steps.

The GUI's step library and the YAML pipeline loader both resolve steps through a
registry: it maps a stable step ``id`` to its class and can filter by modality.
"""
from __future__ import annotations


class Registry:
    def __init__(self):
        self._by_id: dict = {}

    def register(self, step_cls):
        """Register a Step subclass. Returns it, so it also works as a decorator."""
        step_id = getattr(step_cls, "id", "")
        if not step_id:
            raise ValueError(
                f"{step_cls.__name__} has no `id`; set a stable class-level id"
            )
        if step_id in self._by_id:
            raise ValueError(f"duplicate step id: {step_id!r}")
        self._by_id[step_id] = step_cls
        return step_cls

    def get(self, step_id: str):
        if step_id not in self._by_id:
            raise KeyError(f"no step registered with id {step_id!r}")
        return self._by_id[step_id]

    def all(self) -> list:
        return list(self._by_id.values())

    def for_modality(self, modality: str) -> list:
        return [c for c in self._by_id.values() if modality in c.modalities]


#: Shared registry that built-in and plugin steps register into.
default_registry = Registry()


def register(step_cls):
    """Decorator: register a Step in the :data:`default_registry`."""
    return default_registry.register(step_cls)
