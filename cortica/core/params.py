"""Declarative parameter schema.

A step declares its parameters as data (a list of :class:`Param`). That single
declaration drives three things:

* **validation** — :meth:`Param.validate` coerces and range-checks a value;
* **serialization** — validated values are stored in the pipeline YAML;
* **GUI forms** — :meth:`Param.to_dict` gives the GUI enough to build a widget,
  so no per-step Qt code is needed.
"""
from __future__ import annotations

from .errors import ParamError

__all__ = ["Param", "Float", "Int", "Bool", "Choice", "Str", "ParamError"]

_BOOL_TRUE = {"true", "1", "yes", "on"}
_BOOL_FALSE = {"false", "0", "no", "off"}


class Param:
    """Base parameter. Subclasses set ``type`` and implement :meth:`validate`."""

    type = "param"

    def __init__(self, name: str, default, *, label: str | None = None):
        self.name = name
        self.default = default
        self.label = label or name

    def validate(self, value):
        raise NotImplementedError

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.type,
            "default": self.default,
            "label": self.label,
        }


class Float(Param):
    type = "float"

    def __init__(self, name, default, *, min=None, max=None, unit=None, label=None):
        super().__init__(name, default, label=label)
        self.min = min
        self.max = max
        self.unit = unit

    def validate(self, value) -> float:
        # bool is a subclass of int; reject it explicitly for numeric params.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ParamError(f"{self.name}: expected a number, got {value!r}")
        v = float(value)
        if self.min is not None and v < self.min:
            raise ParamError(f"{self.name}: {v} is below minimum {self.min}")
        if self.max is not None and v > self.max:
            raise ParamError(f"{self.name}: {v} is above maximum {self.max}")
        return v

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update(min=self.min, max=self.max, unit=self.unit)
        return d


class Int(Param):
    type = "int"

    def __init__(self, name, default, *, min=None, max=None, label=None):
        super().__init__(name, default, label=label)
        self.min = min
        self.max = max

    def validate(self, value) -> int:
        if isinstance(value, bool):
            raise ParamError(f"{self.name}: expected an integer, got {value!r}")
        if isinstance(value, float):
            if not value.is_integer():
                raise ParamError(f"{self.name}: expected an integer, got {value!r}")
            value = int(value)
        if not isinstance(value, int):
            raise ParamError(f"{self.name}: expected an integer, got {value!r}")
        if self.min is not None and value < self.min:
            raise ParamError(f"{self.name}: {value} is below minimum {self.min}")
        if self.max is not None and value > self.max:
            raise ParamError(f"{self.name}: {value} is above maximum {self.max}")
        return int(value)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update(min=self.min, max=self.max)
        return d


class Bool(Param):
    type = "bool"

    def validate(self, value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            s = value.strip().lower()
            if s in _BOOL_TRUE:
                return True
            if s in _BOOL_FALSE:
                return False
        if value in (0, 1):
            return bool(value)
        raise ParamError(f"{self.name}: expected a boolean, got {value!r}")


class Choice(Param):
    type = "choice"

    def __init__(self, name, default, *, options, label=None):
        super().__init__(name, default, label=label)
        self.options = list(options)

    def validate(self, value):
        if value not in self.options:
            raise ParamError(f"{self.name}: {value!r} is not one of {self.options}")
        return value

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["options"] = self.options
        return d


class Str(Param):
    type = "str"

    def validate(self, value) -> str:
        if not isinstance(value, str):
            raise ParamError(f"{self.name}: expected a string, got {value!r}")
        return value
