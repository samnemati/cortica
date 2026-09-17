"""A form widget auto-generated from a step's parameter schema.

Given a list of :class:`~cortica.core.params.Param` and current values, it builds
one editor per parameter and emits :attr:`changed` (with the full values dict)
whenever the user edits a field. No per-step Qt code is needed.
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLineEdit,
    QSpinBox,
    QWidget,
)

from ..core.params import Bool, Choice, Float, Int

_INT_MIN, _INT_MAX = -(2**31), 2**31 - 1


class ParamForm(QWidget):
    changed = Signal(dict)

    def __init__(self, params, values, parent=None):
        super().__init__(parent)
        self._fields = []  # list of (name, getter)
        layout = QFormLayout(self)
        for param in params:
            value = values.get(param.name, param.default)
            widget, getter = self._build(param, value)
            self._fields.append((param.name, getter))
            layout.addRow(param.label, widget)

    def _build(self, param, value):
        if isinstance(param, Float):
            widget = QDoubleSpinBox()
            widget.setDecimals(3)
            widget.setRange(
                param.min if param.min is not None else -1e9,
                param.max if param.max is not None else 1e9,
            )
            if param.unit:
                widget.setSuffix(f" {param.unit}")
            widget.setValue(float(value))
            widget.valueChanged.connect(self._emit)
            return widget, widget.value
        if isinstance(param, Int):
            widget = QSpinBox()
            widget.setRange(
                int(param.min) if param.min is not None else _INT_MIN,
                int(param.max) if param.max is not None else _INT_MAX,
            )
            widget.setValue(int(value))
            widget.valueChanged.connect(self._emit)
            return widget, widget.value
        if isinstance(param, Bool):
            widget = QCheckBox()
            widget.setChecked(bool(value))
            widget.stateChanged.connect(self._emit)
            return widget, widget.isChecked
        if isinstance(param, Choice):
            widget = QComboBox()
            widget.addItems([str(option) for option in param.options])
            widget.setCurrentText(str(value))
            widget.currentTextChanged.connect(self._emit)
            return widget, widget.currentText
        # Str and any other param type
        widget = QLineEdit()
        widget.setText(str(value))
        widget.textChanged.connect(self._emit)
        return widget, widget.text

    def values(self) -> dict:
        return {name: getter() for name, getter in self._fields}

    def _emit(self, *_args) -> None:
        self.changed.emit(self.values())
