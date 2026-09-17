"""Tests for ParamForm — a form auto-generated from a step's params schema.

This is the payoff of the declarative schema: one declaration -> validation,
serialization, and (here) an editing widget, with no per-step Qt code.
"""
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtWidgets import QDoubleSpinBox  # noqa: E402

from cortica.core.params import Bool, Choice, Float  # noqa: E402
from cortica.gui.param_form import ParamForm  # noqa: E402


def test_values_reflect_defaults(qtbot):
    form = ParamForm([Float("l_freq", 1.0), Float("h_freq", 40.0)], {})
    qtbot.addWidget(form)
    assert form.values() == {"l_freq": 1.0, "h_freq": 40.0}


def test_provided_values_override_defaults(qtbot):
    form = ParamForm([Float("h_freq", 40.0)], {"h_freq": 30.0})
    qtbot.addWidget(form)
    assert form.values()["h_freq"] == 30.0


def test_editing_a_field_emits_changed_with_new_values(qtbot):
    form = ParamForm([Float("sfreq", 250.0)], {})
    qtbot.addWidget(form)
    spin = form.findChild(QDoubleSpinBox)
    with qtbot.waitSignal(form.changed) as blocker:
        spin.setValue(128.0)
    assert blocker.args[0]["sfreq"] == 128.0


def test_choice_and_bool_render_and_read_back(qtbot):
    form = ParamForm(
        [Choice("method", "fir", options=["fir", "iir"]), Bool("verbose", True)], {}
    )
    qtbot.addWidget(form)
    values = form.values()
    assert values["method"] == "fir"
    assert values["verbose"] is True
