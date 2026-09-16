"""Smoke tests for the main window (run offscreen via conftest).

These check the wiring — the library reflects registered steps, adding a step
updates the pipeline panel, and the threaded Run produces a result — not pixels.
"""
import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")
mne = pytest.importorskip("mne")

from PySide6.QtCore import Qt  # noqa: E402

import cortica.steps  # noqa: E402,F401  register built-in steps
from cortica.gui.main_window import MainWindow  # noqa: E402
from cortica.io import dataset_from_raw  # noqa: E402


def _raw():
    info = mne.create_info(["EEG 001", "EEG 002"], 200.0, ch_types="eeg")
    return mne.io.RawArray(np.random.RandomState(0).randn(2, 800) * 1e-6, info, verbose=False)


def _find_item(listw, step_id):
    for i in range(listw.count()):
        if listw.item(i).data(Qt.ItemDataRole.UserRole) == step_id:
            return listw.item(i)
    raise AssertionError(f"{step_id} not found in library")


def test_window_lists_registered_steps_after_load(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w.state.set_source(dataset_from_raw(_raw()))
    labels = [w.library.item(i).text() for i in range(w.library.count())]
    assert any("Band-pass" in t for t in labels)
    assert any("Resample" in t for t in labels)


def test_adding_a_step_updates_the_pipeline_panel(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w.state.set_source(dataset_from_raw(_raw()))
    w._add_from_item(_find_item(w.library, "bandpass_filter"))
    assert len(w.state.pipeline.steps) == 1
    assert w.pipeline_list.count() == 1


def test_run_button_processes_and_updates_result(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w.state.set_source(dataset_from_raw(_raw()))
    w.state.add_step("resample", {"sfreq": 100.0})
    with qtbot.waitSignal(w.state.resultChanged, timeout=5000):
        w._run()
    assert w.state.result is not None
    assert w.state.result.payload.info["sfreq"] == 100.0


def test_cli_no_command_launches_gui(monkeypatch):
    import cortica.gui.app as app_mod
    from cortica.cli import main

    called = {}

    def fake_run_app(*args, **kwargs):
        called["ran"] = True
        return 0

    monkeypatch.setattr(app_mod, "run_app", fake_run_app)
    assert main([]) == 0
    assert called["ran"] is True
