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
from PySide6.QtWidgets import QDoubleSpinBox  # noqa: E402

import cortica.steps  # noqa: E402,F401  register built-in steps
from cortica.gui.main_window import MainWindow  # noqa: E402
from cortica.gui.param_form import ParamForm  # noqa: E402
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


def test_load_eeg_sample_populates_the_window(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w._load_eeg_sample()
    assert w.state.modality == "eeg"
    assert w.library.count() > 0


def test_load_fnirs_sample_switches_modality_to_fnirs(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w._load_fnirs_sample()
    assert w.state.modality == "fnirs"


def test_selecting_a_step_shows_its_param_form(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w._load_eeg_sample()
    w.state.add_step("bandpass_filter")
    w.pipeline_list.setCurrentRow(0)
    form = w.findChild(ParamForm)
    assert form is not None
    assert "l_freq" in form.values()


def test_editing_a_param_updates_the_pipeline_step(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w._load_eeg_sample()
    w.state.add_step("bandpass_filter")
    w.pipeline_list.setCurrentRow(0)
    form = w.findChild(ParamForm)
    form.findChildren(QDoubleSpinBox)[0].setValue(2.0)  # l_freq is the first field
    assert w.state.pipeline.steps[0].params["l_freq"] == 2.0


def test_remove_selected_step(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w._load_eeg_sample()
    w.state.add_step("bandpass_filter")
    w.state.add_step("resample")
    w.pipeline_list.setCurrentRow(0)
    w._remove_selected()
    assert [s.step_id for s in w.state.pipeline.steps] == ["resample"]


def test_move_selected_step_down(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w._load_eeg_sample()
    w.state.add_step("bandpass_filter")
    w.state.add_step("resample")
    w.pipeline_list.setCurrentRow(0)
    w._move_selected(1)
    assert [s.step_id for s in w.state.pipeline.steps] == ["resample", "bandpass_filter"]


def test_viewer_handles_an_epochs_result(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w._load_eeg_sample()
    w.state.add_step("epochs_fixed", {"duration": 1.0})
    w.state.run_sync()  # Raw -> Epochs (3-D)
    w._replot()  # must not raise on 3-D data
    assert w.state.result.payload.get_data().ndim == 3


def test_viewer_handles_an_evoked_result(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w._load_eeg_sample()
    w.state.add_step("epochs_fixed", {"duration": 1.0})
    w.state.add_step("average")
    w.state.run_sync()  # Raw -> Epochs -> Evoked (2-D)
    w._replot()
    assert w.state.result.payload.get_data().ndim == 2


def test_time_series_is_the_default_view(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    assert w._view_mode == "time"


def test_switching_to_power_spectrum_view_replots(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w._load_eeg_sample()
    w.view_selector.setCurrentText("Power spectrum")
    assert w._view_mode == "psd"


def test_topography_is_a_view_option(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w._load_eeg_sample()
    w.view_selector.setCurrentText("Topography")
    assert w._view_mode == "topo"


def test_topomap_renders_for_a_sample_with_a_montage(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w._load_eeg_sample()
    w._set_view("topo")  # must render the head map without raising
    assert w.band_selector.count() == 5  # delta/theta/alpha/beta/gamma


def test_changing_the_band_updates_state(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w._load_eeg_sample()
    w._set_view("topo")
    w.band_selector.setCurrentText("Beta")
    assert w._band == "Beta"


def test_export_report_writes_html(qtbot, tmp_path):
    w = MainWindow()
    qtbot.addWidget(w)
    w._load_eeg_sample()
    w.state.add_step("bandpass_filter")
    out = tmp_path / "report.html"
    w._export_report_to(str(out))
    assert out.exists()
    assert "Band-pass filter" in out.read_text()


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
