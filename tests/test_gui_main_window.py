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


def test_channel_list_populates_all_checked_on_load(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w._load_eeg_sample()
    assert w.channel_list.count() == 10  # the EEG sample has 10 channels
    assert all(
        w.channel_list.item(i).checkState() == Qt.CheckState.Checked
        for i in range(w.channel_list.count())
    )


def test_unchecking_a_channel_removes_it_from_picks(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w._load_eeg_sample()
    first = w.channel_list.item(0).text()
    w.channel_list.item(0).setCheckState(Qt.CheckState.Unchecked)
    picks = w._current_picks()
    assert first not in picks
    assert len(picks) == 9


def test_time_frequency_view_renders(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w._load_eeg_sample()
    w.view_selector.setCurrentText("Time-frequency")  # must render without raising
    assert w._view_mode == "tfr"


def test_connectivity_is_a_view_option(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w._load_eeg_sample()
    w.view_selector.setCurrentText("Connectivity")  # raw data -> shows "needs epochs"
    assert w._view_mode == "conn"


def test_connectivity_renders_with_epochs(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w._load_eeg_sample()
    w.state.add_step("epochs_fixed", {"duration": 1.0})
    w.state.run_sync()  # result is Epochs
    w._set_view("conn")  # computes the matrix + renders without raising
    assert w._view_mode == "conn"


def test_connectivity_offers_the_full_measure_set(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    assert w.conn_method_selector.count() == 6  # plv/coh/wpli/imcoh/pli/ciplv


def test_connectogram_style_renders_with_epochs(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w._load_eeg_sample()
    w.state.add_step("epochs_fixed", {"duration": 1.0})
    w.state.run_sync()
    w._set_view("conn")
    w.conn_style_selector.setCurrentText("Connectogram")  # circular graph, must render
    assert w._conn_style == "connectogram"


def test_connectivity_edge_threshold_updates_state(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w._load_eeg_sample()
    w.state.add_step("epochs_fixed", {"duration": 1.0})
    w.state.run_sync()
    w._set_view("conn")
    w.conn_threshold_slider.setValue(50)  # show only edges >= 50% of the strongest
    assert w._conn_threshold == 0.5


def test_changing_band_recomputes_connectivity(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w._load_eeg_sample()
    w.state.add_step("epochs_fixed", {"duration": 1.0})
    w.state.run_sync()
    w._set_view("conn")  # computes for the default band (Alpha)
    w.band_selector.setCurrentText("Beta")  # must recompute for the new band
    assert w._conn_cache[1] == ("plv", "Beta")


def test_connectivity_matrix_is_cached_across_style_switch(qtbot, monkeypatch):
    from cortica.gui import main_window

    w = MainWindow()
    qtbot.addWidget(w)
    w._load_eeg_sample()
    w.state.add_step("epochs_fixed", {"duration": 1.0})
    w.state.run_sync()
    calls = {"n": 0}
    real = main_window.viz.connectivity

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(main_window.viz, "connectivity", counting)
    w._set_view("conn")  # computes once
    assert calls["n"] == 1
    w.conn_style_selector.setCurrentText("Connectogram")  # reuse cache — no recompute
    assert calls["n"] == 1


def test_decoding_view_renders_with_multi_condition_epochs(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w._load_eeg_sample()
    w.state.add_step("epochs_events", {"tmin": -0.1, "tmax": 0.4})
    w.state.run_sync()  # Epochs with target/standard
    w._set_view("decoding")  # cross-validated decoding, must render
    assert w._view_mode == "decoding"


def test_decoding_temporal_generalization_mode_renders(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w._load_eeg_sample()
    w.state.add_step("epochs_events", {"tmin": -0.1, "tmax": 0.4})
    w.state.run_sync()
    w._set_view("decoding")
    w.decode_mode_selector.setCurrentText("Temporal generalization")  # train×test heatmap
    assert w._decode_mode == "generalization"


def test_decoding_csp_mode_with_lda_classifier_renders(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w._load_eeg_sample()
    w.state.add_step("epochs_events", {"tmin": -0.1, "tmax": 0.4})
    w.state.run_sync()
    w._set_view("decoding")
    w.classifier_selector.setCurrentText("LDA")
    w.decode_mode_selector.setCurrentText("CSP (whole epoch)")
    assert w._decode_mode == "csp"
    assert w._decode_classifier == "lda"


def test_statistics_view_renders_with_multi_condition_epochs(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w._load_eeg_sample()
    w.state.add_step("epochs_events", {"tmin": -0.1, "tmax": 0.4})
    w.state.run_sync()
    w._set_view("stats")  # cluster permutation comparison, must render
    assert w._view_mode == "stats"


def test_comparison_view_renders_with_multi_condition_epochs(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w._load_eeg_sample()
    w.state.add_step("epochs_events", {"tmin": -0.1, "tmax": 0.4})
    w.state.run_sync()
    w._set_view("compare")  # overlays conditions + difference wave, must render
    assert w._view_mode == "compare"


def test_source_view_renders_a_result(qtbot):
    # exercise the view + rendering without the heavy fsaverage computation
    w = MainWindow()
    qtbot.addWidget(w)
    w._on_sources_ready((["superiorparietal-lh", "cuneus-rh"], [2.0, 1.5]))
    assert w._view_mode == "source"
    assert w._source_result[0][0] == "superiorparietal-lh"


def test_set_view_syncs_the_view_dropdown(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w._load_eeg_sample()
    w._set_view("topo")  # programmatic view change must keep the dropdown in step
    assert w.view_selector.currentText() == "Topography"


def test_action_driven_source_view_syncs_the_dropdown(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w._on_sources_ready((["superiorparietal-lh"], [2.0]))  # opened via toolbar action
    assert w.view_selector.currentText() == "Source"


def test_busy_indicator_sets_and_restores_the_cursor(qtbot):
    from PySide6.QtWidgets import QApplication

    w = MainWindow()
    qtbot.addWidget(w)
    w._show_computing("Computing…")
    assert QApplication.overrideCursor() is not None  # user sees a wait cursor
    w._done_computing()
    assert QApplication.overrideCursor() is None  # restored afterwards


def test_workflow_guide_populates_after_load(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w._load_eeg_sample()
    assert w.stage_label.text()  # the stage tracker rendered
    assert w._next_layout.count() > 0  # at least one suggested next step


def test_workflow_step_suggestion_adds_it_to_the_pipeline(qtbot):
    from cortica.workflow import Suggestion

    w = MainWindow()
    qtbot.addWidget(w)
    w._load_eeg_sample()
    w._apply_suggestion(Suggestion("Band-pass filter", "step", "bandpass_filter", ""))
    assert w.state.pipeline.steps[-1].step_id == "bandpass_filter"


def test_workflow_view_suggestion_switches_the_view(qtbot):
    from cortica.workflow import Suggestion

    w = MainWindow()
    qtbot.addWidget(w)
    w._load_eeg_sample()
    w._apply_suggestion(Suggestion("Decoding", "view", "decoding", ""))
    assert w._view_mode == "decoding"


def test_workflow_suggestions_update_when_data_becomes_epochs(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w._load_eeg_sample()
    w.state.add_step("epochs_events", {"tmin": -0.1, "tmax": 0.4})
    w.state.run_sync()  # current() is now Epochs -> suggestions change
    texts = [w._next_layout.itemAt(i).widget().text() for i in range(w._next_layout.count())]
    assert any("Average" in t for t in texts)


def test_apply_ica_choice_adds_an_ica_step(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w._load_eeg_sample()
    w._apply_ica_choice(6, [0, 2])
    step = w.state.pipeline.steps[-1]
    assert step.step_id == "ica"
    assert step.params["n_components"] == 6
    assert step.params["exclude"] == "0,2"


def test_export_report_writes_html(qtbot, tmp_path):
    w = MainWindow()
    qtbot.addWidget(w)
    w._load_eeg_sample()
    w.state.add_step("bandpass_filter")
    out = tmp_path / "report.html"
    w._export_report_to(str(out))
    assert out.exists()
    assert "Band-pass filter" in out.read_text()


def test_save_figure_button_is_present_in_the_view_row(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    assert w.save_fig_button.text().startswith("Save figure")  # always-visible, not toolbar-only


def test_save_figure_writes_svg_for_a_matplotlib_view(qtbot, tmp_path):
    w = MainWindow()
    qtbot.addWidget(w)
    w._load_eeg_sample()
    w._set_view("topo")  # a matplotlib view
    out = tmp_path / "fig.svg"
    w._save_figure_to(str(out))
    assert out.exists() and out.stat().st_size > 0


def test_save_figure_writes_png_for_the_timeseries_view(qtbot, tmp_path):
    w = MainWindow()
    qtbot.addWidget(w)
    w._load_eeg_sample()  # time series (pyqtgraph) is the default view
    out = tmp_path / "fig.png"
    w._save_figure_to(str(out))
    assert out.exists() and out.stat().st_size > 0


def test_preview_report_builds_and_opens_in_browser(qtbot, monkeypatch):
    import webbrowser
    from pathlib import Path

    w = MainWindow()
    qtbot.addWidget(w)
    w._load_eeg_sample()
    w.state.add_step("bandpass_filter")
    opened = []
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url) or True)
    w._preview_report()
    assert opened  # the browser was invoked
    assert Path(w._last_preview_path).exists()
    assert "Band-pass filter" in Path(w._last_preview_path).read_text()


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
