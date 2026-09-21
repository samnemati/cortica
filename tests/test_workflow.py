"""Tests for the guided-workflow logic (pure, Qt-free)."""
import numpy as np
import pytest

mne = pytest.importorskip("mne")

from cortica.io import dataset_from_raw  # noqa: E402
from cortica.samples import eeg_sample, fnirs_sample  # noqa: E402
from cortica.steps.epoch import Average, EventEpochs  # noqa: E402
from cortica.workflow import (  # noqa: E402
    STAGES,
    current_stage,
    stage_status,
    suggestions,
)


def _raw_no_montage():
    info = mne.create_info(["C3", "C4", "Cz"], 200.0, ch_types="eeg")
    return dataset_from_raw(mne.io.RawArray(np.zeros((3, 400)), info, verbose=False))


def test_stages_run_import_to_report():
    assert [key for key, _ in STAGES] == ["import", "preprocess", "segment", "analyze", "report"]


def test_no_data_suggests_opening_a_recording():
    sug = suggestions(None, None)
    assert sug[0].kind == "action"
    assert sug[0].target == "open"


def test_current_stage_tracks_the_data_kind():
    assert current_stage(None) == "import"
    assert current_stage(eeg_sample()) == "preprocess"
    epochs = EventEpochs().apply(eeg_sample(), {"tmin": -0.1, "tmax": 0.4})
    assert current_stage(epochs) == "analyze"


def test_stage_status_marks_done_current_todo():
    rows = stage_status(eeg_sample())  # raw -> preprocess is current
    status = {key: st for key, _, st in rows}
    assert status["import"] == "done"
    assert status["preprocess"] == "current"
    assert status["analyze"] == "todo"


def test_raw_without_montage_suggests_setting_one():
    targets = [s.target for s in suggestions(_raw_no_montage(), None)]
    assert "set_montage" in targets


def test_eeg_sample_already_has_montage_so_none_is_suggested():
    # the sample carries positions already; suggest filtering/epoching instead
    targets = [s.target for s in suggestions(eeg_sample(), None)]
    assert "set_montage" not in targets
    assert "bandpass_filter" in targets
    assert "epochs_events" in targets


def test_fnirs_raw_suggests_optical_density_first():
    targets = [s.target for s in suggestions(fnirs_sample(), None)]
    assert targets[0] == "optical_density"


def test_epochs_suggest_averaging_and_analysis_views():
    epochs = EventEpochs().apply(eeg_sample(), {"tmin": -0.1, "tmax": 0.4})
    sug = suggestions(epochs, None)
    kinds = {s.target: s.kind for s in sug}
    assert kinds.get("average") == "step"
    assert kinds.get("decoding") == "view"
    assert kinds.get("conn") == "view"


def test_evoked_suggests_source_localization_and_report():
    evoked = Average().apply(
        EventEpochs().apply(eeg_sample(), {"tmin": -0.1, "tmax": 0.4}), {}
    )
    targets = [s.target for s in suggestions(evoked, None)]
    assert "source" in targets
    assert "preview" in targets
