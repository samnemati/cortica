"""Tests for AppState — the GUI's controller (no widgets involved).

It holds the loaded source, the pipeline being built, and the last result, and
exposes a modality-aware list of available steps. Widgets are thin views over it.
"""
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from cortica.core.dataset import Dataset  # noqa: E402
from cortica.core.params import Float  # noqa: E402
from cortica.core.registry import Registry  # noqa: E402
from cortica.core.step import Step  # noqa: E402
from cortica.gui.state import AppState  # noqa: E402


class Scale(Step):
    id = "scale"
    name = "Scale"
    category = "Preprocess"
    modalities = ["eeg", "fnirs"]
    params = [Float("factor", 2.0)]

    def run(self, ds, p):
        return ds.derive(ds.payload * p["factor"])


class EegOnly(Step):
    id = "eeg_only"
    name = "EEG only"
    category = "Preprocess"
    modalities = ["eeg"]
    params = []

    def run(self, ds, p):
        return ds.derive(ds.payload)


@pytest.fixture
def reg():
    r = Registry()
    r.register(Scale)
    r.register(EegOnly)
    return r


def test_set_source_sets_modality_and_resets_pipeline(qtbot, reg):
    st = AppState(registry=reg)
    st.set_source(Dataset(3.0, "fnirs"))
    assert st.modality == "fnirs"
    assert st.pipeline.modality == "fnirs"
    assert st.pipeline.steps == []


def test_available_steps_flag_applicability_by_modality(qtbot, reg):
    st = AppState(registry=reg)
    st.set_source(Dataset(1.0, "fnirs"))
    applicable = {cls.id: ok for cls, ok in st.available_steps()}
    assert applicable["scale"] is True
    assert applicable["eeg_only"] is False  # EEG-only step greyed on fNIRS data


def test_add_step_emits_pipeline_changed(qtbot, reg):
    st = AppState(registry=reg)
    st.set_source(Dataset(2.0, "eeg"))
    with qtbot.waitSignal(st.pipelineChanged):
        st.add_step("scale", {"factor": 3})
    assert len(st.pipeline.steps) == 1


def test_run_applies_pipeline_to_source_without_mutating_it(qtbot, reg):
    st = AppState(registry=reg)
    st.set_source(Dataset(2.0, "eeg"))
    st.add_step("scale", {"factor": 3})
    with qtbot.waitSignal(st.resultChanged):
        result = st.run_sync()
    assert result.payload == 6.0
    assert st.current().payload == 6.0
    assert st.source.payload == 2.0  # source untouched


def test_run_without_source_raises(qtbot, reg):
    st = AppState(registry=reg)
    with pytest.raises(RuntimeError):
        st.run_sync()
