"""Tests for the Pipeline: ordered execution, serialization, and the
reproducibility guarantee (run -> serialize -> reload -> re-run is identical)."""
import pytest

from cortica.core.dataset import Dataset
from cortica.core.errors import StepError
from cortica.core.params import Float, Int
from cortica.core.pipeline import Pipeline
from cortica.core.registry import Registry
from cortica.core.step import Step


class Scale(Step):
    id = "scale"
    name = "Scale"
    category = "Test"
    modalities = ["eeg", "fnirs"]
    params = [Float("factor", 2.0)]

    def run(self, ds, p):
        return ds.derive(ds.payload * p["factor"])


class AddN(Step):
    id = "add_n"
    name = "Add N"
    category = "Test"
    modalities = ["eeg", "fnirs"]
    params = [Int("n", 1)]

    def run(self, ds, p):
        return ds.derive(ds.payload + p["n"])


@pytest.fixture
def registry():
    r = Registry()
    r.register(Scale)
    r.register(AddN)
    return r


def test_run_executes_steps_in_order(registry):
    p = Pipeline("eeg").add("scale", {"factor": 3}).add("add_n", {"n": 10})
    out = p.run(Dataset(2, "eeg"), registry)
    assert out.payload == 16  # (2 * 3) + 10 — order matters


def test_run_records_validated_history(registry):
    p = Pipeline("eeg").add("scale", {"factor": 3})
    out = p.run(Dataset(2, "eeg"), registry)
    assert out.history[-1] == {"step": "scale", "params": {"factor": 3.0}}


def test_to_dict_includes_version_modality_and_steps(registry):
    p = Pipeline("eeg").add("scale", {"factor": 3})
    d = p.to_dict()
    assert d["modality"] == "eeg"
    assert d["steps"] == [{"id": "scale", "params": {"factor": 3}}]
    assert "cortica_version" in d


def test_yaml_roundtrip_preserves_steps_and_modality(registry):
    p = Pipeline("eeg").add("scale", {"factor": 3}).add("add_n", {"n": 10})
    reloaded = Pipeline.from_yaml(p.to_yaml())
    assert reloaded.modality == "eeg"
    assert reloaded.to_dict()["steps"] == p.to_dict()["steps"]


def test_reproducibility_run_serialize_reload_rerun_is_identical(registry):
    src = Dataset(2, "eeg")
    p = Pipeline("eeg").add("scale", {"factor": 3}).add("add_n", {"n": 10})

    first = p.run(src, registry)
    second = Pipeline.from_yaml(p.to_yaml()).run(src, registry)

    assert first.payload == second.payload
    assert first.history == second.history


def test_run_propagates_step_error_on_wrong_modality(registry):
    class EegOnly(Step):
        id = "eeg_only"
        name = "EEG only"
        modalities = ["eeg"]
        params = []

        def run(self, ds, p):
            return ds.derive(ds.payload)

    registry.register(EegOnly)
    p = Pipeline("fnirs").add("eeg_only")
    with pytest.raises(StepError):
        p.run(Dataset(1, "fnirs"), registry)
