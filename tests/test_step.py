"""Tests for the Step base class and its apply() orchestration.

A concrete test step (no MNE) exercises the base contract: modality gating,
parameter validation/defaults, preconditions, and automatic history recording.
"""
import pytest

from cortica.core.dataset import Dataset
from cortica.core.errors import ParamError, StepError
from cortica.core.params import Float
from cortica.core.step import Step


class AddOffset(Step):
    id = "add_offset"
    name = "Add offset"
    category = "Test"
    modalities = ["eeg", "fnirs"]
    params = [Float("offset", 1.0)]

    def run(self, ds, p):
        return ds.derive(ds.payload + p["offset"])


def make_ds(payload=0.0, modality="eeg"):
    return Dataset(payload, modality)


def test_apply_runs_and_appends_history():
    out = AddOffset().apply(make_ds(0.0), {"offset": 2.5})
    assert out.payload == 2.5
    assert out.history[-1] == {"step": "add_offset", "params": {"offset": 2.5}}


def test_apply_does_not_mutate_input_dataset():
    ds = make_ds(0.0)
    AddOffset().apply(ds, {"offset": 2.5})
    assert ds.payload == 0.0
    assert ds.history == ()


def test_apply_uses_default_when_param_missing():
    out = AddOffset().apply(make_ds(0.0), {})
    assert out.payload == 1.0


def test_apply_rejects_wrong_modality():
    step = AddOffset()
    step.modalities = ["fnirs"]  # instance override: pretend fNIRS-only
    with pytest.raises(StepError):
        step.apply(make_ds(0.0, "eeg"))


def test_apply_rejects_unknown_param():
    with pytest.raises(ParamError):
        AddOffset().apply(make_ds(0.0), {"bogus": 1})


def test_check_precondition_can_block_run():
    class NeedsNonNegative(AddOffset):
        id = "needs_non_negative"

        def check(self, ds):
            if ds.payload < 0:
                raise StepError("payload must be non-negative")

    with pytest.raises(StepError):
        NeedsNonNegative().apply(make_ds(-1.0))


def test_applies_to_reflects_modalities():
    step = AddOffset()
    assert step.applies_to("eeg") is True
    assert step.applies_to("meg") is False
