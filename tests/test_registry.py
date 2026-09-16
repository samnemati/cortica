"""Tests for the step registry (lookup + modality filtering)."""
import pytest

from cortica.core.registry import Registry
from cortica.core.step import Step


class FakeEeg(Step):
    id = "fake_eeg"
    name = "Fake EEG"
    category = "Test"
    modalities = ["eeg"]


class FakeBoth(Step):
    id = "fake_both"
    name = "Fake Both"
    category = "Test"
    modalities = ["eeg", "fnirs"]


def test_register_and_get_returns_the_class():
    r = Registry()
    r.register(FakeEeg)
    assert r.get("fake_eeg") is FakeEeg


def test_get_unknown_id_raises_keyerror():
    r = Registry()
    with pytest.raises(KeyError):
        r.get("nope")


def test_register_duplicate_id_raises():
    r = Registry()
    r.register(FakeEeg)
    with pytest.raises(ValueError):
        r.register(FakeEeg)


def test_register_step_without_id_raises():
    class NoId(Step):
        pass

    r = Registry()
    with pytest.raises(ValueError):
        r.register(NoId)


def test_for_modality_filters_to_applicable_steps():
    r = Registry()
    r.register(FakeEeg)
    r.register(FakeBoth)
    assert {c.id for c in r.for_modality("fnirs")} == {"fake_both"}
    assert {c.id for c in r.for_modality("eeg")} == {"fake_eeg", "fake_both"}


def test_all_returns_every_registered_class():
    r = Registry()
    r.register(FakeEeg)
    r.register(FakeBoth)
    assert set(r.all()) == {FakeEeg, FakeBoth}
