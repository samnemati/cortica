"""Tests for the ICA artifact-removal step.

A fixed random_state makes the decomposition (and therefore which component index
means what) reproducible, so an ``exclude`` list is stable across runs.
"""
import pytest

mne = pytest.importorskip("mne")

from cortica.core.registry import default_registry  # noqa: E402
from cortica.samples import eeg_sample  # noqa: E402
from cortica.steps.ica import ICA  # noqa: E402


def test_ica_fits_excludes_and_applies_keeping_channels():
    ds = eeg_sample()
    out = ICA().apply(ds, {"n_components": 4, "exclude": "0"})
    assert len(out.payload.ch_names) == len(ds.payload.ch_names)
    assert hasattr(out.payload, "get_data")


def test_ica_registered():
    import cortica.steps  # noqa: F401

    assert default_registry.get("ica") is ICA
