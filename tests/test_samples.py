"""Tests for the built-in synthetic sample recordings (used by the GUI's
"Load sample" menu and by demos). They must be detected as the right modality."""
import pytest

mne = pytest.importorskip("mne")

from cortica.samples import eeg_sample, fnirs_sample  # noqa: E402


def test_eeg_sample_is_detected_as_eeg():
    ds = eeg_sample()
    assert ds.modality == "eeg"
    assert ds.meta["n_channels"] >= 4
    assert hasattr(ds.payload, "get_data")  # a real MNE Raw


def test_fnirs_sample_is_detected_as_fnirs():
    ds = fnirs_sample()
    assert ds.modality == "fnirs"
    assert ds.meta["n_channels"] >= 2
    assert hasattr(ds.payload, "get_data")
