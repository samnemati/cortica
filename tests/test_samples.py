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


def test_eeg_sample_carries_stimulus_events_for_epoching():
    ds = eeg_sample()
    annotations = ds.payload.annotations
    assert len(annotations) > 0
    assert "target" in set(annotations.description)


def test_fnirs_sample_is_detected_as_fnirs():
    ds = fnirs_sample()
    assert ds.modality == "fnirs"
    assert ds.meta["n_channels"] >= 2
    assert hasattr(ds.payload, "get_data")


def test_fnirs_sample_is_raw_cw_amplitude_so_the_chain_can_run():
    ds = fnirs_sample()
    assert "fnirs_cw_amplitude" in set(ds.payload.get_channel_types())
