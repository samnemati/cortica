"""Tests for loading data into a Dataset (modality detection + file reading)."""
import numpy as np
import pytest

mne = pytest.importorskip("mne")

from cortica.io import dataset_from_raw, load_raw  # noqa: E402


def _eeg_raw(sfreq=200.0, n=400):
    info = mne.create_info(["EEG 001", "EEG 002"], sfreq, ch_types="eeg")
    data = np.random.RandomState(0).randn(2, n) * 1e-6
    return mne.io.RawArray(data, info, verbose=False)


def test_dataset_from_raw_detects_eeg_and_records_meta():
    ds = dataset_from_raw(_eeg_raw())
    assert ds.modality == "eeg"
    assert ds.meta["sfreq"] == 200.0
    assert ds.meta["n_channels"] == 2


def test_dataset_from_raw_detects_fnirs_from_channel_types():
    info = mne.create_info(["S1_D1 hbo", "S1_D1 hbr"], 10.0, ch_types=["hbo", "hbr"])
    raw = mne.io.RawArray(np.zeros((2, 50)), info, verbose=False)
    assert dataset_from_raw(raw).modality == "fnirs"


def test_load_raw_reads_a_fif_file(tmp_path):
    raw = _eeg_raw()
    path = tmp_path / "sample_raw.fif"
    raw.save(path, overwrite=True, verbose=False)

    ds = load_raw(str(path))

    assert ds.modality == "eeg"
    assert ds.payload.info["sfreq"] == 200.0
