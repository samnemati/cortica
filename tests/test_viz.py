"""Tests for viz — pure data-prep for the signal views (no Qt).

Keeping this logic out of the widgets means it runs in CI without a Qt binding and
the plotting code stays thin.
"""
import numpy as np
import pytest

mne = pytest.importorskip("mne")

from cortica.samples import eeg_sample  # noqa: E402
from cortica.steps.epoch import FixedLengthEpochs  # noqa: E402
from cortica.viz import BANDS, band_power, spectrum, time_frequency, traces  # noqa: E402


def test_traces_returns_times_and_2d_data_for_raw():
    times, data = traces(eeg_sample().payload)
    assert data.ndim == 2
    assert len(times) == data.shape[1]


def test_traces_averages_epochs_to_2d():
    epochs = FixedLengthEpochs().apply(eeg_sample(), {"duration": 1.0})
    times, data = traces(epochs.payload)
    assert data.ndim == 2  # epochs collapsed to their mean


def test_spectrum_returns_freqs_and_power_per_channel():
    freqs, psds = spectrum(eeg_sample().payload, fmax=40.0)
    assert freqs.ndim == 1
    assert psds.ndim == 2
    assert psds.shape[1] == len(freqs)
    assert freqs[0] >= 0


def test_spectrum_shows_the_alpha_peak():
    # the EEG sample has a 10 Hz alpha rhythm; 10 Hz power must exceed 30 Hz power
    freqs, psds = spectrum(eeg_sample().payload, fmax=40.0)
    mean_power = psds.mean(axis=0)

    def power_at(hz):
        return mean_power[np.argmin(np.abs(freqs - hz))]

    assert power_at(10) > power_at(30)


def test_bands_are_the_standard_five():
    assert set(BANDS) == {"Delta", "Theta", "Alpha", "Beta", "Gamma"}


def test_band_power_returns_one_value_per_channel():
    ds = eeg_sample()
    power = band_power(ds.payload, "Alpha")
    assert power.shape == (ds.meta["n_channels"],)


def test_alpha_band_power_exceeds_gamma_for_the_sample():
    ds = eeg_sample()
    assert band_power(ds.payload, "Alpha").mean() > band_power(ds.payload, "Gamma").mean()


def test_traces_can_subset_channels():
    times, data = traces(eeg_sample().payload, picks=["O1", "O2"])
    assert data.shape[0] == 2


def test_spectrum_can_subset_channels():
    freqs, psds = spectrum(eeg_sample().payload, fmax=40.0, picks=["O1", "O2"])
    assert psds.shape[0] == 2


def test_band_power_can_subset_channels():
    power = band_power(eeg_sample().payload, "Alpha", picks=["O1", "O2"])
    assert power.shape == (2,)


def test_time_frequency_returns_freq_by_time_power():
    times, freqs, power = time_frequency(eeg_sample().payload, picks=["O1", "O2"], fmax=30.0)
    assert power.ndim == 2
    assert power.shape == (len(freqs), len(times))
