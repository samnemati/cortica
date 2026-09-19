"""Tests for MNE-backed preprocessing steps.

Uses tiny in-memory RawArrays (no downloads). The band-pass test checks real
behavior: a 50 Hz component is attenuated while a 5 Hz component is retained.
"""
import numpy as np
import pytest

mne = pytest.importorskip("mne")

from cortica.core.registry import default_registry  # noqa: E402
from cortica.io import dataset_from_raw  # noqa: E402
from cortica.samples import eeg_sample  # noqa: E402
from cortica.steps.preprocess import (  # noqa: E402
    AverageReference,
    BandpassFilter,
    InterpolateBads,
    NotchFilter,
    Resample,
)

SFREQ = 200.0


def _raw_two_tones(dur=8.0):
    n = int(SFREQ * dur)
    t = np.arange(n) / SFREQ
    data = (np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 50 * t))[None, :] * 1e-6
    info = mne.create_info(["EEG 001"], SFREQ, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose=False)


def _mag_at(x, freq):
    freqs = np.fft.rfftfreq(len(x), 1 / SFREQ)
    spectrum = np.abs(np.fft.rfft(x))
    return spectrum[np.argmin(np.abs(freqs - freq))]


def test_bandpass_filter_attenuates_out_of_band_and_keeps_in_band():
    raw = _raw_two_tones()
    out = BandpassFilter().apply(dataset_from_raw(raw), {"l_freq": 1.0, "h_freq": 20.0})

    x0, x1 = raw.get_data()[0], out.payload.get_data()[0]
    keep_ratio = _mag_at(x1, 5) / _mag_at(x0, 5)
    drop_ratio = _mag_at(x1, 50) / _mag_at(x0, 50)

    assert drop_ratio < 0.3   # 50 Hz strongly attenuated
    assert keep_ratio > 0.6   # 5 Hz largely retained


def test_bandpass_does_not_mutate_the_input_raw():
    raw = _raw_two_tones()
    before = raw.get_data().copy()
    BandpassFilter().apply(dataset_from_raw(raw), {"l_freq": 1.0, "h_freq": 20.0})
    assert np.allclose(raw.get_data(), before)


def test_resample_changes_sfreq_and_updates_meta():
    ds = dataset_from_raw(_raw_two_tones(dur=4.0))
    out = Resample().apply(ds, {"sfreq": 100.0})

    assert out.payload.info["sfreq"] == 100.0
    assert out.meta["sfreq"] == 100.0
    assert out.payload.n_times == 400  # 100 Hz * 4 s


def test_notch_filter_runs_and_returns_a_new_raw():
    out = NotchFilter().apply(eeg_sample(), {"freq": 60.0})
    assert hasattr(out.payload, "get_data")


def test_average_reference_zeroes_the_cross_channel_mean():
    out = AverageReference().apply(eeg_sample(), {})
    assert np.allclose(out.payload.get_data().mean(axis=0), 0, atol=1e-12)


def test_interpolate_bads_clears_the_bad_list():
    raw = eeg_sample().payload.copy()
    raw.info["bads"] = ["O1"]
    out = InterpolateBads().apply(dataset_from_raw(raw), {})
    assert out.payload.info["bads"] == []


def test_preprocess_steps_register_themselves():
    for step_id, cls in [
        ("bandpass_filter", BandpassFilter),
        ("notch_filter", NotchFilter),
        ("resample", Resample),
        ("reref_average", AverageReference),
        ("interpolate_bads", InterpolateBads),
    ]:
        assert default_registry.get(step_id) is cls
