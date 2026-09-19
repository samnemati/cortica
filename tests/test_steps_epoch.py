"""Tests for epoching and averaging steps.

Fixed-length epoching works on any continuous recording (no event channel needed),
so it runs on the synthetic samples. Averaging turns epochs into an evoked response.
Preconditions make out-of-order use fail with a clear message instead of crashing.
"""
import numpy as np
import pytest

mne = pytest.importorskip("mne")

from cortica.core.errors import StepError  # noqa: E402
from cortica.core.registry import default_registry  # noqa: E402
from cortica.io import dataset_from_raw  # noqa: E402
from cortica.samples import eeg_sample  # noqa: E402
from cortica.steps.epoch import Average, EventEpochs, FixedLengthEpochs  # noqa: E402


def _raw(dur=10.0, sfreq=100.0):
    n = int(dur * sfreq)
    data = np.random.RandomState(0).randn(3, n) * 1e-6
    info = mne.create_info(["EEG 1", "EEG 2", "EEG 3"], sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose=False)


def _raw_with_stim(dur=10.0, sfreq=100.0):
    n = int(dur * sfreq)
    eeg = mne.io.RawArray(
        np.random.RandomState(0).randn(2, n) * 1e-6,
        mne.create_info(["EEG 1", "EEG 2"], sfreq, ch_types="eeg"),
        verbose=False,
    )
    stim = np.zeros((1, n))
    stim[0, [100, 300, 500, 700]] = 1  # four trigger pulses
    stim_raw = mne.io.RawArray(stim, mne.create_info(["STI 014"], sfreq, ["stim"]), verbose=False)
    eeg.add_channels([stim_raw], force_update_info=True)
    return eeg


def test_fixed_length_epochs_segments_the_recording():
    ds = dataset_from_raw(_raw(dur=10.0, sfreq=100.0))
    out = FixedLengthEpochs().apply(ds, {"duration": 2.0})
    data = out.payload.get_data()
    assert data.ndim == 3          # (n_epochs, n_channels, n_times)
    assert data.shape[0] == 5      # 10 s / 2 s
    assert data.shape[1] == 3      # channels preserved
    assert out.modality == "eeg"


def test_epoching_rejects_already_epoched_data():
    ds = dataset_from_raw(_raw())
    epoched = FixedLengthEpochs().apply(ds, {"duration": 2.0})
    with pytest.raises(StepError):
        FixedLengthEpochs().apply(epoched, {"duration": 1.0})


def test_average_requires_epochs():
    ds = dataset_from_raw(_raw())  # continuous Raw, not epoched
    with pytest.raises(StepError):
        Average().apply(ds, {})


def test_average_produces_an_evoked_from_epochs():
    ds = dataset_from_raw(_raw(dur=10.0, sfreq=100.0))
    epoched = FixedLengthEpochs().apply(ds, {"duration": 2.0})
    evoked = Average().apply(epoched, {})
    data = evoked.payload.get_data()
    assert data.ndim == 2          # (n_channels, n_times) — an Evoked
    assert data.shape[0] == 3


def test_event_epochs_segments_around_annotations():
    out = EventEpochs().apply(eeg_sample(), {"tmin": -0.1, "tmax": 0.4})
    data = out.payload.get_data()
    assert data.ndim == 3
    assert data.shape[0] >= 1  # at least one event epoched


def test_event_epochs_requires_events():
    ds = dataset_from_raw(_raw())  # _raw() has no annotations
    with pytest.raises(StepError):
        EventEpochs().apply(ds, {"tmin": -0.1, "tmax": 0.4})


def test_event_epochs_rejects_already_epoched_data():
    epoched = FixedLengthEpochs().apply(dataset_from_raw(_raw()), {"duration": 1.0})
    with pytest.raises(StepError):
        EventEpochs().apply(epoched, {})


def test_event_epochs_from_a_stim_channel():
    out = EventEpochs().apply(
        dataset_from_raw(_raw_with_stim()), {"tmin": -0.1, "tmax": 0.3, "source": "stim"}
    )
    data = out.payload.get_data()
    assert data.ndim == 3
    assert data.shape[0] >= 1


def test_epoch_steps_register_themselves():
    import cortica.steps  # noqa: F401
    assert default_registry.get("epochs_fixed") is FixedLengthEpochs
    assert default_registry.get("epochs_events") is EventEpochs
    assert default_registry.get("average") is Average
