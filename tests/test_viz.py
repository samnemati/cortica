"""Tests for viz — pure data-prep for the signal views (no Qt).

Keeping this logic out of the widgets means it runs in CI without a Qt binding and
the plotting code stays thin.
"""
import os.path as op

import numpy as np
import pytest

mne = pytest.importorskip("mne")

from cortica.samples import eeg_sample, fnirs_sample  # noqa: E402
from cortica.steps.epoch import Average, EventEpochs, FixedLengthEpochs  # noqa: E402
from cortica.steps.fnirs import BeerLambert, OpticalDensity  # noqa: E402
from cortica.viz import (  # noqa: E402
    BANDS,
    CONNECTIVITY_METHODS,
    DECODE_CLASSIFIERS,
    band_power,
    cluster_test,
    condition_comparison,
    connectivity,
    decoding,
    decoding_csp,
    glm_analysis,
    glm_conditions,
    glm_contrast,
    regression_erp,
    source_localization,
    spatiotemporal_cluster_test,
    spectrum,
    temporal_generalization,
    threshold_matrix,
    time_frequency,
    traces,
)

_FSAVERAGE = op.expanduser("~/mne_data/MNE-fsaverage-data/fsaverage/bem/fsaverage-ico-5-src.fif")


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


def test_time_frequency_supports_multitaper():
    times, freqs, power = time_frequency(
        eeg_sample().payload, picks=["O1"], fmax=20.0, method="multitaper"
    )
    assert power.shape == (len(freqs), len(times))


def test_connectivity_methods_include_plv():
    assert "PLV" in CONNECTIVITY_METHODS


def test_connectivity_methods_cover_the_full_set():
    assert set(CONNECTIVITY_METHODS.values()) == {
        "plv", "coh", "wpli", "imcoh", "pli", "ciplv"
    }


def test_connectivity_returns_a_channel_by_channel_matrix():
    epochs = FixedLengthEpochs().apply(eeg_sample(), {"duration": 1.0})
    matrix, names = connectivity(epochs.payload, method="plv", band="Alpha")
    assert len(names) == 10
    assert matrix.shape == (10, 10)


def test_threshold_matrix_keeps_only_the_strongest_edges():
    # off-diagonal edges are 0.2, 0.5, 0.9; drop the weakest ~2/3 -> keep the top one
    m = np.array([[0.0, 0.2, 0.9], [0.2, 0.0, 0.5], [0.9, 0.5, 0.0]])
    out = threshold_matrix(m, 0.66)
    assert out[0, 2] == 0.9 and out[2, 0] == 0.9  # strongest kept, still symmetric
    assert out[1, 2] == 0.0  # 0.5 dropped
    assert out[0, 1] == 0.0  # 0.2 dropped


def test_threshold_matrix_zero_fraction_keeps_everything():
    m = np.array([[0.0, 0.2, 0.9], [0.2, 0.0, 0.5], [0.9, 0.5, 0.0]])
    assert np.array_equal(threshold_matrix(m, 0.0), m)


def test_decoding_returns_accuracy_over_time():
    epochs = EventEpochs().apply(eeg_sample(), {"tmin": -0.1, "tmax": 0.4}).payload
    times, scores = decoding(epochs)
    assert scores.ndim == 1
    assert len(scores) == len(times)


def test_decode_classifiers_offer_logreg_lda_svm():
    assert set(DECODE_CLASSIFIERS.values()) == {"logreg", "lda", "svm"}


def test_decoding_supports_choosing_the_classifier():
    epochs = EventEpochs().apply(eeg_sample(), {"tmin": -0.1, "tmax": 0.4}).payload
    times, scores = decoding(epochs, classifier="lda")
    assert len(scores) == len(times)


def test_temporal_generalization_returns_a_square_matrix():
    epochs = EventEpochs().apply(eeg_sample(), {"tmin": -0.1, "tmax": 0.4}).payload
    times, matrix = temporal_generalization(epochs)
    assert matrix.shape == (len(times), len(times))


def test_decoding_csp_returns_a_scalar_accuracy():
    epochs = EventEpochs().apply(eeg_sample(), {"tmin": -0.1, "tmax": 0.4}).payload
    acc = decoding_csp(epochs)
    assert 0.0 <= acc <= 1.0


def test_cluster_test_returns_condition_means_and_mask():
    epochs = EventEpochs().apply(eeg_sample(), {"tmin": -0.1, "tmax": 0.4}).payload
    times, mean_a, mean_b, sig, labels = cluster_test(epochs, n_permutations=100)
    assert len(mean_a) == len(mean_b) == len(sig) == len(times)
    assert len(labels) == 2


def test_regression_erp_returns_channel_by_time_maps():
    epochs = EventEpochs().apply(eeg_sample(), {"tmin": -0.1, "tmax": 0.4}).payload
    times, names, beta, tval = regression_erp(epochs)
    assert len(names) == 10
    assert beta.shape == (10, len(times))
    assert tval.shape == (10, len(times))


def test_regression_erp_accepts_a_continuous_predictor():
    epochs = EventEpochs().apply(eeg_sample(), {"tmin": -0.1, "tmax": 0.4}).payload
    predictor = np.arange(len(epochs), dtype=float)
    times, names, beta, tval = regression_erp(epochs, predictor=predictor)
    assert beta.shape == (10, len(times))


def test_glm_analysis_returns_per_channel_condition_betas():
    haemo = BeerLambert().apply(
        OpticalDensity().apply(fnirs_sample(), {}), {"ppf": 6.0}
    ).payload
    table = glm_analysis(haemo, stim_dur=4.0)
    assert "theta" in table.columns and "Condition" in table.columns
    conditions = glm_conditions(table)
    assert "Task" in conditions and "Control" in conditions
    assert "constant" not in conditions  # design regressors are excluded


def test_glm_contrast_returns_per_channel_effects():
    haemo = BeerLambert().apply(
        OpticalDensity().apply(fnirs_sample(), {}), {"ppf": 6.0}
    ).payload
    table = glm_contrast(haemo, "Task", "Control", stim_dur=4.0)
    assert "effect" in table.columns
    assert (table["Chroma"] == "hbo").any()


def test_spatiotemporal_cluster_returns_channel_by_time_maps():
    epochs = EventEpochs().apply(eeg_sample(), {"tmin": -0.1, "tmax": 0.4}).payload
    times, names, stat, significant = spatiotemporal_cluster_test(epochs, n_permutations=100)
    assert len(names) == 10
    assert stat.shape == (10, len(times))
    assert significant.shape == (10, len(times))
    assert significant.dtype == bool


def test_condition_comparison_returns_means_and_difference():
    epochs = EventEpochs().apply(eeg_sample(), {"tmin": -0.1, "tmax": 0.4}).payload
    times, means, difference = condition_comparison(epochs)
    assert set(means) == {"standard", "target"}
    assert len(means["standard"]) == len(times)
    assert difference is not None and len(difference) == len(times)


@pytest.mark.skipif(not op.exists(_FSAVERAGE), reason="fsaverage template not downloaded")
def test_source_localization_returns_region_activity():
    evoked = Average().apply(FixedLengthEpochs().apply(eeg_sample(), {"duration": 1.0}), {}).payload
    names, strengths = source_localization(evoked, n_regions=8)
    assert len(names) == 8
    assert len(strengths) == 8
