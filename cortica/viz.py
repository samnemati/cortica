"""Pure data-prep for the signal views (no Qt).

Turns whatever MNE object a Dataset carries (Raw, Epochs, Evoked) into plain arrays
the plot widgets can draw, so the widgets stay thin and this is testable in CI.
"""
from __future__ import annotations

import numpy as np


def traces(payload, picks=None):
    """Return ``(times, data)`` with data shaped ``(n_channels, n_times)``.

    ``picks`` (channel names) subsets channels; epochs (3-D) are averaged.
    """
    data = np.asarray(payload.get_data(picks=picks))
    if data.ndim == 3:  # epochs (n_epochs, n_channels, n_times)
        data = data.mean(axis=0)
    times = np.asarray(getattr(payload, "times", np.arange(data.shape[1])))
    return times, data


def spectrum(payload, fmax=None, picks=None):
    """Return ``(freqs, psds)`` with psds shaped ``(n_channels, n_freqs)``.

    Uses MNE's PSD; ``picks`` subsets channels; epochs are averaged.
    """
    kwargs = {"verbose": False}
    if fmax is not None:
        kwargs["fmax"] = fmax
    if picks is not None:
        kwargs["picks"] = picks
    psds, freqs = payload.compute_psd(**kwargs).get_data(return_freqs=True)
    psds = np.asarray(psds)
    if psds.ndim == 3:  # epochs (n_epochs, n_channels, n_freqs)
        psds = psds.mean(axis=0)
    return np.asarray(freqs), psds


#: Standard EEG frequency bands (Hz).
BANDS = {
    "Delta": (1.0, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 12.0),
    "Beta": (12.0, 30.0),
    "Gamma": (30.0, 45.0),
}


def band_power(payload, band, picks=None):
    """Return mean power in a named band (from :data:`BANDS`) per channel."""
    fmin, fmax = BANDS[band]
    freqs, psds = spectrum(payload, fmax=fmax + 5.0, picks=picks)
    mask = (freqs >= fmin) & (freqs <= fmax)
    return psds[:, mask].mean(axis=1)


def time_frequency(payload, picks=None, fmax=40.0, method="morlet"):
    """Return ``(times, freqs, power)`` — a time-frequency map averaged across the
    selected channels; ``power`` is shaped ``(n_freqs, n_times)``.

    ``method`` is ``"morlet"`` or ``"multitaper"``.
    """
    freqs = np.arange(2.0, fmax, 1.0)
    kwargs = {"method": method, "freqs": freqs, "n_cycles": freqs / 2.0, "verbose": False}
    if picks is not None:
        kwargs["picks"] = picks
    tfr = payload.compute_tfr(**kwargs)
    power = np.asarray(tfr.data)
    if power.ndim == 4:  # epochs (n_epochs, n_channels, n_freqs, n_times)
        power = power.mean(axis=0)
    power = power.mean(axis=0)  # average across channels -> (n_freqs, n_times)
    return np.asarray(tfr.times), np.asarray(tfr.freqs), power


#: Connectivity measures (display label -> mne-connectivity method name).
CONNECTIVITY_METHODS = {"PLV": "plv", "Coherence": "coh", "wPLI": "wpli"}


def connectivity(payload, method="plv", band="Alpha"):
    """Return ``(matrix, ch_names)`` — a symmetric channel-by-channel connectivity
    matrix for a frequency band. ``payload`` must be Epochs.
    """
    from mne_connectivity import spectral_connectivity_epochs

    fmin, fmax = BANDS[band]
    con = spectral_connectivity_epochs(
        payload, method=method, mode="multitaper", fmin=fmin, fmax=fmax,
        faverage=True, verbose=False,
    )
    matrix = np.asarray(con.get_data(output="dense"))
    if matrix.ndim == 3:
        matrix = matrix[:, :, 0]
    matrix = matrix + matrix.T  # returned lower-triangular; make it symmetric
    return matrix, list(payload.ch_names)


def decoding(epochs, max_cv=5):
    """Temporal decoding: cross-validated accuracy of classifying each epoch's
    condition at every time point. Returns ``(times, scores)``. Needs Epochs with
    at least two conditions.
    """
    from mne.decoding import SlidingEstimator, cross_val_multiscore
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    labels = epochs.events[:, 2]
    _, counts = np.unique(labels, return_counts=True)
    if len(counts) < 2:
        raise ValueError("Decoding needs at least two conditions.")
    cv = int(min(max_cv, counts.min()))
    if cv < 2:
        raise ValueError("Not enough trials per condition to cross-validate.")
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    estimator = SlidingEstimator(clf, scoring="accuracy", verbose=False)
    scores = cross_val_multiscore(estimator, epochs.get_data(), labels, cv=cv, verbose=False)
    return np.asarray(epochs.times), np.asarray(scores).mean(axis=0)


def cluster_test(epochs, n_permutations=200):
    """Channel-averaged cluster-based permutation test between the two conditions
    in ``epochs``. Returns ``(times, mean_a, mean_b, significant_mask, labels)``.
    """
    from mne.stats import permutation_cluster_test

    labels = list(epochs.event_id)
    if len(labels) < 2:
        raise ValueError("Statistics need two conditions.")
    a = epochs[labels[0]].get_data().mean(axis=1)  # (n_epochs, n_times), channels averaged
    b = epochs[labels[1]].get_data().mean(axis=1)
    _, clusters, p_values, _ = permutation_cluster_test(
        [a, b], n_permutations=n_permutations, seed=97, out_type="mask", verbose=False
    )
    significant = np.zeros(a.shape[1], dtype=bool)
    for cluster, p in zip(clusters, p_values):
        if p < 0.05:
            significant |= np.asarray(cluster)
    return np.asarray(epochs.times), a.mean(axis=0), b.mean(axis=0), significant, labels
