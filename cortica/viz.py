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
CONNECTIVITY_METHODS = {
    "PLV": "plv",
    "Coherence": "coh",
    "wPLI": "wpli",
    "Imag. coherence": "imcoh",
    "PLI": "pli",
    "ciPLV": "ciplv",
}


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


def threshold_matrix(matrix, drop_fraction):
    """Proportional (density) threshold: keep the strongest edges and zero out the
    weakest ``drop_fraction`` of off-diagonal edges (ranked by absolute value).

    ``drop_fraction`` in ``[0, 1)`` — ``0`` keeps every edge, ``0.9`` keeps only the
    strongest 10%. Returns a symmetric copy. This ranks edges rather than comparing
    to the peak, so it always thins the graph regardless of the absolute scale.
    """
    m = np.array(matrix, dtype=float)
    if drop_fraction <= 0.0 or m.shape[0] < 2:
        return m
    iu = np.triu_indices(m.shape[0], 1)
    weights = np.abs(m[iu])
    keep = max(1, int(round((1.0 - drop_fraction) * weights.size)))
    if keep >= weights.size:
        return m
    cutoff = np.sort(weights)[::-1][keep - 1]  # magnitude of the weakest kept edge
    mask = np.abs(m) < cutoff
    np.fill_diagonal(mask, False)
    m[mask] = 0.0
    return m


#: Decoding classifiers (display label -> key).
DECODE_CLASSIFIERS = {"Logistic regression": "logreg", "LDA": "lda", "SVM": "svm"}


def _base_classifier(name):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    if name == "lda":
        return LinearDiscriminantAnalysis()
    if name == "svm":
        return SVC()
    return LogisticRegression(max_iter=1000)


def _decoding_labels_cv(epochs, max_cv):
    labels = epochs.events[:, 2]
    _, counts = np.unique(labels, return_counts=True)
    if len(counts) < 2:
        raise ValueError("Decoding needs at least two conditions.")
    cv = int(min(max_cv, counts.min()))
    if cv < 2:
        raise ValueError("Not enough trials per condition to cross-validate.")
    return labels, cv


def decoding(epochs, classifier="logreg", max_cv=5):
    """Temporal decoding: cross-validated accuracy at every time point.
    Returns ``(times, scores)``. Needs Epochs with at least two conditions.
    """
    from mne.decoding import SlidingEstimator, cross_val_multiscore
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    labels, cv = _decoding_labels_cv(epochs, max_cv)
    clf = make_pipeline(StandardScaler(), _base_classifier(classifier))
    estimator = SlidingEstimator(clf, scoring="accuracy", verbose=False)
    scores = cross_val_multiscore(estimator, epochs.get_data(), labels, cv=cv, verbose=False)
    return np.asarray(epochs.times), np.asarray(scores).mean(axis=0)


def temporal_generalization(epochs, classifier="logreg", max_cv=5):
    """Temporal generalization: train at each time, test at every other time.
    Returns ``(times, matrix)`` with ``matrix`` shaped ``(n_times, n_times)``.
    """
    from mne.decoding import GeneralizingEstimator, cross_val_multiscore
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    labels, cv = _decoding_labels_cv(epochs, max_cv)
    clf = make_pipeline(StandardScaler(), _base_classifier(classifier))
    estimator = GeneralizingEstimator(clf, scoring="accuracy", verbose=False)
    scores = cross_val_multiscore(estimator, epochs.get_data(), labels, cv=cv, verbose=False)
    return np.asarray(epochs.times), np.asarray(scores).mean(axis=0)


def decoding_csp(epochs, classifier="logreg", n_components=4, max_cv=5):
    """CSP + classifier whole-epoch decoding. Returns overall CV accuracy (float)."""
    from mne.decoding import CSP, cross_val_multiscore
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    labels, cv = _decoding_labels_cv(epochs, max_cv)
    n_comp = min(n_components, epochs.get_data().shape[1])
    pipe = make_pipeline(CSP(n_components=n_comp), StandardScaler(), _base_classifier(classifier))
    scores = cross_val_multiscore(pipe, epochs.get_data(), labels, cv=cv, verbose=False)
    return float(np.asarray(scores).mean())


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


def condition_comparison(epochs):
    """Per-condition average time course (channels averaged) for every condition in
    ``epochs``, plus the difference wave when there are exactly two. Returns
    ``(times, {label: mean_tc}, difference_or_None)``.
    """
    labels = list(epochs.event_id)
    if len(labels) < 2:
        raise ValueError("Condition comparison needs at least two conditions.")
    means = {label: epochs[label].average().get_data().mean(axis=0) for label in labels}
    difference = means[labels[0]] - means[labels[1]] if len(labels) == 2 else None
    return np.asarray(epochs.times), means, difference


def source_localization(evoked, n_regions=12):
    """Estimate cortical sources on the fsaverage template (dSPM) and return the
    most active anatomical regions as ``(region_names, strengths)``.

    ``evoked`` must be an Evoked. Downloads the fsaverage template on first use.
    """
    import os.path as op

    import mne
    from mne.datasets import fetch_fsaverage

    fs_dir = fetch_fsaverage(verbose=False)
    subjects_dir = op.dirname(fs_dir)
    src_path = op.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
    bem_path = op.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")

    evoked = evoked.copy()
    evoked.set_eeg_reference("average", projection=True, verbose=False)
    fwd = mne.make_forward_solution(
        evoked.info, trans="fsaverage", src=src_path, bem=bem_path,
        eeg=True, mindist=5.0, verbose=False,
    )
    inv = mne.minimum_norm.make_inverse_operator(
        evoked.info, fwd, mne.make_ad_hoc_cov(evoked.info), verbose=False
    )
    stc = mne.minimum_norm.apply_inverse(evoked, inv, lambda2=1 / 9, method="dSPM", verbose=False)

    labels = [
        label
        for label in mne.read_labels_from_annot(
            "fsaverage", "aparc", subjects_dir=subjects_dir, verbose=False
        )
        if "unknown" not in label.name
    ]
    label_tc = mne.extract_label_time_course(
        stc, labels, inv["src"], mode="mean_flip", allow_empty=True, verbose=False
    )
    strength = np.abs(np.asarray(label_tc)).mean(axis=1)
    order = np.argsort(strength)[::-1][:n_regions]
    return [labels[i].name for i in order], strength[order]
