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
