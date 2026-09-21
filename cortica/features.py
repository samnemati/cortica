"""Extract tabular brain-feature values (Qt-free) for CSV export and statistics.

Two tables underpin both the CSV export and the brain-behavior stats:

* band power per channel and frequency band (from :func:`cortica.viz.band_power`)
* connectivity per channel pair (from :func:`cortica.viz.connectivity`)

Keeping these as plain arrays/rows here means they are unit-tested without a GUI and
reused by the export action and (later) the correlation/regression features.
"""
from __future__ import annotations

import csv

import numpy as np

from . import viz


def band_power_table(payload, picks=None):
    """Return ``(channel_names, band_names, matrix)`` of mean band power.

    ``matrix`` is shaped ``(n_channels, n_bands)`` over the five standard bands.
    """
    bands = list(viz.BANDS)
    columns = [viz.band_power(payload, band, picks=picks) for band in bands]
    matrix = np.asarray(columns).T  # (n_channels, n_bands)
    names = list(payload.ch_names) if picks is None else list(picks)
    return names, bands, matrix


def connectivity_table(epochs, method="plv", band="Alpha"):
    """Return ``[(channel_a, channel_b, value), ...]`` for the upper-triangle pairs."""
    matrix, names = viz.connectivity(epochs, method=method, band=band)
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            rows.append((names[i], names[j], float(matrix[i, j])))
    return rows


def write_band_power_csv(payload, path, picks=None) -> None:
    """Write a channel-by-band power table to ``path`` as CSV."""
    names, bands, matrix = band_power_table(payload, picks=picks)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["channel", *bands])
        for name, row in zip(names, matrix):
            writer.writerow([name, *(f"{value:.6g}" for value in row)])


def write_connectivity_csv(epochs, path, method="plv", band="Alpha") -> None:
    """Write a per-pair connectivity table to ``path`` as CSV."""
    rows = connectivity_table(epochs, method=method, band=band)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["channel_a", "channel_b", f"{method}_{band}"])
        for channel_a, channel_b, value in rows:
            writer.writerow([channel_a, channel_b, f"{value:.6g}"])
