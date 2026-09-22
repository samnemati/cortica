"""Group-level (across-subjects) brain-behavior analysis (Qt-free).

Subjects are assembled from recording files (one subject per file); a behavioral
table is joined by subject id; then one brain value per subject (e.g. mean band power
at a channel) is correlated/regressed against a behavioral column across subjects.

The pure functions here are unit-tested without a GUI; the group dialog drives them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import viz


def subject_id_from_path(path) -> str:
    """Derive a subject id from a recording filename (dropping a trailing raw tag)."""
    stem = Path(path).stem
    for suffix in ("_raw", "-raw", "_eeg", "-eeg"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def subject_band_power(path, band, channel, loader=None) -> float:
    """Load a recording and return mean ``band`` power at ``channel`` (a scalar)."""
    from .io import load_raw

    dataset = (loader or load_raw)(path)
    return float(viz.band_power(dataset.payload, band, picks=[channel])[0])


def subject_connectivity(path, method, band, ch_a, ch_b, duration=2.0, loader=None) -> float:
    """Load a recording, epoch it into fixed-length windows, and return the ``method``
    connectivity in ``band`` between ``ch_a`` and ``ch_b`` (a scalar).
    """
    from .io import load_raw
    from .steps.epoch import FixedLengthEpochs

    dataset = (loader or load_raw)(path)
    epochs = FixedLengthEpochs().apply(dataset, {"duration": duration})
    matrix, names = viz.connectivity(epochs.payload, method=method, band=band)
    i, j = names.index(ch_a), names.index(ch_b)
    return float(matrix[i, j])


def subject_glm_beta(path, condition, channel, chroma="hbo", stim_dur=5.0, loader=None) -> float:
    """Load an fNIRS recording, run the first-level GLM, and return the beta (theta)
    for ``condition`` at ``channel`` (an S-D pair like ``S1_D1``) and ``chroma``.
    """
    from mne.preprocessing.nirs import beer_lambert_law, optical_density

    from .io import load_raw

    dataset = (loader or load_raw)(path)
    haemo = beer_lambert_law(optical_density(dataset.payload), ppf=6.0)
    table = viz.glm_analysis(haemo, stim_dur=stim_dur)
    name = f"{channel} {chroma}"
    match = table[(table["Condition"] == condition) & (table["ch_name"] == name)]
    if match.empty:
        raise ValueError(f"No GLM estimate for {name} in condition {condition}.")
    return float(match["theta"].iloc[0])


def align_behavior(subject_ids, table, id_column, value_column):
    """Match a behavioral column to ``subject_ids`` by id.

    Returns ``(kept_ids, values)`` for the subjects that have a numeric value,
    preserving the order of ``subject_ids``.
    """
    import pandas as pd

    lookup = {
        str(sid): val
        for sid, val in zip(table[id_column].astype(str), table[value_column])
    }
    kept, values = [], []
    for sid in subject_ids:
        value = lookup.get(str(sid))
        if value is not None and pd.notna(value):
            kept.append(sid)
            values.append(float(value))
    return kept, np.asarray(values, dtype=float)


def group_correlation(feature_values, behavior_values, method="pearson"):
    """Return ``(r, p)`` correlating one brain feature with behavior across subjects."""
    from scipy import stats

    func = stats.spearmanr if method == "spearman" else stats.pearsonr
    result = func(
        np.asarray(feature_values, dtype=float), np.asarray(behavior_values, dtype=float)
    )
    return float(result[0]), float(result[1])


def linear_fit(x, y):
    """Least-squares line through ``(x, y)``. Returns ``(slope, intercept, r, p)``."""
    from scipy import stats

    result = stats.linregress(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return (
        float(result.slope),
        float(result.intercept),
        float(result.rvalue),
        float(result.pvalue),
    )
