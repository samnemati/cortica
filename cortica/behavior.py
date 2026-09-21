"""Import behavioral data and relate it to brain features (Qt-free).

Within a single recording this reads a per-trial table (CSV/TSV/Excel), computes
single-trial band power per channel from Epochs, and correlates or regresses each
channel's single-trial power against a chosen behavioral column across trials.

The pure functions here are unit-tested without a GUI and are reused by the group
layer (which supplies one row per subject instead of one row per trial).
"""
from __future__ import annotations

import numpy as np

from . import viz


def read_table(path):
    """Read a behavioral table from CSV/TSV/Excel into a pandas DataFrame."""
    import pandas as pd

    lower = path.lower()
    if lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    sep = "\t" if lower.endswith((".tsv", ".txt")) else ","
    return pd.read_csv(path, sep=sep)


def numeric_columns(table) -> list[str]:
    """Return the names of numeric columns (candidate behavioral scores)."""
    import pandas as pd

    return [c for c in table.columns if pd.api.types.is_numeric_dtype(table[c])]


def single_trial_band_power(epochs, band, picks=None):
    """Return ``(n_trials, n_channels)`` mean power in ``band`` for each epoch."""
    fmin, fmax = viz.BANDS[band]
    kwargs = {"fmin": fmin, "fmax": fmax, "verbose": False}
    if picks is not None:
        kwargs["picks"] = picks
    psd = epochs.compute_psd(**kwargs)
    data = np.asarray(psd.get_data())  # (n_epochs, n_channels, n_freqs)
    return data.mean(axis=2)


def correlate_with_behavior(feature_matrix, behavior, method="pearson"):
    """Correlate each column of ``feature_matrix`` (n_obs, n_features) with ``behavior``.

    Returns ``(r, p)`` arrays of length ``n_features``. ``method`` is ``"pearson"`` or
    ``"spearman"``.
    """
    from scipy import stats

    features = np.asarray(feature_matrix, dtype=float)
    behavior = np.asarray(behavior, dtype=float)
    func = stats.spearmanr if method == "spearman" else stats.pearsonr
    n_features = features.shape[1]
    r = np.zeros(n_features)
    p = np.zeros(n_features)
    for i in range(n_features):
        result = func(features[:, i], behavior)
        r[i], p[i] = float(result[0]), float(result[1])
    return r, p


def multiple_regression_r2(feature_matrix, behavior) -> float:
    """R-squared of predicting ``behavior`` from all features (linear regression)."""
    from sklearn.linear_model import LinearRegression

    features = np.asarray(feature_matrix, dtype=float)
    behavior = np.asarray(behavior, dtype=float)
    model = LinearRegression().fit(features, behavior)
    return float(model.score(features, behavior))
