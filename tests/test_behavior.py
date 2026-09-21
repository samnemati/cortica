"""Tests for behavioral-data import and brain-behavior statistics (Qt-free)."""
import numpy as np
import pytest

mne = pytest.importorskip("mne")

from cortica.behavior import (  # noqa: E402
    correlate_with_behavior,
    multiple_regression_r2,
    numeric_columns,
    read_table,
    single_trial_band_power,
)
from cortica.samples import eeg_sample  # noqa: E402
from cortica.steps.epoch import EventEpochs  # noqa: E402


def test_read_table_reads_csv_and_finds_numeric_columns(tmp_path):
    path = tmp_path / "behavior.csv"
    path.write_text("trial,rt_ms,label\n1,412,a\n2,528,b\n3,389,a\n")
    table = read_table(str(path))
    assert list(table.columns) == ["trial", "rt_ms", "label"]
    numeric = numeric_columns(table)
    assert "rt_ms" in numeric
    assert "label" not in numeric  # text column is not a behavioral score


def test_single_trial_band_power_is_trials_by_channels():
    epochs = EventEpochs().apply(eeg_sample(), {"tmin": -0.1, "tmax": 0.4}).payload
    power = single_trial_band_power(epochs, "Alpha")
    assert power.shape == (len(epochs), len(epochs.ch_names))


def test_correlate_with_behavior_finds_the_related_feature():
    rng = np.random.RandomState(0)
    features = rng.randn(40, 5)
    behavior = features[:, 2] * 2.0 + rng.randn(40) * 0.1  # feature 2 drives behavior
    r, p = correlate_with_behavior(features, behavior)
    assert r.shape == (5,) and p.shape == (5,)
    assert int(np.argmax(np.abs(r))) == 2
    assert p[2] < 0.001


def test_multiple_regression_recovers_a_linear_combination():
    rng = np.random.RandomState(1)
    features = rng.randn(50, 3)
    behavior = features @ np.array([1.0, -2.0, 0.5])
    assert multiple_regression_r2(features, behavior) > 0.99
