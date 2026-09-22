"""Tests for group-level (across-subjects) brain-behavior analysis (Qt-free)."""
import pytest

mne = pytest.importorskip("mne")

from cortica.group import (  # noqa: E402
    align_behavior,
    group_correlation,
    linear_fit,
    subject_band_power,
    subject_connectivity,
    subject_id_from_path,
)
from cortica.samples import eeg_sample  # noqa: E402


def test_subject_id_drops_the_raw_tag():
    assert subject_id_from_path("/data/S01_raw.fif") == "S01"
    assert subject_id_from_path("sub-02-eeg.edf") == "sub-02"
    assert subject_id_from_path("plain.snirf") == "plain"


def test_align_behavior_matches_by_id_and_keeps_order():
    import pandas as pd

    table = pd.DataFrame({"subject": ["S01", "S02", "S03"], "IQ": [100, 110, 120]})
    kept, values = align_behavior(["S02", "S01", "S99"], table, "subject", "IQ")
    assert kept == ["S02", "S01"]  # S99 has no behavior row, dropped
    assert list(values) == [110.0, 100.0]


def test_group_correlation_and_linear_fit_recover_a_relationship():
    x = [1, 2, 3, 4, 5]
    y = [2.1, 3.9, 6.2, 7.8, 10.1]
    r, p = group_correlation(x, y)
    assert r > 0.99 and p < 0.001
    slope, intercept, rr, pp = linear_fit(x, y)
    assert 1.8 < slope < 2.2


def test_subject_band_power_from_a_saved_recording(tmp_path):
    path = tmp_path / "S01_raw.fif"
    eeg_sample().payload.save(str(path), overwrite=True, verbose=False)
    value = subject_band_power(str(path), "Alpha", "O1")
    assert value > 0


def test_subject_connectivity_from_a_saved_recording(tmp_path):
    path = tmp_path / "S01_raw.fif"
    eeg_sample().payload.save(str(path), overwrite=True, verbose=False)
    value = subject_connectivity(str(path), "plv", "Alpha", "O1", "O2", duration=1.0)
    assert 0.0 <= value <= 1.0  # PLV is bounded in [0, 1]
