"""Tests for the HTML report — the shareable, reproducible artifact.

This first version needs no matplotlib: it records the pipeline (ordered steps +
params), a dataset summary, and a provenance block. Rich figures come later.
"""
import numpy as np
import pytest

mne = pytest.importorskip("mne")

from cortica.core.pipeline import Pipeline  # noqa: E402
from cortica.io import dataset_from_raw  # noqa: E402
from cortica.report import build_report  # noqa: E402


def _dataset():
    info = mne.create_info(["EEG 001", "EEG 002"], 200.0, ch_types="eeg")
    raw = mne.io.RawArray(np.zeros((2, 400)), info, verbose=False)
    return dataset_from_raw(raw)


def test_build_report_writes_html_with_steps_and_provenance(tmp_path):
    pipe = (
        Pipeline("eeg")
        .add("bandpass_filter", {"l_freq": 1.0, "h_freq": 20.0})
        .add("resample", {"sfreq": 100.0})
    )
    out = tmp_path / "report.html"

    build_report(_dataset(), pipe, str(out))

    assert out.exists()
    html = out.read_text()
    assert "Band-pass filter" in html      # resolved display name, not just the id
    assert "Resample" in html
    assert "1.0" in html and "20.0" in html  # params are recorded
    assert "eeg" in html.lower()             # dataset summary
    assert "cortica" in html.lower()         # provenance / version


def test_build_report_escapes_html_in_values(tmp_path):
    pipe = Pipeline("eeg").add("bandpass_filter", {"l_freq": "<script>", "h_freq": 20.0})
    out = tmp_path / "report.html"
    build_report(_dataset(), pipe, str(out))
    html = out.read_text()
    assert "<script>" not in html            # the literal tag must be escaped
    assert "&lt;script&gt;" in html
