"""Tests for the HTML report — the shareable, reproducible artifact.

This first version needs no matplotlib: it records the pipeline (ordered steps +
params), a dataset summary, and a provenance block. Rich figures come later.
"""
import numpy as np
import pytest

mne = pytest.importorskip("mne")

import datetime  # noqa: E402

from cortica.core.pipeline import Pipeline  # noqa: E402
from cortica.io import dataset_from_raw  # noqa: E402
from cortica.report import build_report, default_report_name  # noqa: E402


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


def test_build_report_embeds_a_figure(tmp_path):
    pipe = Pipeline("eeg").add("bandpass_filter", {"l_freq": 1.0, "h_freq": 20.0})
    out = tmp_path / "report.html"
    build_report(_dataset(), pipe, str(out))
    assert "data:image/png;base64," in out.read_text()  # an embedded figure


def test_build_report_shows_evoked_with_gfp(tmp_path):
    import numpy as np

    from cortica.io import dataset_from_raw
    from cortica.steps.epoch import Average, FixedLengthEpochs

    info = mne.create_info(["EEG 1", "EEG 2", "EEG 3"], 200.0, ch_types="eeg")
    raw = mne.io.RawArray(np.random.RandomState(0).randn(3, 2000) * 1e-6, info, verbose=False)
    epochs = FixedLengthEpochs().apply(dataset_from_raw(raw), {"duration": 0.5})
    evoked = Average().apply(epochs, {})
    out = tmp_path / "report.html"
    build_report(evoked, Pipeline("eeg"), str(out))
    assert "Evoked response" in out.read_text()


def test_default_report_name_summarizes_steps_and_stamps_time():
    pipe = Pipeline("eeg").add("bandpass_filter").add("average")
    name = default_report_name(pipe, now=datetime.datetime(2026, 9, 21, 15, 30, 0))
    assert name == "cortica-eeg-bp-avg-20260921-153000.html"


def test_default_report_name_includes_a_user_label():
    pipe = Pipeline("eeg").add("average")
    name = default_report_name(
        pipe, label="Subject 03 oddball", now=datetime.datetime(2026, 9, 21, 9, 5, 1)
    )
    assert name == "cortica-subject-03-oddball-eeg-avg-20260921-090501.html"


def test_default_report_names_differ_by_time_so_reports_never_clobber():
    pipe = Pipeline("eeg").add("average")
    a = default_report_name(pipe, now=datetime.datetime(2026, 9, 21, 15, 30, 0))
    b = default_report_name(pipe, now=datetime.datetime(2026, 9, 21, 15, 30, 1))
    assert a != b


def test_build_report_escapes_html_in_values(tmp_path):
    pipe = Pipeline("eeg").add("bandpass_filter", {"l_freq": "<script>", "h_freq": 20.0})
    out = tmp_path / "report.html"
    build_report(_dataset(), pipe, str(out))
    html = out.read_text()
    assert "<script>" not in html            # the literal tag must be escaped
    assert "&lt;script&gt;" in html
