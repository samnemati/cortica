"""Tests for the headless CLI runner (`cortica run ...`)."""
import numpy as np
import pytest

mne = pytest.importorskip("mne")

from cortica.cli import main  # noqa: E402
from cortica.core.pipeline import Pipeline  # noqa: E402


def _write_raw(path):
    info = mne.create_info(["EEG 001", "EEG 002"], 200.0, ch_types="eeg")
    raw = mne.io.RawArray(np.random.RandomState(0).randn(2, 1600) * 1e-6, info, verbose=False)
    raw.save(path, overwrite=True, verbose=False)


def _write_pipeline(path):
    pipe = (
        Pipeline("eeg")
        .add("bandpass_filter", {"l_freq": 1.0, "h_freq": 20.0})
        .add("resample", {"sfreq": 100.0})
    )
    path.write_text(pipe.to_yaml())


def test_run_command_processes_input_and_writes_output(tmp_path):
    raw_path = tmp_path / "input_raw.fif"
    pipe_path = tmp_path / "analysis.pipeline.yaml"
    out_path = tmp_path / "output_raw.fif"
    _write_raw(raw_path)
    _write_pipeline(pipe_path)

    rc = main(["run", str(pipe_path), str(raw_path), "--out", str(out_path)])

    assert rc == 0
    assert out_path.exists()
    processed = mne.io.read_raw_fif(out_path, verbose=False)
    assert processed.info["sfreq"] == 100.0  # resample step took effect


def test_run_command_writes_report(tmp_path):
    raw_path = tmp_path / "input_raw.fif"
    pipe_path = tmp_path / "analysis.pipeline.yaml"
    report_path = tmp_path / "report.html"
    _write_raw(raw_path)
    _write_pipeline(pipe_path)

    rc = main(["run", str(pipe_path), str(raw_path), "--report", str(report_path)])

    assert rc == 0
    assert report_path.exists()
    assert "Band-pass filter" in report_path.read_text()


def test_no_command_without_gui_prints_install_hint(capsys):
    rc = main([])
    assert rc == 1
    assert "gui" in capsys.readouterr().err.lower()
