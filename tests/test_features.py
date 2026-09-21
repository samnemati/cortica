"""Tests for tabular feature extraction and CSV export."""
import pytest

mne = pytest.importorskip("mne")

from cortica.features import (  # noqa: E402
    band_power_table,
    connectivity_table,
    write_band_power_csv,
    write_connectivity_csv,
)
from cortica.samples import eeg_sample  # noqa: E402
from cortica.steps.epoch import FixedLengthEpochs  # noqa: E402


def test_band_power_table_is_channels_by_bands():
    ds = eeg_sample()
    names, bands, matrix = band_power_table(ds.payload)
    assert len(names) == ds.meta["n_channels"]
    assert bands == ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
    assert matrix.shape == (len(names), len(bands))


def test_connectivity_table_lists_every_unique_pair():
    epochs = FixedLengthEpochs().apply(eeg_sample(), {"duration": 1.0}).payload
    rows = connectivity_table(epochs, method="plv", band="Alpha")
    assert len(rows) == 10 * 9 // 2  # 45 unique pairs for 10 channels
    assert all(len(row) == 3 for row in rows)


def test_write_band_power_csv_has_header_and_channels(tmp_path):
    out = tmp_path / "power.csv"
    write_band_power_csv(eeg_sample().payload, str(out))
    text = out.read_text()
    assert text.splitlines()[0] == "channel,Delta,Theta,Alpha,Beta,Gamma"
    assert "O1" in text


def test_write_connectivity_csv_has_header_and_pairs(tmp_path):
    epochs = FixedLengthEpochs().apply(eeg_sample(), {"duration": 1.0}).payload
    out = tmp_path / "conn.csv"
    write_connectivity_csv(epochs, str(out), method="plv", band="Alpha")
    lines = out.read_text().splitlines()
    assert lines[0] == "channel_a,channel_b,plv_Alpha"
    assert len(lines) == 1 + 45  # header + 45 pairs
