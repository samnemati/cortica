"""Tests for the ICA component-inspector dialog."""
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")
mne = pytest.importorskip("mne")

from cortica.gui.ica_dialog import ICADialog  # noqa: E402
from cortica.samples import eeg_sample  # noqa: E402


def test_ica_dialog_lists_components_and_reports_exclusions(qtbot):
    dlg = ICADialog(eeg_sample().payload, n_components=4)
    qtbot.addWidget(dlg)
    assert len(dlg._checkboxes) == dlg.ica.n_components_
    dlg._checkboxes[1].setChecked(True)
    assert dlg.excluded() == [1]
