"""Smoke tests for the group-analysis dialog (offscreen)."""
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")
mne = pytest.importorskip("mne")

from cortica.gui.group_dialog import GroupDialog  # noqa: E402
from cortica.samples import eeg_sample  # noqa: E402


def _save_subjects(tmp_path, n):
    paths = []
    for i in range(n):
        path = tmp_path / f"S{i:02d}_raw.fif"
        eeg_sample().payload.save(str(path), overwrite=True, verbose=False)
        paths.append(str(path))
    return paths


def test_group_dialog_aligns_subjects_and_computes(qtbot, tmp_path):
    import pandas as pd

    paths = _save_subjects(tmp_path, 4)
    dialog = GroupDialog(channels=["O1", "O2"])
    qtbot.addWidget(dialog)
    dialog._add_paths(paths)
    assert dialog.table.rowCount() == 4
    dialog._set_behavior_table(
        pd.DataFrame({"subject": [f"S{i:02d}" for i in range(4)], "score": [1.0, 2.0, 3.0, 4.0]})
    )
    dialog.id_column.setCurrentText("subject")
    dialog.value_column.setCurrentText("score")
    dialog.channel.setCurrentText("O1")
    dialog._compute()
    assert len(dialog._result_rows) == 4  # all four subjects matched their behavior row


def test_group_dialog_draw_reports_a_correlation(qtbot):
    dialog = GroupDialog(channels=["O1"])
    qtbot.addWidget(dialog)
    dialog.value_column.addItem("score")
    dialog.value_column.setCurrentText("score")
    dialog._draw([1.0, 2.0, 3.0, 4.0], [2.0, 4.1, 5.9, 8.0], "Alpha", "O1", "pearson")
    assert "r=" in dialog.result_label.text()


def test_group_dialog_guides_when_empty(qtbot):
    dialog = GroupDialog()
    qtbot.addWidget(dialog)
    dialog._compute()  # nothing added yet, must not raise
    assert "Add recordings" in dialog.result_label.text()
