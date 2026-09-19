"""ICA component inspector.

Fits ICA on the given data (fixed random_state, matching the ICA step), shows each
component's scalp map, and lets the user tick which components to remove. The chosen
indices are read back by the main window and stored on an ICA pipeline step.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
)


class ICADialog(QDialog):
    def __init__(self, payload, n_components: int = 15, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fit ICA — choose components to remove")

        import mne
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        self.n_components = min(n_components, len(payload.ch_names))
        self.ica = mne.preprocessing.ICA(
            n_components=self.n_components, random_state=97, max_iter="auto", verbose=False
        )
        self.ica.fit(payload.copy())

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel("Tick the components that look like artifacts, then Add to pipeline.")
        )
        figures = self.ica.plot_components(show=False)
        for figure in figures if isinstance(figures, list) else [figures]:
            layout.addWidget(FigureCanvasQTAgg(figure))

        self._checkboxes = []
        row = QHBoxLayout()
        for i in range(self.ica.n_components_):
            checkbox = QCheckBox(f"ICA{i:03d}")
            self._checkboxes.append(checkbox)
            row.addWidget(checkbox)
        row.addStretch(1)
        layout.addLayout(row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Add to pipeline")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def excluded(self) -> list[int]:
        return [i for i, checkbox in enumerate(self._checkboxes) if checkbox.isChecked()]
