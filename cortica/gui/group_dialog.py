"""Group-level brain-behavior analysis dialog.

Add recording files (one subject each), join a behavioral table by subject id, pick a
brain feature (mean band power at a channel), then correlate it against a behavioral
column across subjects as a scatter plot. Export the per-subject table as CSV.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from .. import group, viz


class GroupDialog(QDialog):
    def __init__(self, channels=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Group analysis: brain vs behavior across subjects")
        self.resize(900, 660)
        self._paths: list[str] = []
        self._behavior = None
        self._result_rows: list[tuple] = []

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        add_button = QPushButton("Add recordings…")
        add_button.clicked.connect(self._browse_recordings)
        top.addWidget(add_button)
        behavior_button = QPushButton("Import behavior…")
        behavior_button.clicked.connect(self._browse_behavior)
        top.addWidget(behavior_button)
        top.addWidget(QLabel("Match id:"))
        self.id_column = QComboBox()
        self.id_column.setMinimumWidth(110)
        top.addWidget(self.id_column)
        top.addWidget(QLabel("Behavior:"))
        self.value_column = QComboBox()
        self.value_column.setMinimumWidth(110)
        top.addWidget(self.value_column)
        top.addStretch(1)
        layout.addLayout(top)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["subject", "file"])
        self.table.setColumnWidth(0, 140)
        self.table.setColumnWidth(1, 620)
        layout.addWidget(self.table)

        feature_row = QHBoxLayout()
        feature_row.addWidget(QLabel("Feature: mean"))
        self.band = QComboBox()
        self.band.addItems(list(viz.BANDS))
        self.band.setCurrentText("Alpha")
        feature_row.addWidget(self.band)
        feature_row.addWidget(QLabel("power at"))
        self.channel = QComboBox()
        self.channel.setMinimumWidth(90)
        if channels:
            self.channel.addItems(list(channels))
        feature_row.addWidget(self.channel)
        feature_row.addWidget(QLabel("Method:"))
        self.method = QComboBox()
        self.method.addItems(["Pearson", "Spearman"])
        feature_row.addWidget(self.method)
        compute_button = QPushButton("Compute & correlate")
        compute_button.setObjectName("runButton")
        compute_button.clicked.connect(self._compute)
        feature_row.addWidget(compute_button)
        feature_row.addStretch(1)
        layout.addLayout(feature_row)

        self._fig = Figure(figsize=(5, 3))
        self._canvas = FigureCanvasQTAgg(self._fig)
        layout.addWidget(self._canvas)

        self.result_label = QLabel("Add recordings and a behavior file, then Compute.")
        layout.addWidget(self.result_label)
        export_button = QPushButton("Export table (CSV)…")
        export_button.clicked.connect(self._export)
        layout.addWidget(export_button)

    # ---- assembling the group ----------------------------------------------
    def _browse_recordings(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add recordings", "",
            "Recordings (*.fif *.edf *.bdf *.vhdr *.set *.snirf);;All files (*)",
        )
        if paths:
            self._add_paths(paths)

    def _add_paths(self, paths) -> None:
        for path in paths:
            if path not in self._paths:
                self._paths.append(path)
        self.table.setRowCount(len(self._paths))
        for i, path in enumerate(self._paths):
            self.table.setItem(i, 0, QTableWidgetItem(group.subject_id_from_path(path)))
            self.table.setItem(i, 1, QTableWidgetItem(path))
        if self.channel.count() == 0 and self._paths:
            try:
                from ..io import load_raw

                self.channel.addItems(list(load_raw(self._paths[0]).payload.ch_names))
            except Exception:
                pass

    def _browse_behavior(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import behavior", "",
            "Tables (*.csv *.tsv *.txt *.xlsx *.xls);;All files (*)",
        )
        if not path:
            return
        from .. import behavior as beh

        try:
            table = beh.read_table(path)
        except Exception as exc:
            self.result_label.setText(f"Could not read behavior file: {exc}")
            return
        self._set_behavior_table(table)

    def _set_behavior_table(self, table) -> None:
        from .. import behavior as beh

        self._behavior = table
        self.id_column.clear()
        self.id_column.addItems([str(c) for c in table.columns])
        self.value_column.clear()
        self.value_column.addItems(beh.numeric_columns(table))
        self.result_label.setText(f"Behavior loaded: {len(table)} rows.")

    # ---- compute ------------------------------------------------------------
    def _compute(self) -> None:
        if not self._paths or self._behavior is None:
            self.result_label.setText("Add recordings and import a behavior file first.")
            return
        channel = self.channel.currentText()
        if not channel:
            self.result_label.setText("Pick a channel for the feature.")
            return
        band = self.band.currentText()
        method = "spearman" if self.method.currentText() == "Spearman" else "pearson"
        ids, feature_lookup = [], {}
        for path in self._paths:
            try:
                value = group.subject_band_power(path, band, channel)
            except Exception:
                continue
            sid = group.subject_id_from_path(path)
            ids.append(sid)
            feature_lookup[sid] = value
        kept_ids, behavior_values = group.align_behavior(
            ids, self._behavior, self.id_column.currentText(), self.value_column.currentText()
        )
        feature_values = [feature_lookup[sid] for sid in kept_ids]
        self._result_rows = list(zip(kept_ids, feature_values, list(behavior_values)))
        self._draw(feature_values, list(behavior_values), band, channel, method)

    def _draw(self, features, behavior, band, channel, method) -> None:
        import numpy as np

        self._fig.clear()
        ax = self._fig.add_subplot(111)
        if len(features) < 3:
            ax.set_axis_off()
            ax.text(
                0.5, 0.5,
                "Need at least 3 subjects with matching behavior.\n"
                "Check the id column matches your subject ids.",
                ha="center", va="center",
            )
            self._canvas.draw_idle()
            self.result_label.setText(f"Only {len(features)} subject(s) matched behavior.")
            return
        r, p = group.group_correlation(features, behavior, method=method)
        slope, intercept, _, _ = group.linear_fit(features, behavior)
        ax.scatter(features, behavior, color="#2b6cb0")
        xs = np.array([min(features), max(features)])
        ax.plot(xs, slope * xs + intercept, color="#c53030", linewidth=1.5)
        ax.set_xlabel(f"{band} power at {channel}")
        ax.set_ylabel(self.value_column.currentText())
        ax.set_title(f"{method.title()}: r={r:.2f}, p={p:.3f}  (n={len(features)})")
        self._fig.tight_layout()
        self._canvas.draw_idle()
        self.result_label.setText(
            f"{method.title()} r={r:.3f}, p={p:.3f}, n={len(features)} subjects"
        )

    def _export(self) -> None:
        import csv

        if not self._result_rows:
            self.result_label.setText("Compute first, then export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export group table", "cortica-group.csv", "CSV (*.csv)"
        )
        if not path:
            return
        feature_name = f"{self.band.currentText()}_{self.channel.currentText()}"
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["subject", feature_name, self.value_column.currentText()])
            for sid, feature, behavior in self._result_rows:
                writer.writerow([sid, f"{feature:.6g}", f"{behavior:.6g}"])
        self.result_label.setText(f"Exported {len(self._result_rows)} subjects to {path}")
