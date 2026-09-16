"""The Cortica main window.

A thin three-pane view over :class:`AppState`: a modality-aware step library
(left), a signal viewer (center, pyqtgraph), and the pipeline + Run (right).
Every user action goes through AppState; the window only reflects its signals.
Heavy runs go through :func:`run_in_background` so the UI stays responsive.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .. import io
from .runner import run_in_background
from .state import AppState

_TRACE_COLORS = ["#5ac8fa", "#f5a623", "#5fd3a6", "#f76d8e", "#9b8cff", "#3fc1c9"]


class MainWindow(QMainWindow):
    def __init__(self, state: AppState | None = None, parent=None):
        super().__init__(parent)
        self.state = state or AppState()
        self._run_signals = None  # keeps a running worker's signals alive
        self.setWindowTitle("Cortica")
        self._build_ui()
        self._connect_state()
        self._refresh_library()
        self._refresh_pipeline()
        self._replot()

    # ---- construction -------------------------------------------------------
    def _build_ui(self) -> None:
        toolbar = self.addToolBar("Main")
        open_action = QAction("Open…", self)
        open_action.triggered.connect(self._open)
        toolbar.addAction(open_action)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.addWidget(QLabel("Step library"))
        self.library = QListWidget()
        self.library.itemDoubleClicked.connect(self._on_library_double_clicked)
        left_layout.addWidget(self.library)

        self.plot = pg.PlotWidget()
        self.plot.setBackground("#0c141e")
        self.plot.showGrid(x=True, y=True, alpha=0.15)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.addWidget(QLabel("Pipeline"))
        self.pipeline_list = QListWidget()
        right_layout.addWidget(self.pipeline_list)
        self.run_button = QPushButton("Run pipeline")
        self.run_button.clicked.connect(self._run)
        right_layout.addWidget(self.run_button)

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(self.plot)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([250, 600, 320])
        self.setCentralWidget(splitter)
        self.statusBar().showMessage("Open a recording to begin.")

    def _connect_state(self) -> None:
        self.state.sourceChanged.connect(self._refresh_library)
        self.state.sourceChanged.connect(self._replot)
        self.state.pipelineChanged.connect(self._refresh_pipeline)
        self.state.resultChanged.connect(self._replot)

    # ---- step library -------------------------------------------------------
    def _refresh_library(self) -> None:
        self.library.clear()
        for cls, applicable in self.state.available_steps():
            item = QListWidgetItem(cls.name)
            item.setData(Qt.ItemDataRole.UserRole, cls.id)
            if not applicable:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                item.setToolTip(f"Not applicable to {self.state.modality} data")
            self.library.addItem(item)

    def _on_library_double_clicked(self, item: QListWidgetItem) -> None:
        if item.flags() & Qt.ItemFlag.ItemIsEnabled:
            self._add_from_item(item)

    def _add_from_item(self, item: QListWidgetItem) -> None:
        self.state.add_step(item.data(Qt.ItemDataRole.UserRole))
        self.statusBar().showMessage(f"Added {item.text()}")

    # ---- pipeline panel -----------------------------------------------------
    def _refresh_pipeline(self) -> None:
        self.pipeline_list.clear()
        for i, pstep in enumerate(self.state.pipeline.steps, 1):
            try:
                name = self.state.registry.get(pstep.step_id).name
            except KeyError:
                name = pstep.step_id
            self.pipeline_list.addItem(f"{i}. {name}")

    # ---- run ----------------------------------------------------------------
    def _run(self) -> None:
        if self.state.source is None:
            self.statusBar().showMessage("Load a recording first.")
            return
        self.run_button.setEnabled(False)
        self.statusBar().showMessage("Running pipeline…")
        source, pipeline, registry = self.state.source, self.state.pipeline, self.state.registry
        # Keep the signals object alive until it fires; a local ref would be
        # garbage-collected before the queued cross-thread signal is delivered.
        self._run_signals = run_in_background(lambda: pipeline.run(source, registry))
        self._run_signals.finished.connect(self._on_run_finished)
        self._run_signals.failed.connect(self._on_run_failed)

    def _on_run_finished(self, result) -> None:
        self.state.result = result
        self.state.resultChanged.emit()
        self.run_button.setEnabled(True)
        self.statusBar().showMessage(f"Ran {len(self.state.pipeline.steps)} step(s).")
        self._run_signals = None

    def _on_run_failed(self, message: str) -> None:
        self.run_button.setEnabled(True)
        self.statusBar().showMessage(f"Run failed: {message}")
        self._run_signals = None

    # ---- open ---------------------------------------------------------------
    def _open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open recording",
            "",
            "Recordings (*.fif *.edf *.bdf *.vhdr *.set *.snirf);;All files (*)",
        )
        if not path:
            return
        try:
            self.state.set_source(io.load_raw(path))
        except Exception as exc:
            self.statusBar().showMessage(f"Could not open file: {exc}")
            return
        self.statusBar().showMessage(f"Loaded {path}")

    # ---- signal viewer ------------------------------------------------------
    def _replot(self) -> None:
        self.plot.clear()
        ds = self.state.current()
        payload = getattr(ds, "payload", None) if ds else None
        if payload is None or not hasattr(payload, "get_data"):
            return
        data = payload.get_data()
        times = getattr(payload, "times", np.arange(data.shape[1]))
        n = min(len(data), 6)
        for i in range(n):
            channel = data[i]
            scale = channel.std() or 1.0
            offset = (n - 1 - i)
            y = channel / (4 * scale) + offset
            self.plot.plot(times, y, pen=pg.mkPen(_TRACE_COLORS[i % len(_TRACE_COLORS)], width=1))
