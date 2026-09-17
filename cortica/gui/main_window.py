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
    QHBoxLayout,
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
from .param_form import ParamForm
from .runner import run_in_background
from .state import AppState

_TRACE_COLORS = ["#5ac8fa", "#f5a623", "#5fd3a6", "#f76d8e", "#9b8cff", "#3fc1c9"]


class MainWindow(QMainWindow):
    def __init__(self, state: AppState | None = None, parent=None):
        super().__init__(parent)
        self.state = state or AppState()
        self._run_signals = None  # keeps a running worker's signals alive
        self._param_form: ParamForm | None = None
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

        sample_menu = self.menuBar().addMenu("Sample")
        eeg_action = QAction("Load EEG sample", self)
        eeg_action.triggered.connect(self._load_eeg_sample)
        sample_menu.addAction(eeg_action)
        fnirs_action = QAction("Load fNIRS sample", self)
        fnirs_action.triggered.connect(self._load_fnirs_sample)
        sample_menu.addAction(fnirs_action)

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
        self.pipeline_list.currentRowChanged.connect(self._show_params_for)
        right_layout.addWidget(self.pipeline_list)

        buttons = QHBoxLayout()
        up_button = QPushButton("↑")
        up_button.setToolTip("Move step up")
        up_button.clicked.connect(lambda: self._move_selected(-1))
        down_button = QPushButton("↓")
        down_button.setToolTip("Move step down")
        down_button.clicked.connect(lambda: self._move_selected(1))
        remove_button = QPushButton("Remove")
        remove_button.clicked.connect(self._remove_selected)
        buttons.addWidget(up_button)
        buttons.addWidget(down_button)
        buttons.addWidget(remove_button)
        right_layout.addLayout(buttons)

        self._param_placeholder = QLabel("Select a step to edit its parameters.")
        self._param_placeholder.setWordWrap(True)
        self._param_placeholder.setStyleSheet("color: #8a99a8;")
        right_layout.addWidget(self._param_placeholder)
        self._param_container = QWidget()
        self._param_layout = QVBoxLayout(self._param_container)
        self._param_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self._param_container)

        right_layout.addStretch(1)
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
        row = self.pipeline_list.currentRow()
        self.pipeline_list.blockSignals(True)
        self.pipeline_list.clear()
        for i, pstep in enumerate(self.state.pipeline.steps, 1):
            try:
                name = self.state.registry.get(pstep.step_id).name
            except KeyError:
                name = pstep.step_id
            self.pipeline_list.addItem(f"{i}. {name}")
        row = min(row, self.pipeline_list.count() - 1)
        self.pipeline_list.setCurrentRow(row)
        self.pipeline_list.blockSignals(False)
        self._show_params_for(row)

    def _show_params_for(self, row: int) -> None:
        if self._param_form is not None:
            self._param_form.setParent(None)
            self._param_form.deleteLater()
            self._param_form = None
        steps = self.state.pipeline.steps
        if row is None or row < 0 or row >= len(steps):
            self._param_placeholder.setVisible(True)
            return
        self._param_placeholder.setVisible(False)
        pstep = steps[row]
        try:
            params = self.state.registry.get(pstep.step_id).params
        except KeyError:
            params = []
        form = ParamForm(params, pstep.params)
        form.changed.connect(lambda values, i=row: self.state.set_step_params(i, values))
        self._param_layout.addWidget(form)
        self._param_form = form

    def _remove_selected(self) -> None:
        row = self.pipeline_list.currentRow()
        if 0 <= row < len(self.state.pipeline.steps):
            self.state.remove_step(row)

    def _move_selected(self, delta: int) -> None:
        row = self.pipeline_list.currentRow()
        if row < 0:
            return
        self.state.move_step(row, delta)
        new_row = row + delta
        if 0 <= new_row < self.pipeline_list.count():
            self.pipeline_list.setCurrentRow(new_row)

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

    def _load_eeg_sample(self) -> None:
        from .. import samples

        self.state.set_source(samples.eeg_sample())
        self.statusBar().showMessage("Loaded synthetic EEG sample.")

    def _load_fnirs_sample(self) -> None:
        from .. import samples

        self.state.set_source(samples.fnirs_sample())
        self.statusBar().showMessage("Loaded synthetic fNIRS sample.")

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
