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
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .. import io, viz
from .param_form import ParamForm
from .runner import run_in_background
from .state import AppState

_TRACE_COLORS = ["#5ac8fa", "#f5a623", "#5fd3a6", "#f76d8e", "#9b8cff", "#3fc1c9"]

#: View dropdown labels paired with their internal mode keys (one source of truth
#: for both directions, so the dropdown always reflects the active view).
_VIEWS = [
    ("Time series", "time"),
    ("Power spectrum", "psd"),
    ("Topography", "topo"),
    ("Time-frequency", "tfr"),
    ("Connectivity", "conn"),
    ("Decoding", "decoding"),
    ("Statistics", "stats"),
    ("Comparison", "compare"),
    ("Source", "source"),
]


class MainWindow(QMainWindow):
    def __init__(self, state: AppState | None = None, parent=None):
        super().__init__(parent)
        self.state = state or AppState()
        self._run_signals = None  # keeps a running worker's signals alive
        self._param_form: ParamForm | None = None
        self._view_mode = "time"
        self._band = "Alpha"
        self._topo_threshold = 1.0
        self._tfr_method = "morlet"
        self._conn_method = "plv"
        self._conn_style = "matrix"
        self._conn_threshold = 0.0
        self._decode_mode = "time"
        self._decode_classifier = "logreg"
        self._source_result = None
        self._src_signals = None
        self.setWindowTitle("Cortica")
        self._build_ui()
        self._connect_state()
        self._refresh_library()
        self._refresh_pipeline()
        self._replot()

    # ---- construction -------------------------------------------------------
    def _build_ui(self) -> None:
        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)
        # Sample loaders first, right in the window toolbar, so they're easy to find.
        eeg_action = QAction("Load EEG sample", self)
        eeg_action.triggered.connect(self._load_eeg_sample)
        toolbar.addAction(eeg_action)
        fnirs_action = QAction("Load fNIRS sample", self)
        fnirs_action.triggered.connect(self._load_fnirs_sample)
        toolbar.addAction(fnirs_action)
        toolbar.addSeparator()
        open_action = QAction("Open…", self)
        open_action.triggered.connect(self._open)
        toolbar.addAction(open_action)
        export_action = QAction("Export report…", self)
        export_action.triggered.connect(self._export_report)
        toolbar.addAction(export_action)
        ica_action = QAction("Fit ICA…", self)
        ica_action.triggered.connect(self._fit_ica)
        toolbar.addAction(ica_action)
        source_action = QAction("Localize sources…", self)
        source_action.triggered.connect(self._localize_sources)
        toolbar.addAction(source_action)

        # Keep the same actions in a menu too (native menu bar on macOS/Linux).
        sample_menu = self.menuBar().addMenu("Sample")
        sample_menu.addAction(eeg_action)
        sample_menu.addAction(fnirs_action)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.addWidget(QLabel("Step library"))
        self.library = QListWidget()
        self.library.itemDoubleClicked.connect(self._on_library_double_clicked)
        left_layout.addWidget(self.library)
        left_layout.addWidget(QLabel("Channels"))
        self.channel_list = QListWidget()
        self.channel_list.itemChanged.connect(self._on_channels_changed)
        left_layout.addWidget(self.channel_list)

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        view_row = QHBoxLayout()
        view_row.setContentsMargins(8, 6, 8, 0)
        view_row.addWidget(QLabel("View:"))
        self.view_selector = QComboBox()
        self.view_selector.addItems([label for label, _ in _VIEWS])
        self.view_selector.currentTextChanged.connect(self._on_view_changed)
        view_row.addWidget(self.view_selector)
        # Topography-only controls (hidden unless the head-map view is active).
        self.band_label = QLabel("Band:")
        view_row.addWidget(self.band_label)
        self.band_selector = QComboBox()
        self.band_selector.addItems(list(viz.BANDS))
        self.band_selector.setCurrentText(self._band)
        self.band_selector.currentTextChanged.connect(self._on_band_changed)
        view_row.addWidget(self.band_selector)
        self.threshold_label = QLabel("Threshold:")
        view_row.addWidget(self.threshold_label)
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(10, 100)
        self.threshold_slider.setValue(100)
        self.threshold_slider.setFixedWidth(120)
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        view_row.addWidget(self.threshold_slider)
        self.tfr_method_label = QLabel("Method:")
        view_row.addWidget(self.tfr_method_label)
        self.tfr_method_selector = QComboBox()
        self.tfr_method_selector.addItems(["Morlet", "Multitaper"])
        self.tfr_method_selector.currentTextChanged.connect(self._on_tfr_method_changed)
        view_row.addWidget(self.tfr_method_selector)
        self.conn_method_label = QLabel("Measure:")
        view_row.addWidget(self.conn_method_label)
        self.conn_method_selector = QComboBox()
        self.conn_method_selector.addItems(list(viz.CONNECTIVITY_METHODS))
        self.conn_method_selector.currentTextChanged.connect(self._on_conn_method_changed)
        view_row.addWidget(self.conn_method_selector)
        self.conn_style_label = QLabel("Style:")
        view_row.addWidget(self.conn_style_label)
        self.conn_style_selector = QComboBox()
        self.conn_style_selector.addItems(["Matrix", "Connectogram"])
        self.conn_style_selector.currentTextChanged.connect(self._on_conn_style_changed)
        view_row.addWidget(self.conn_style_selector)
        self.conn_threshold_label = QLabel("Sparsity:")
        view_row.addWidget(self.conn_threshold_label)
        self.conn_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.conn_threshold_slider.setRange(0, 90)  # % of the weakest edges to hide
        self.conn_threshold_slider.setValue(0)
        self.conn_threshold_slider.setFixedWidth(120)
        self.conn_threshold_slider.setToolTip(
            "Hide the weakest edges — right shows only the strongest"
        )
        self.conn_threshold_slider.valueChanged.connect(self._on_conn_threshold_changed)
        view_row.addWidget(self.conn_threshold_slider)
        # Decoding-only controls (hidden unless the decoding view is active).
        self.decode_mode_label = QLabel("Mode:")
        view_row.addWidget(self.decode_mode_label)
        self.decode_mode_selector = QComboBox()
        self.decode_mode_selector.addItems(
            ["Over time", "Temporal generalization", "CSP (whole epoch)"]
        )
        self.decode_mode_selector.currentTextChanged.connect(self._on_decode_mode_changed)
        view_row.addWidget(self.decode_mode_selector)
        self.classifier_label = QLabel("Classifier:")
        view_row.addWidget(self.classifier_label)
        self.classifier_selector = QComboBox()
        self.classifier_selector.addItems(list(viz.DECODE_CLASSIFIERS))
        self.classifier_selector.currentTextChanged.connect(self._on_classifier_changed)
        view_row.addWidget(self.classifier_selector)
        view_row.addStretch(1)
        center_layout.addLayout(view_row)

        self.plot = pg.PlotWidget()
        self.plot.setBackground("#0c141e")
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        center_layout.addWidget(self.plot)
        self._mpl_fig = Figure(figsize=(4, 4))
        self._mpl_canvas = FigureCanvasQTAgg(self._mpl_fig)
        center_layout.addWidget(self._mpl_canvas)
        self._mpl_canvas.hide()
        self._update_view_controls()

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
        splitter.addWidget(center)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([250, 600, 320])
        self.setCentralWidget(splitter)
        self.statusBar().showMessage(
            "Click “Load EEG sample” or “Load fNIRS sample” to try it — or Open your own recording."
        )

    def _connect_state(self) -> None:
        self.state.sourceChanged.connect(self._refresh_library)
        self.state.sourceChanged.connect(self._refresh_channels)
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

    # ---- channel selection --------------------------------------------------
    def _refresh_channels(self) -> None:
        self.channel_list.blockSignals(True)
        self.channel_list.clear()
        ds = self.state.current()
        payload = getattr(ds, "payload", None) if ds else None
        for name in getattr(payload, "ch_names", []):
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.channel_list.addItem(item)
        self.channel_list.blockSignals(False)

    def _current_picks(self) -> list:
        return [
            self.channel_list.item(i).text()
            for i in range(self.channel_list.count())
            if self.channel_list.item(i).checkState() == Qt.CheckState.Checked
        ]

    def _on_channels_changed(self, item) -> None:
        self._replot()

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

    # ---- report -------------------------------------------------------------
    def _export_report(self) -> None:
        if self.state.current() is None:
            self.statusBar().showMessage("Load a recording first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export report", "cortica-report.html", "HTML (*.html)"
        )
        if path:
            self._export_report_to(path)

    def _export_report_to(self, path: str) -> None:
        from ..report import build_report

        dataset = self.state.current()
        if dataset is None:
            self.statusBar().showMessage("Load a recording first.")
            return
        build_report(dataset, self.state.pipeline, path)
        self.statusBar().showMessage(f"Report written to {path}")

    # ---- ICA ----------------------------------------------------------------
    def _fit_ica(self) -> None:
        ds = self.state.current()
        payload = getattr(ds, "payload", None) if ds else None
        if payload is None or not hasattr(payload, "get_data"):
            self.statusBar().showMessage("Load a recording first.")
            return
        from .ica_dialog import ICADialog

        self.statusBar().showMessage("Fitting ICA…")
        dialog = ICADialog(payload, parent=self)
        if dialog.exec():
            self._apply_ica_choice(dialog.n_components, dialog.excluded())

    def _apply_ica_choice(self, n_components: int, excluded: list) -> None:
        self.state.add_step(
            "ica",
            {"n_components": n_components, "exclude": ",".join(str(i) for i in excluded)},
        )
        self.statusBar().showMessage(f"Added ICA (removing {len(excluded)} component(s)).")

    # ---- source localization ------------------------------------------------
    def _localize_sources(self) -> None:
        import mne

        ds = self.state.current()
        payload = getattr(ds, "payload", None) if ds else None
        if not isinstance(payload, mne.Evoked):
            self.statusBar().showMessage(
                "Source localization needs an evoked — average epochs first."
            )
            return
        self.statusBar().showMessage("Localizing sources (fsaverage template, ~10 s)…")
        self._src_signals = run_in_background(lambda: viz.source_localization(payload))
        self._src_signals.finished.connect(self._on_sources_ready)
        self._src_signals.failed.connect(self._on_sources_failed)

    def _on_sources_ready(self, result) -> None:
        self._source_result = result
        self._src_signals = None
        self._set_view("source")
        self.statusBar().showMessage("Source localization complete.")

    def _on_sources_failed(self, message: str) -> None:
        self._src_signals = None
        self.statusBar().showMessage(f"Source localization: {message}")

    # ---- signal viewer ------------------------------------------------------
    def _on_view_changed(self, text: str) -> None:
        self._set_view(dict(_VIEWS).get(text, "time"))

    def _set_view(self, mode: str) -> None:
        self._view_mode = mode
        label = {m: lbl for lbl, m in _VIEWS}.get(mode)
        if label is not None and self.view_selector.currentText() != label:
            # Keep the dropdown in step without re-triggering _on_view_changed.
            self.view_selector.blockSignals(True)
            self.view_selector.setCurrentText(label)
            self.view_selector.blockSignals(False)
        self._replot()

    def _on_band_changed(self, band: str) -> None:
        self._band = band
        if self._view_mode == "topo":
            self._replot()

    def _on_threshold_changed(self, value: int) -> None:
        self._topo_threshold = value / 100.0
        if self._view_mode == "topo":
            self._replot()

    def _on_tfr_method_changed(self, text: str) -> None:
        self._tfr_method = text.lower()
        if self._view_mode == "tfr":
            self._replot()

    def _on_conn_method_changed(self, text: str) -> None:
        self._conn_method = viz.CONNECTIVITY_METHODS.get(text, "plv")
        if self._view_mode == "conn":
            self._replot()

    def _on_conn_style_changed(self, text: str) -> None:
        self._conn_style = "connectogram" if text == "Connectogram" else "matrix"
        if self._view_mode == "conn":
            self._replot()

    def _on_conn_threshold_changed(self, value: int) -> None:
        self._conn_threshold = value / 100.0
        if self._view_mode == "conn":
            self._replot()

    def _on_decode_mode_changed(self, text: str) -> None:
        self._decode_mode = {
            "Temporal generalization": "generalization",
            "CSP (whole epoch)": "csp",
        }.get(text, "time")
        if self._view_mode == "decoding":
            self._replot()

    def _on_classifier_changed(self, text: str) -> None:
        self._decode_classifier = viz.DECODE_CLASSIFIERS.get(text, "logreg")
        if self._view_mode == "decoding":
            self._replot()

    def _update_view_controls(self) -> None:
        view = self._view_mode
        for widget in (self.band_label, self.band_selector):
            widget.setVisible(view in ("topo", "conn"))
        for widget in (self.threshold_label, self.threshold_slider):
            widget.setVisible(view == "topo")
        for widget in (self.tfr_method_label, self.tfr_method_selector):
            widget.setVisible(view == "tfr")
        for widget in (self.conn_method_label, self.conn_method_selector,
                       self.conn_style_label, self.conn_style_selector,
                       self.conn_threshold_label, self.conn_threshold_slider):
            widget.setVisible(view == "conn")
        for widget in (self.decode_mode_label, self.decode_mode_selector,
                       self.classifier_label, self.classifier_selector):
            widget.setVisible(view == "decoding")

    def _replot(self) -> None:
        ds = self.state.current()
        payload = getattr(ds, "payload", None) if ds else None
        is_mpl = self._view_mode in (
            "topo", "tfr", "conn", "decoding", "stats", "source", "compare"
        )
        self.plot.setVisible(not is_mpl)
        self._mpl_canvas.setVisible(is_mpl)
        self._mpl_fig.set_facecolor("white")  # connectogram sets its own dark face
        self._update_view_controls()
        if self._view_mode == "topo":
            self._plot_topomap(payload)
            return
        if self._view_mode == "tfr":
            self._plot_tfr(payload)
            return
        if self._view_mode == "conn":
            self._plot_connectivity(payload)
            return
        if self._view_mode == "decoding":
            self._plot_decoding(payload)
            return
        if self._view_mode == "stats":
            self._plot_stats(payload)
            return
        if self._view_mode == "source":
            self._plot_source()
            return
        if self._view_mode == "compare":
            self._plot_compare(payload)
            return
        self.plot.clear()
        if payload is None or not hasattr(payload, "get_data"):
            return
        if self._view_mode == "psd":
            self._plot_spectrum(payload)
        else:
            self._plot_traces(payload)

    def _plot_topomap(self, payload) -> None:
        self._mpl_fig.clear()
        ax = self._mpl_fig.add_subplot(111)
        ax.set_axis_off()
        if payload is None or not hasattr(payload, "get_data"):
            ax.text(0.5, 0.5, "Load a recording to see a head map.", ha="center", va="center")
            self._mpl_canvas.draw_idle()
            return
        picks = self._current_picks()
        if not picks:
            ax.text(0.5, 0.5, "Select at least one channel.", ha="center", va="center")
            self._mpl_canvas.draw_idle()
            return
        try:
            import mne

            names = list(payload.ch_names)
            idx = [i for i, name in enumerate(names) if name in picks]
            power = viz.band_power(payload, self._band)[idx]
            vmax = self._topo_threshold * float(np.nanmax(power)) if power.size else None
            info = mne.pick_info(payload.info, idx)
            mne.viz.plot_topomap(
                power, info, axes=ax, show=False, cmap="RdBu_r", vlim=(None, vmax)
            )
            ax.set_title(f"{self._band} power")
        except Exception as exc:
            ax.clear()
            ax.set_axis_off()
            ax.text(
                0.5, 0.5,
                "Head map needs electrode positions.\nAdd a “Set montage” step.",
                ha="center", va="center",
            )
            self.statusBar().showMessage(f"Head map: {exc}")
        self._mpl_canvas.draw_idle()

    def _plot_tfr(self, payload) -> None:
        self._mpl_fig.clear()
        ax = self._mpl_fig.add_subplot(111)
        if payload is None or not hasattr(payload, "get_data"):
            ax.set_axis_off()
            ax.text(0.5, 0.5, "Load a recording to see time-frequency.", ha="center", va="center")
            self._mpl_canvas.draw_idle()
            return
        picks = self._current_picks()
        if not picks:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "Select at least one channel.", ha="center", va="center")
            self._mpl_canvas.draw_idle()
            return
        try:
            times, freqs, power = viz.time_frequency(
                payload, picks=picks, fmax=40.0, method=self._tfr_method
            )
            image = ax.imshow(
                power, aspect="auto", origin="lower", cmap="RdBu_r",
                extent=[times[0], times[-1], freqs[0], freqs[-1]],
            )
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            ax.set_title("Time-frequency power (mean of selected channels)")
            self._mpl_fig.colorbar(image, ax=ax)
        except Exception as exc:
            ax.clear()
            ax.set_axis_off()
            ax.text(0.5, 0.5, "Time-frequency unavailable.", ha="center", va="center")
            self.statusBar().showMessage(f"Time-frequency: {exc}")
        self._mpl_canvas.draw_idle()

    def _plot_connectivity(self, payload) -> None:
        import mne

        self._mpl_fig.clear()
        if not isinstance(payload, mne.BaseEpochs):
            ax = self._mpl_fig.add_subplot(111)
            ax.set_axis_off()
            ax.text(
                0.5, 0.5,
                "Connectivity needs epochs.\nAdd an Epochs step and Run.",
                ha="center", va="center",
            )
            self._mpl_canvas.draw_idle()
            return
        try:
            matrix, names = viz.connectivity(payload, method=self._conn_method, band=self._band)
            matrix = viz.threshold_matrix(matrix, self._conn_threshold)
            signed = float(np.nanmin(matrix)) < 0.0  # imcoh spans negative values
            if self._conn_style == "connectogram":
                self._plot_connectogram(matrix, names, signed)
            else:
                self._plot_conn_matrix(matrix, names, signed)
        except Exception as exc:
            self._mpl_fig.clear()
            ax = self._mpl_fig.add_subplot(111)
            ax.set_axis_off()
            ax.text(0.5, 0.5, "Connectivity unavailable.", ha="center", va="center")
            self.statusBar().showMessage(f"Connectivity: {exc}")
        self._mpl_canvas.draw_idle()

    def _plot_conn_matrix(self, matrix, names, signed) -> None:
        ax = self._mpl_fig.add_subplot(111)
        if signed:
            lim = max(abs(float(np.nanmin(matrix))), abs(float(np.nanmax(matrix))), 1e-6)
            image = ax.imshow(matrix, cmap="RdBu_r", vmin=-lim, vmax=lim)
        else:
            image = ax.imshow(matrix, cmap="magma", vmin=0, vmax=1)
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=90, fontsize=7)
        ax.set_yticklabels(names, fontsize=7)
        ax.set_title(f"{self._conn_method.upper()} — {self._band}")
        self._mpl_fig.colorbar(image, ax=ax)

    def _plot_connectogram(self, matrix, names, signed) -> None:
        from mne_connectivity.viz import plot_connectivity_circle

        self._mpl_fig.set_facecolor("#0c141e")
        ax = self._mpl_fig.add_subplot(111, polar=True)
        arr = np.asarray(matrix)
        n_nonzero = int((np.triu(arr, 1) != 0).sum())
        n_lines = n_nonzero if self._conn_threshold > 0.0 else None
        if signed:
            lim = max(abs(float(np.nanmin(arr))), abs(float(np.nanmax(arr))), 1e-6)
            cmap, vmin, vmax = "RdBu_r", -lim, lim
        else:
            cmap, vmin, vmax = "magma", 0.0, 1.0
        plot_connectivity_circle(
            arr, names, n_lines=n_lines, ax=ax, colormap=cmap, vmin=vmin, vmax=vmax,
            facecolor="#0c141e", textcolor="#dfe7ef", node_edgecolor="#0c141e",
            colorbar=True, interactive=False, show=False,
            title=f"{self._conn_method.upper()} — {self._band}",
        )

    def _plot_decoding(self, payload) -> None:
        import mne

        self._mpl_fig.clear()
        ax = self._mpl_fig.add_subplot(111)
        if not isinstance(payload, mne.BaseEpochs) or len(payload.event_id) < 2:
            ax.set_axis_off()
            ax.text(
                0.5, 0.5,
                "Decoding needs epochs with ≥2 conditions.\nAdd Epochs by events, then Run.",
                ha="center", va="center",
            )
            self._mpl_canvas.draw_idle()
            return
        chance = 1.0 / len(set(payload.events[:, 2]))
        clf = self._decode_classifier.upper()
        try:
            if self._decode_mode == "generalization":
                times, matrix = viz.temporal_generalization(payload, classifier=clf.lower())
                spread = max(float(np.abs(matrix - chance).max()), 0.01)
                image = ax.imshow(
                    matrix, origin="lower", cmap="RdBu_r",
                    vmin=chance - spread, vmax=chance + spread,
                    extent=[times[0], times[-1], times[0], times[-1]],
                )
                ax.set_xlabel("Test time (s)")
                ax.set_ylabel("Train time (s)")
                ax.set_title(f"Temporal generalization — {clf}")
                self._mpl_fig.colorbar(image, ax=ax, label="Accuracy")
            elif self._decode_mode == "csp":
                acc = viz.decoding_csp(payload, classifier=clf.lower())
                ax.bar([0], [acc], color="#5ac8fa", width=0.5)
                ax.axhline(chance, color="#8a99a8", linestyle="--", label="chance")
                ax.set_xlim(-1.5, 1.5)
                ax.set_ylim(0, 1)
                ax.set_xticks([0])
                ax.set_xticklabels(["CSP + " + clf])
                ax.set_ylabel("Accuracy")
                ax.set_title("CSP whole-epoch decoding")
                ax.text(0, acc + 0.02, f"{acc:.2f}", ha="center", va="bottom")
                ax.legend(loc="upper right", fontsize=8)
            else:
                times, scores = viz.decoding(payload, classifier=clf.lower())
                ax.plot(times, scores, color="#2b6cb0", linewidth=2)
                ax.axhline(chance, color="#8a99a8", linestyle="--", label="chance")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Accuracy")
                ax.set_title(f"Decoding over time — {clf}")
                ax.legend(loc="upper right", fontsize=8)
        except Exception as exc:
            ax.clear()
            ax.set_axis_off()
            ax.text(0.5, 0.5, "Decoding unavailable.", ha="center", va="center")
            self.statusBar().showMessage(f"Decoding: {exc}")
        self._mpl_fig.tight_layout()
        self._mpl_canvas.draw_idle()

    def _plot_stats(self, payload) -> None:
        import mne

        self._mpl_fig.clear()
        ax = self._mpl_fig.add_subplot(111)
        if not isinstance(payload, mne.BaseEpochs) or len(payload.event_id) < 2:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "Statistics need epochs with two conditions.",
                    ha="center", va="center")
            self._mpl_canvas.draw_idle()
            return
        try:
            times, mean_a, mean_b, significant, labels = viz.cluster_test(payload)
            ax.plot(times, mean_a * 1e6, label=labels[0])
            ax.plot(times, mean_b * 1e6, label=labels[1])
            lo, hi = ax.get_ylim()
            if significant.any():
                ax.fill_between(times, lo, hi, where=significant, color="0.7",
                                alpha=0.4, step="mid", label="p < 0.05")
                ax.set_ylim(lo, hi)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude (µV)")
            ax.set_title("Condition comparison (cluster permutation)")
            ax.legend(loc="upper right", fontsize=8)
        except Exception as exc:
            ax.clear()
            ax.set_axis_off()
            ax.text(0.5, 0.5, "Statistics unavailable.", ha="center", va="center")
            self.statusBar().showMessage(f"Statistics: {exc}")
        self._mpl_canvas.draw_idle()

    def _plot_source(self) -> None:
        self._mpl_fig.clear()
        ax = self._mpl_fig.add_subplot(111)
        if not self._source_result:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "Use “Localize sources…” on an evoked.", ha="center", va="center")
            self._mpl_canvas.draw_idle()
            return
        names, strengths = self._source_result
        positions = list(range(len(names)))
        ax.barh(positions, list(strengths), color="#2b6cb0")
        ax.set_yticks(positions)
        ax.set_yticklabels(names, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Mean |dSPM|")
        ax.set_title("Most active regions (fsaverage source estimate)")
        self._mpl_fig.tight_layout()
        self._mpl_canvas.draw_idle()

    def _plot_compare(self, payload) -> None:
        import mne

        self._mpl_fig.clear()
        ax = self._mpl_fig.add_subplot(111)
        if not isinstance(payload, mne.BaseEpochs) or len(payload.event_id) < 2:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "Comparison needs epochs with two or more conditions.",
                    ha="center", va="center")
            self._mpl_canvas.draw_idle()
            return
        try:
            times, means, difference = viz.condition_comparison(payload)
            for label, time_course in means.items():
                ax.plot(times, time_course * 1e6, label=label)
            if difference is not None:
                ax.plot(times, difference * 1e6, "k--", linewidth=1, label="difference")
            ax.axhline(0, color="0.8", linewidth=0.8)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude (µV)")
            ax.set_title("Condition comparison")
            ax.legend(loc="upper right", fontsize=8)
        except Exception as exc:
            ax.clear()
            ax.set_axis_off()
            ax.text(0.5, 0.5, "Comparison unavailable.", ha="center", va="center")
            self.statusBar().showMessage(f"Comparison: {exc}")
        self._mpl_canvas.draw_idle()

    def _plot_traces(self, payload) -> None:
        picks = self._current_picks()
        if not picks:
            return
        self.plot.setLabel("bottom", "Time", units="s")
        self.plot.setLabel("left", "Channels (stacked)")
        times, data = viz.traces(payload, picks=picks)
        n = min(len(data), 6)
        for i in range(n):
            channel = data[i]
            scale = channel.std() or 1.0
            y = channel / (4 * scale) + (n - 1 - i)
            self.plot.plot(times, y, pen=pg.mkPen(_TRACE_COLORS[i % len(_TRACE_COLORS)], width=1))

    def _plot_spectrum(self, payload) -> None:
        picks = self._current_picks()
        if not picks:
            return
        self.plot.setLabel("bottom", "Frequency", units="Hz")
        self.plot.setLabel("left", "Power (dB)")
        try:
            freqs, psds = viz.spectrum(payload, fmax=45.0, picks=picks)
        except Exception as exc:
            self.statusBar().showMessage(f"Spectrum unavailable: {exc}")
            return
        n = min(len(psds), 6)
        for i in range(n):
            power_db = 10 * np.log10(np.maximum(psds[i], 1e-30))
            pen = pg.mkPen(_TRACE_COLORS[i % len(_TRACE_COLORS)], width=1)
            self.plot.plot(freqs, power_db, pen=pen)
