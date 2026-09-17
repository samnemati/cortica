"""AppState — the GUI controller.

Holds the loaded source dataset, the pipeline being assembled, and the last run
result, and emits Qt signals when they change. Widgets are thin views bound to
these signals; all the logic lives here so it is testable without a display.
"""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from ..core.dataset import Dataset
from ..core.pipeline import Pipeline
from ..core.registry import default_registry


class AppState(QObject):
    sourceChanged = Signal()
    pipelineChanged = Signal()
    resultChanged = Signal()

    def __init__(self, registry=None, parent=None):
        super().__init__(parent)
        self._registry = registry or default_registry
        self.source: Dataset | None = None
        self.result: Dataset | None = None
        self.pipeline = Pipeline("eeg")

    @property
    def registry(self):
        return self._registry

    @property
    def modality(self) -> str | None:
        return self.source.modality if self.source else None

    def set_source(self, dataset: Dataset) -> None:
        """Load a dataset: reset the pipeline to its modality and clear the result."""
        self.source = dataset
        self.result = None
        self.pipeline = Pipeline(dataset.modality)
        self.sourceChanged.emit()
        self.pipelineChanged.emit()

    def available_steps(self) -> list[tuple[type, bool]]:
        """Every registered step with a flag: does it apply to the current modality?"""
        modality = self.modality
        steps = sorted(self._registry.all(), key=lambda c: (c.category, c.name))
        return [(cls, modality is not None and modality in cls.modalities) for cls in steps]

    def add_step(self, step_id: str, params: dict | None = None) -> None:
        self.pipeline.add(step_id, params)
        self.pipelineChanged.emit()

    def remove_step(self, index: int) -> None:
        del self.pipeline.steps[index]
        self.pipelineChanged.emit()

    def set_step_params(self, index: int, params: dict) -> None:
        """Replace the params of the step at ``index`` (no list rebuild)."""
        self.pipeline.steps[index].params = dict(params)

    def move_step(self, index: int, delta: int) -> None:
        """Move a step up (delta=-1) or down (delta=+1); out-of-range is a no-op."""
        target = index + delta
        steps = self.pipeline.steps
        if index != target and 0 <= target < len(steps):
            steps[index], steps[target] = steps[target], steps[index]
            self.pipelineChanged.emit()

    def run_sync(self) -> Dataset:
        """Run the pipeline on the source (not the previous result). Sets result."""
        if self.source is None:
            raise RuntimeError("No dataset loaded")
        self.result = self.pipeline.run(self.source, self._registry)
        self.resultChanged.emit()
        return self.result

    def current(self) -> Dataset | None:
        """The dataset to display: the result if we have run, else the source."""
        return self.result or self.source
