"""The value that flows between steps.

A ``Dataset`` is treated as immutable: steps never mutate it in place, they
:meth:`derive` a new one. That is what makes undo/redo and re-running safe.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Dataset:
    """Current signal payload plus its modality, metadata, and applied history.

    Attributes:
        payload: the underlying data (an MNE ``Raw``/``Epochs`` object in real use;
            anything in tests).
        modality: ``"eeg"`` or ``"fnirs"``.
        meta: free-form metadata (sampling rate, channel count, source file, ...).
        history: tuple of applied step records ``{"step": id, "params": {...}}``.
    """

    payload: Any
    modality: str
    meta: dict = field(default_factory=dict)
    history: tuple = ()

    def derive(self, payload: Any, record: dict | None = None) -> Dataset:
        """Return a new Dataset with a new payload, optionally recording a step."""
        history = self.history + ((record,) if record is not None else ())
        return Dataset(payload, self.modality, dict(self.meta), history)
