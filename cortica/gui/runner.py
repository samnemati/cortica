"""Run a callable on a background thread and report the outcome via Qt signals.

Used so heavy pipeline runs never block the GUI thread. The worker is started via
a zero-delay timer so a caller can connect to the signals before it fires.
"""
from __future__ import annotations

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal, Slot


class WorkerSignals(QObject):
    finished = Signal(object)  # emits the callable's return value
    failed = Signal(str)       # emits the error message


class _Worker(QRunnable):
    def __init__(self, fn, signals: WorkerSignals):
        super().__init__()
        self._fn = fn
        self._signals = signals

    @Slot()
    def run(self):
        try:
            result = self._fn()
        except Exception as exc:
            self._signals.failed.emit(str(exc))
        else:
            self._signals.finished.emit(result)


def run_in_background(fn, pool: QThreadPool | None = None) -> WorkerSignals:
    """Run ``fn()`` on a thread pool; return signals that report the outcome."""
    signals = WorkerSignals()
    worker = _Worker(fn, signals)
    pool = pool or QThreadPool.globalInstance()
    QTimer.singleShot(0, lambda: pool.start(worker))
    return signals
