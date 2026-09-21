"""Tests for the background runner used to keep the UI responsive."""
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from cortica.gui.runner import run_in_background  # noqa: E402


def _boom():
    raise ValueError("nope")


def test_runs_function_off_thread_and_emits_result(qtbot):
    signals = run_in_background(lambda: 2 + 3)
    with qtbot.waitSignal(signals.finished, timeout=3000) as blocker:
        pass
    assert blocker.args == [5]


def test_emits_failed_with_message_on_exception(qtbot):
    signals = run_in_background(_boom)
    with qtbot.waitSignal(signals.failed, timeout=3000) as blocker:
        pass
    assert "nope" in blocker.args[0]


def test_passes_a_progress_emitter_to_functions_that_accept_one(qtbot):
    def work(progress):
        progress(1, 3, "step")
        return "ok"

    signals = run_in_background(work)
    with qtbot.waitSignal(signals.progress, timeout=3000) as blocker:
        pass
    assert blocker.args == [1, 3, "step"]
