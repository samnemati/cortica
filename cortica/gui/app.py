"""Desktop application entry point.

``run_app`` creates the QApplication and shows the main window. It is imported
lazily (only when the GUI is actually launched) so that importing this module —
and therefore the CLI — does not require PySide6 unless the GUI is used.
"""
from __future__ import annotations


def run_app(argv: list[str] | None = None) -> int:
    from PySide6.QtWidgets import QApplication

    import cortica.steps  # noqa: F401  register the built-in steps

    from .main_window import MainWindow

    app = QApplication.instance() or QApplication(argv or [])
    window = MainWindow()
    window.resize(1100, 640)
    window.show()
    return app.exec()
