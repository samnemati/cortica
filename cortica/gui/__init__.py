"""PySide6 desktop GUI (optional extra).

The GUI only orchestrates the engine: every user action appends or edits a step
in a :class:`cortica.core.pipeline.Pipeline`. It never performs analysis itself.
PySide6 is imported lazily so the package imports without Qt installed.
"""
