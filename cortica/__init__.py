"""Cortica — a guided, reproducible workbench for EEG & fNIRS analysis.

Cortica turns every action in its GUI into a step in a re-runnable pipeline, so
reproducibility and a shareable report come "for free": the same pipeline that
ran interactively can be re-run headless on new data and rendered into a report.

Layers:
    cortica.core    — headless engine (no Qt): Dataset, Step, Pipeline, registry.
    cortica.steps   — concrete analysis steps (MNE-backed; imported lazily).
    cortica.report  — pipeline + outputs -> HTML report.
    cortica.gui     — PySide6 desktop app (optional extra; imported lazily).
"""

__version__ = "0.1.0"
