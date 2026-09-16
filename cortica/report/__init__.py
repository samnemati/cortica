"""Render a pipeline and its outputs into a self-contained HTML report.

The current renderer (:func:`build_report`) uses only the standard library and
records the pipeline, a dataset summary, and a provenance block. Richer per-step
figures via ``mne.Report`` are planned for a later slice.
"""
from .html import build_report

__all__ = ["build_report"]
