"""Render a pipeline and its outputs into a self-contained HTML report.

Built on ``mne.Report``. Each step may contribute a section (its parameters,
figures, and a plain-language summary); the report embeds a provenance block so
a reader can reproduce it.
"""
