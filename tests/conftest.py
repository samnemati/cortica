"""Shared pytest configuration.

Force Qt to the offscreen platform so GUI tests run without a display (CI, servers).
Set before any PySide6 import so the plugin picks it up.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
