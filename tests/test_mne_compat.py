"""Tests for the MNE version-compat shims."""
import pytest

mne = pytest.importorskip("mne")

from cortica._mne_compat import resolve_montage_name  # noqa: E402


def test_resolve_returns_a_name_the_installed_mne_knows():
    assert resolve_montage_name("colin27_1005") in mne.channels.get_builtin_montages()


def test_resolve_maps_the_renamed_alias():
    # Exactly one of the two names exists in any given MNE (standard_* was renamed
    # to colin27_*); resolving the absent one must yield the one that is present.
    builtins = set(mne.channels.get_builtin_montages())
    present = "colin27_1005" if "colin27_1005" in builtins else "standard_1005"
    absent = "standard_1005" if present == "colin27_1005" else "colin27_1005"
    assert resolve_montage_name(absent) == present
