"""Small shims for differences between MNE versions.

MNE renamed the template 10-05 / 10-20 montages from ``standard_1005`` /
``standard_1020`` to ``colin27_1005`` / ``colin27_1020``. Older MNE — still the
version pip resolves on Python 3.10 — knows only the ``standard_*`` names, while
newer MNE knows only the ``colin27_*`` names. Resolve whichever the installed MNE
actually provides so the same code and the same saved pipelines run everywhere.
"""
from __future__ import annotations

_MONTAGE_ALIASES = {
    "colin27_1005": "standard_1005",
    "colin27_1020": "standard_1020",
    "standard_1005": "colin27_1005",
    "standard_1020": "colin27_1020",
}


def resolve_montage_name(name: str) -> str:
    """Return ``name`` if the installed MNE knows it, else its cross-version alias.

    Falls back to ``name`` unchanged when neither is available, so MNE raises its
    own clear error rather than us masking an genuinely unknown montage.
    """
    import mne

    available = set(mne.channels.get_builtin_montages())
    if name in available:
        return name
    alt = _MONTAGE_ALIASES.get(name)
    if alt is not None and alt in available:
        return alt
    return name
