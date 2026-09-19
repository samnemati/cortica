"""ICA-based artifact removal.

Fitting uses a fixed ``random_state`` so the decomposition is reproducible: the
``exclude`` indices chosen in the inspector mean the same thing on every re-run.
"""
from __future__ import annotations

from ..core.params import Int, Str
from ..core.registry import register
from ..core.step import Step


def _parse_indices(text: str) -> list[int]:
    return [int(tok) for tok in str(text).replace(" ", "").split(",") if tok]


@register
class ICA(Step):
    id = "ica"
    name = "ICA (artifact removal)"
    category = "Artifacts & quality"
    modalities = ["eeg"]
    params = [
        Int("n_components", 15, min=1, label="Components"),
        Str("exclude", "", label="Exclude (indices, comma-separated)"),
    ]

    def run(self, ds, p):
        import mne

        raw = ds.payload.copy()
        n = min(p["n_components"], len(raw.ch_names))
        ica = mne.preprocessing.ICA(
            n_components=n, random_state=97, max_iter="auto", verbose=False
        )
        ica.fit(raw)
        ica.exclude = [i for i in _parse_indices(p["exclude"]) if 0 <= i < ica.n_components_]
        ica.apply(raw, verbose=False)
        return ds.derive(raw)
