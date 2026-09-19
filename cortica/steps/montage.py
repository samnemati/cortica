"""Electrode montage — assigns 3-D positions to channels by name.

Positions are required for anything spatial (topographic head maps, interpolation).
Real recordings often lack them; this step applies a standard montage by channel name.
"""
from __future__ import annotations

from ..core.errors import StepError
from ..core.params import Choice
from ..core.registry import register
from ..core.step import Step


@register
class SetMontage(Step):
    id = "set_montage"
    name = "Set montage"
    category = "Import & setup"
    modalities = ["eeg"]
    params = [
        Choice(
            "montage",
            "colin27_1005",
            options=["colin27_1005", "colin27_1020", "biosemi64", "easycap-M1"],
            label="Montage",
        ),
    ]

    def check(self, ds) -> None:
        import mne

        if not isinstance(ds.payload, mne.io.BaseRaw):
            raise StepError("Set montage needs continuous (Raw) data.")

    def run(self, ds, p):
        from .._mne_compat import resolve_montage_name

        raw = ds.payload.copy()
        name = resolve_montage_name(p["montage"])
        raw.set_montage(name, match_case=False, on_missing="warn", verbose=False)
        return ds.derive(raw)
