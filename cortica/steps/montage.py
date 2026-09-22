"""Sensor montage — assigns 3-D positions to channels.

Positions are required for anything spatial (topographic head maps, interpolation,
source localization). EEG files often lack them, so a standard montage is applied by
channel name; fNIRS files usually carry optode positions already, but a standard probe
or a custom layout can still be set here. A custom montage file (any format MNE's
``read_custom_montage`` supports: .tsv/.csv/.sfp/.elc/.bvef/.xyz/...) overrides the
built-in choice, so users can bring their own electrode or optode coordinates.
"""
from __future__ import annotations

from ..core.errors import StepError
from ..core.params import Choice, Str
from ..core.registry import register
from ..core.step import Step

#: Curated built-in montages (EEG standards + common fNIRS probes). Anything else can
#: be loaded from a custom file. ``standard_1005/1020`` are resolved to the installed
#: MNE's names (older MNE) or ``colin27_*`` (newer MNE) at apply time.
_BUILTIN_MONTAGES = [
    "standard_1005",
    "standard_1020",
    "biosemi16",
    "biosemi32",
    "biosemi64",
    "biosemi128",
    "GSN-HydroCel-64_1.0",
    "GSN-HydroCel-128",
    "GSN-HydroCel-256",
    "easycap-M1",
    "easycap-M10",
    "artinis-octamon",
    "artinis-brite23",
]


@register
class SetMontage(Step):
    id = "set_montage"
    name = "Set montage"
    category = "Import & setup"
    modalities = ["eeg", "fnirs"]
    params = [
        Choice("montage", "standard_1005", options=_BUILTIN_MONTAGES, label="Built-in montage"),
        Str("custom_file", "", label="Custom montage file (overrides built-in)"),
    ]

    def check(self, ds) -> None:
        import mne

        if not isinstance(ds.payload, mne.io.BaseRaw):
            raise StepError("Set montage needs continuous (Raw) data.")

    def run(self, ds, p):
        import mne

        from .._mne_compat import resolve_montage_name

        raw = ds.payload.copy()
        custom = (p.get("custom_file") or "").strip()
        if custom:
            montage = mne.channels.read_custom_montage(custom)
        else:
            montage = resolve_montage_name(p["montage"])
        raw.set_montage(montage, match_case=False, on_missing="warn", verbose=False)
        return ds.derive(raw)
