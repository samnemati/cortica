"""fNIRS-specific steps (MNE-backed).

The typical fNIRS chain is: raw CW amplitude -> optical density -> (quality/motion
handling) -> Beer-Lambert -> HbO/HbR. Each step checks its input channel type so
using them out of order fails with a clear message rather than an opaque MNE error.
"""
from __future__ import annotations

from ..core.errors import StepError
from ..core.params import Float
from ..core.registry import register
from ..core.step import Step


def _channel_types(payload) -> set:
    try:
        return set(payload.get_channel_types())
    except Exception:
        return set()


@register
class OpticalDensity(Step):
    id = "optical_density"
    name = "Optical density"
    category = "Preprocess"
    modalities = ["fnirs"]
    params = []

    def check(self, ds) -> None:
        if "fnirs_cw_amplitude" not in _channel_types(ds.payload):
            raise StepError("Optical density needs raw fNIRS (CW amplitude) data.")

    def run(self, ds, p):
        from mne.preprocessing.nirs import optical_density

        return ds.derive(optical_density(ds.payload, verbose=False))


@register
class ScalpCouplingIndex(Step):
    id = "scalp_coupling_index"
    name = "Scalp coupling index"
    category = "Artifacts & quality"
    modalities = ["fnirs"]
    params = [Float("threshold", 0.7, min=0.0, max=1.0, label="SCI threshold")]

    def check(self, ds) -> None:
        if "fnirs_od" not in _channel_types(ds.payload):
            raise StepError(
                "Scalp coupling index needs optical-density data — add Optical density first."
            )

    def run(self, ds, p):
        from mne.preprocessing.nirs import scalp_coupling_index

        raw = ds.payload.copy()
        sci = scalp_coupling_index(raw, verbose=False)
        bad = [name for name, value in zip(raw.ch_names, sci) if value < p["threshold"]]
        raw.info["bads"] = sorted(set(raw.info["bads"]) | set(bad))
        return ds.derive(raw)


@register
class Tddr(Step):
    id = "tddr"
    name = "Motion correction (TDDR)"
    category = "Artifacts & quality"
    modalities = ["fnirs"]
    params = []

    def check(self, ds) -> None:
        if "fnirs_od" not in _channel_types(ds.payload):
            raise StepError("TDDR needs optical-density data — add Optical density first.")

    def run(self, ds, p):
        from mne.preprocessing.nirs import temporal_derivative_distribution_repair

        return ds.derive(temporal_derivative_distribution_repair(ds.payload.copy(), verbose=False))


@register
class BeerLambert(Step):
    id = "beer_lambert"
    name = "Beer–Lambert law"
    category = "Preprocess"
    modalities = ["fnirs"]
    params = [Float("ppf", 6.0, min=0.1, label="Partial pathlength factor")]

    def check(self, ds) -> None:
        if "fnirs_od" not in _channel_types(ds.payload):
            raise StepError(
                "Beer–Lambert needs optical-density data — add Optical density first."
            )

    def run(self, ds, p):
        from mne.preprocessing.nirs import beer_lambert_law

        return ds.derive(beer_lambert_law(ds.payload.copy(), ppf=p["ppf"]))
