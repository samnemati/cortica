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
                "Scalp coupling index needs optical-density data. Add Optical density first."
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
            raise StepError("TDDR needs optical-density data. Add Optical density first.")

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
                "Beer-Lambert needs optical-density data. Add Optical density first."
            )

    def run(self, ds, p):
        from mne.preprocessing.nirs import beer_lambert_law

        return ds.derive(beer_lambert_law(ds.payload.copy(), ppf=p["ppf"]))


@register
class ShortChannelRegression(Step):
    id = "short_channel_regression"
    name = "Short-channel regression"
    category = "Preprocess"
    modalities = ["fnirs"]
    params = []

    def check(self, ds) -> None:
        if "fnirs_od" not in _channel_types(ds.payload):
            raise StepError(
                "Short-channel regression needs optical-density data. Add Optical density first."
            )
        from mne_nirs.channels import get_short_channels

        try:
            has_short = len(get_short_channels(ds.payload).ch_names) > 0
        except Exception:
            has_short = False
        if not has_short:
            raise StepError(
                "Short-channel regression needs short-separation channels in the montage."
            )

    def run(self, ds, p):
        from mne_nirs.signal_enhancement import short_channel_regression

        return ds.derive(short_channel_regression(ds.payload.copy()))


@register
class EnhanceNegativeCorrelation(Step):
    id = "enhance_negative_correlation"
    name = "Enhance HbO/HbR anti-correlation"
    category = "Preprocess"
    modalities = ["fnirs"]
    params = []

    def check(self, ds) -> None:
        if not ({"hbo", "hbr"} & _channel_types(ds.payload)):
            raise StepError(
                "Enhancing anti-correlation needs haemoglobin data. Add Beer-Lambert law first."
            )

    def run(self, ds, p):
        from mne_nirs.signal_enhancement import enhance_negative_correlation

        return ds.derive(enhance_negative_correlation(ds.payload.copy()))


@register
class KeepLongChannels(Step):
    id = "keep_long_channels"
    name = "Keep long channels"
    category = "Artifacts & quality"
    modalities = ["fnirs"]
    params = []

    def check(self, ds) -> None:
        if not (_channel_types(ds.payload) & {"fnirs_cw_amplitude", "fnirs_od", "hbo", "hbr"}):
            raise StepError("Keep long channels needs fNIRS data.")

    def run(self, ds, p):
        from mne_nirs.channels import get_long_channels

        return ds.derive(get_long_channels(ds.payload.copy()))
