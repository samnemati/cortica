"""Preprocessing steps backed by MNE.

MNE is not imported here; each step calls methods on the ``Raw`` object carried by
the Dataset, so importing this module (and registering the steps) needs no MNE.
Steps copy the payload before transforming it, keeping the input immutable.
"""
from __future__ import annotations

from ..core.errors import StepError
from ..core.params import Float
from ..core.registry import register
from ..core.step import Step


@register
class BandpassFilter(Step):
    id = "bandpass_filter"
    name = "Band-pass filter"
    category = "Preprocess"
    modalities = ["eeg", "fnirs"]
    params = [
        Float("l_freq", 1.0, min=0.0, unit="Hz", label="Low cutoff"),
        Float("h_freq", 40.0, min=0.0, unit="Hz", label="High cutoff"),
    ]

    def run(self, ds, p):
        raw = ds.payload.copy().filter(p["l_freq"], p["h_freq"], verbose=False)
        return ds.derive(raw)


@register
class Resample(Step):
    id = "resample"
    name = "Resample"
    category = "Preprocess"
    modalities = ["eeg", "fnirs"]
    params = [Float("sfreq", 250.0, min=1.0, unit="Hz", label="New sampling rate")]

    def run(self, ds, p):
        raw = ds.payload.copy().resample(p["sfreq"], verbose=False)
        result = ds.derive(raw)
        result.meta["sfreq"] = float(raw.info["sfreq"])
        return result


@register
class NotchFilter(Step):
    id = "notch_filter"
    name = "Notch filter"
    category = "Preprocess"
    modalities = ["eeg"]
    params = [Float("freq", 60.0, min=1.0, unit="Hz", label="Line frequency")]

    def run(self, ds, p):
        nyquist = ds.payload.info["sfreq"] / 2
        freqs, harmonic = [], p["freq"]
        while harmonic < nyquist:
            freqs.append(harmonic)
            harmonic += p["freq"]
        if not freqs:
            raise StepError(f"Notch frequency {p['freq']} Hz is at/above Nyquist ({nyquist} Hz).")
        raw = ds.payload.copy().notch_filter(freqs, verbose=False)
        return ds.derive(raw)


@register
class AverageReference(Step):
    id = "reref_average"
    name = "Re-reference (average)"
    category = "Preprocess"
    modalities = ["eeg"]
    params = []

    def run(self, ds, p):
        raw = ds.payload.copy()
        raw.set_eeg_reference("average", projection=False, verbose=False)
        return ds.derive(raw)


@register
class InterpolateBads(Step):
    id = "interpolate_bads"
    name = "Interpolate bad channels"
    category = "Artifacts & quality"
    modalities = ["eeg"]
    params = []

    def run(self, ds, p):
        raw = ds.payload.copy()
        raw.interpolate_bads(reset_bads=True, verbose=False)
        return ds.derive(raw)
