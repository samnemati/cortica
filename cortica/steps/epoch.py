"""Segmenting and averaging steps (MNE-backed).

Fixed-length epoching cuts continuous data into equal segments — it needs no event
channel, so it works on any recording. Averaging collapses epochs into an evoked
response. Each step checks its input type so misuse fails with a clear message.
"""
from __future__ import annotations

from ..core.errors import StepError
from ..core.params import Float
from ..core.registry import register
from ..core.step import Step


@register
class FixedLengthEpochs(Step):
    id = "epochs_fixed"
    name = "Epochs (fixed length)"
    category = "Segment"
    modalities = ["eeg", "fnirs"]
    params = [
        Float("duration", 2.0, min=0.1, unit="s", label="Epoch length"),
        Float("overlap", 0.0, min=0.0, unit="s", label="Overlap"),
    ]

    def check(self, ds) -> None:
        import mne

        if not isinstance(ds.payload, mne.io.BaseRaw):
            raise StepError("Epoching needs continuous (Raw) data.")

    def run(self, ds, p):
        import mne

        epochs = mne.make_fixed_length_epochs(
            ds.payload, duration=p["duration"], overlap=p["overlap"],
            preload=True, verbose=False,
        )
        return ds.derive(epochs)


@register
class EventEpochs(Step):
    id = "epochs_events"
    name = "Epochs (by events)"
    category = "Segment"
    modalities = ["eeg", "fnirs"]
    params = [
        Float("tmin", -0.2, unit="s", label="Start (rel. event)"),
        Float("tmax", 0.8, unit="s", label="End (rel. event)"),
    ]

    def check(self, ds) -> None:
        import mne

        if not isinstance(ds.payload, mne.io.BaseRaw):
            raise StepError("Event-based epoching needs continuous (Raw) data.")
        if len(ds.payload.annotations) == 0:
            raise StepError(
                "No events found — this recording has no annotations/triggers to epoch on."
            )

    def run(self, ds, p):
        import mne

        events, event_id = mne.events_from_annotations(ds.payload, verbose=False)
        baseline = (None, 0) if p["tmin"] < 0 else None
        epochs = mne.Epochs(
            ds.payload, events, event_id=event_id,
            tmin=p["tmin"], tmax=p["tmax"], baseline=baseline,
            preload=True, verbose=False,
        )
        return ds.derive(epochs)


@register
class Average(Step):
    id = "average"
    name = "Average (evoked)"
    category = "Analyze"
    modalities = ["eeg", "fnirs"]
    params = []

    def check(self, ds) -> None:
        import mne

        if not isinstance(ds.payload, mne.BaseEpochs):
            raise StepError("Averaging needs epoched data — add an Epochs step first.")

    def run(self, ds, p):
        return ds.derive(ds.payload.average())
