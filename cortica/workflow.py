"""Guided-workflow logic (pure, Qt-free).

Given the current dataset and pipeline, decide which workflow stage the user is in
and what the sensible next actions are. The GUI renders this as a left-rail guide;
keeping the logic here means it is unit-tested without a display, and stays honest
to the actual data state rather than a hard-coded wizard.

A :class:`Suggestion` names something the user can do next:

* ``kind="step"``   — add this pipeline step (``target`` is a step id)
* ``kind="view"``   — switch the viewer to this mode (``target`` is a view mode)
* ``kind="action"`` — trigger a toolbar action (``target`` in open/ica/source/preview)
"""
from __future__ import annotations

from dataclasses import dataclass

#: Ordered workflow stages as ``(key, display label)``.
STAGES = [
    ("import", "Import"),
    ("preprocess", "Preprocess"),
    ("segment", "Segment"),
    ("analyze", "Analyze"),
    ("report", "Report"),
]

#: Which stage each data kind puts you in.
_KIND_STAGE = {
    "none": "import",
    "raw": "preprocess",
    "epochs": "analyze",
    "evoked": "analyze",
    "other": "analyze",
}


@dataclass(frozen=True)
class Suggestion:
    label: str
    kind: str  # "step" | "view" | "action"
    target: str
    reason: str


def _payload_kind(payload) -> str:
    if payload is None:
        return "none"
    import mne

    if isinstance(payload, mne.io.BaseRaw):
        return "raw"
    if isinstance(payload, mne.BaseEpochs):
        return "epochs"
    if isinstance(payload, mne.Evoked):
        return "evoked"
    return "other"


def _has_montage(payload) -> bool:
    try:
        return payload.get_montage() is not None
    except Exception:
        return False


def current_stage(dataset, pipeline=None) -> str:
    """Return the stage key for the dataset's current data kind."""
    return _KIND_STAGE.get(_payload_kind(getattr(dataset, "payload", None)), "import")


def stage_status(dataset, pipeline=None):
    """Return ``[(key, label, status), ...]`` with status ``done``/``current``/``todo``."""
    keys = [key for key, _ in STAGES]
    cur_idx = keys.index(current_stage(dataset, pipeline))
    rows = []
    for i, (key, label) in enumerate(STAGES):
        status = "done" if i < cur_idx else ("current" if i == cur_idx else "todo")
        rows.append((key, label, status))
    return rows


def suggestions(dataset, pipeline=None):
    """Return the recommended next :class:`Suggestion` list for the current state."""
    payload = getattr(dataset, "payload", None)
    kind = _payload_kind(payload)
    if kind == "none":
        return [
            Suggestion(
                "Load a recording", "action", "open",
                "Open your EEG or fNIRS file, or load a sample to explore.",
            )
        ]
    modality = getattr(dataset, "modality", "eeg")
    have = {s.step_id for s in getattr(pipeline, "steps", [])} if pipeline else set()

    if kind == "raw" and modality == "fnirs":
        out = _fnirs_raw_suggestions(have)
    elif kind == "raw":
        out = _eeg_raw_suggestions(payload, have)
    elif kind == "epochs":
        out = [
            Suggestion("Average (evoked)", "step", "average",
                       "Average epochs into an evoked response (ERP/ERF)."),
            Suggestion("Compare conditions", "view", "compare",
                       "Overlay each condition's ERP and their difference wave."),
            Suggestion("Decoding", "view", "decoding",
                       "Test whether the conditions are separable over time (MVPA)."),
            Suggestion("Connectivity", "view", "conn",
                       "Measure functional coupling between channels."),
        ]
    elif kind == "evoked":
        out = [
            Suggestion("Localize sources…", "action", "source",
                       "Estimate the cortical origin on the fsaverage template."),
            Suggestion("Preview report", "action", "preview",
                       "Open the shareable HTML report in your browser."),
        ]
    else:
        out = []
    # Drop suggestions whose target is already in the pipeline: added steps, and the
    # "Fit ICA…" action once an ICA step exists (its target "ica" is a step id too).
    # View/action targets like conn/decoding/open/source are never step ids, so kept.
    return [s for s in out if s.target not in have]


def _eeg_raw_suggestions(payload, have):
    out = []
    if not _has_montage(payload) and "set_montage" not in have:
        out.append(Suggestion("Set montage", "step", "set_montage",
                              "Electrode positions unlock head maps and source localization."))
    if "bandpass_filter" not in have:
        out.append(Suggestion("Band-pass filter", "step", "bandpass_filter",
                              "Keep the frequencies you care about (e.g. 1–40 Hz)."))
    out.append(Suggestion("Fit ICA…", "action", "ica",
                          "Find and remove blink or heartbeat components."))
    out.append(Suggestion("Epochs (by events)", "step", "epochs_events",
                          "Cut the recording around event markers to build ERPs."))
    return out


def _fnirs_raw_suggestions(have):
    out = []
    if "optical_density" not in have:
        out.append(Suggestion(
            "Optical density", "step", "optical_density",
            "Convert raw light intensity to optical density; the first fNIRS step.",
        ))
    else:
        if "scalp_coupling_index" not in have:
            out.append(Suggestion("Scalp coupling index", "step", "scalp_coupling_index",
                                  "Flag poorly-coupled optode channels before going further."))
        if "beer_lambert" not in have:
            out.append(Suggestion("Beer–Lambert law", "step", "beer_lambert",
                                  "Convert optical density to HbO/HbR concentrations."))
    out.append(Suggestion("Epochs (by events)", "step", "epochs_events",
                          "Segment around task events."))
    return out
