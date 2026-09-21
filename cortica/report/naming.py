"""Build a descriptive, non-clobbering default file name for a report.

The name summarises the pipeline (short step abbreviations) and stamps the date and
time, so successive exports never overwrite one another and the file says what it is
at a glance. An optional user label is slugified and placed up front.
"""
from __future__ import annotations

import datetime
import re

#: Short, human-readable abbreviations for step ids used in the file name.
_STEP_ABBR = {
    "set_montage": "mtg",
    "bandpass_filter": "bp",
    "notch_filter": "notch",
    "resample": "rs",
    "reref": "ref",
    "interpolate_bads": "interp",
    "ica": "ica",
    "optical_density": "od",
    "beer_lambert": "bl",
    "scalp_coupling_index": "sci",
    "tddr": "tddr",
    "epochs_fixed": "ep",
    "epochs_events": "ep",
    "average": "avg",
}


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:40]


def default_report_name(pipeline, label: str | None = None, now=None) -> str:
    """Return a file name like ``cortica-eeg-bp-avg-20260921-153000.html``.

    ``label`` (optional) is slugified and inserted after ``cortica``. ``now`` is for
    testing; it defaults to the current local time.
    """
    now = now or datetime.datetime.now()
    parts = ["cortica"]
    if label:
        slug = _slugify(label)
        if slug:
            parts.append(slug)
    modality = getattr(pipeline, "modality", None)
    if modality:
        parts.append(modality)
    abbrs: list[str] = []
    for pstep in getattr(pipeline, "steps", []):
        abbr = _STEP_ABBR.get(pstep.step_id, pstep.step_id)
        if abbr not in abbrs:
            abbrs.append(abbr)
    if abbrs:
        parts.append("-".join(abbrs))
    parts.append(now.strftime("%Y%m%d-%H%M%S"))
    return "-".join(parts) + ".html"
