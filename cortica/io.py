"""Load recordings into a :class:`~cortica.core.dataset.Dataset`.

Loading is separate from the pipeline: the CLI/GUI loads an input file into a
Dataset, then the pipeline transforms it. This keeps pipelines re-runnable on new
inputs (the input path is supplied at run time, not baked into the pipeline).

MNE is imported lazily inside :func:`load_raw`, so importing this module does not
require MNE. :func:`dataset_from_raw` only calls methods on the object passed in.
"""
from __future__ import annotations

from .core.dataset import Dataset

# MNE channel types that identify an fNIRS recording.
_FNIRS_TYPES = {
    "fnirs_cw_amplitude",
    "fnirs_od",
    "hbo",
    "hbr",
    "fnirs_fd_ac_amplitude",
    "fnirs_fd_phase",
}


def dataset_from_raw(raw) -> Dataset:
    """Wrap an MNE ``Raw`` in a Dataset, detecting modality from channel types."""
    ch_types = set(raw.get_channel_types())
    modality = "fnirs" if ch_types & _FNIRS_TYPES else "eeg"
    meta = {
        "sfreq": float(raw.info["sfreq"]),
        "n_channels": len(raw.ch_names),
    }
    return Dataset(raw, modality, meta)


def load_raw(path: str) -> Dataset:
    """Read a recording from disk (format inferred from the extension)."""
    import mne

    raw = mne.io.read_raw(path, preload=True, verbose=False)
    return dataset_from_raw(raw)
