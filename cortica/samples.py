"""Built-in synthetic sample recordings.

Small, instant, offline datasets so first-time users (and demos) can try Cortica
without hunting for a file. These are illustrative signals, not real recordings.
MNE is imported lazily so importing this module does not require it.
"""
from __future__ import annotations

import numpy as np

from .core.dataset import Dataset
from .io import dataset_from_raw


def eeg_sample(seconds: float = 10.0, sfreq: float = 200.0) -> Dataset:
    """A synthetic 10-channel EEG recording (posterior channels show alpha)."""
    import mne

    n = int(seconds * sfreq)
    t = np.arange(n) / sfreq
    names = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]
    rng = np.random.RandomState(42)
    rows = []
    for name in names:
        alpha_amp = 1.6e-6 if name[0] in ("O", "P") else 0.5e-6
        signal = alpha_amp * np.sin(2 * np.pi * 10 * t) + rng.standard_normal(n) * 4e-7
        rows.append(signal)
    info = mne.create_info(names, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(np.vstack(rows), info, verbose=False)
    return dataset_from_raw(raw)


def fnirs_sample(seconds: float = 60.0, sfreq: float = 8.0) -> Dataset:
    """A synthetic 4-pair fNIRS recording as raw CW amplitude (two wavelengths each).

    Built as raw optical amplitude with optode geometry so the real fNIRS chain
    (optical density -> Beer-Lambert, SCI, TDDR) runs on it.
    """
    import mne

    n = int(seconds * sfreq)
    t = np.arange(n) / sfreq
    rng = np.random.RandomState(7)
    geometry = {}
    names, rows = [], []
    for pair in range(1, 5):
        key = f"S{pair}_D{pair}"
        source = np.array([0.03 * (pair - 1), 0.0, 0.0])
        detector = source + np.array([0.03, 0.0, 0.0])
        geometry[key] = (source, detector)
        hemodynamic = 0.5 * np.sin(2 * np.pi * 0.05 * t) + 1.0  # positive, slow drift
        for wavelength in (760, 850):
            names.append(f"{key} {wavelength}")
            rows.append(hemodynamic * 1e-3 + rng.standard_normal(n) * 1e-5)
    info = mne.create_info(names, sfreq, ch_types="fnirs_cw_amplitude")
    raw = mne.io.RawArray(np.vstack(rows), info, verbose=False)
    for ch in raw.info["chs"]:
        key, wavelength = ch["ch_name"].split(" ")
        source, detector = geometry[key]
        ch["loc"][0:3] = (source + detector) / 2
        ch["loc"][3:6] = source
        ch["loc"][6:9] = detector
        ch["loc"][9] = float(wavelength)  # MNE reads the wavelength from here
    return dataset_from_raw(raw)
