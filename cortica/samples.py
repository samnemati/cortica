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
    """A synthetic 4-pair fNIRS recording with HbO/HbR channels (~0.05 Hz waves)."""
    import mne

    n = int(seconds * sfreq)
    t = np.arange(n) / sfreq
    rng = np.random.RandomState(7)
    names, types, rows = [], [], []
    for pair in range(1, 5):
        hemo = np.sin(2 * np.pi * 0.05 * t)
        names.append(f"S{pair}_D{pair} hbo")
        types.append("hbo")
        rows.append(hemo * 1e-6 + rng.standard_normal(n) * 5e-8)
        names.append(f"S{pair}_D{pair} hbr")
        types.append("hbr")
        rows.append(-0.4 * hemo * 1e-6 + rng.standard_normal(n) * 5e-8)
    info = mne.create_info(names, sfreq, ch_types=types)
    raw = mne.io.RawArray(np.vstack(rows), info, verbose=False)
    return dataset_from_raw(raw)
