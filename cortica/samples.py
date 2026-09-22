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
    # stimulus events (oddball-style: mostly "standard", occasional "target")
    onsets = np.arange(1.0, seconds - 0.5, 1.0)
    labels = ["target" if i % 4 == 2 else "standard" for i in range(len(onsets))]
    raw.set_annotations(mne.Annotations(onsets, np.zeros(len(onsets)), labels))
    # 10-05 positions (the channel names above are 10-20 sites) so head maps work
    # on the sample out of the box.
    from ._mne_compat import resolve_montage_name

    raw.set_montage(
        resolve_montage_name("colin27_1005"),
        match_case=False, on_missing="ignore", verbose=False,
    )
    return dataset_from_raw(raw)


def fnirs_sample(seconds: float = 60.0, sfreq: float = 8.0) -> Dataset:
    """A synthetic fNIRS recording as raw CW amplitude (two wavelengths each).

    Built as raw optical amplitude with optode geometry so the full fNIRS chain runs
    on it: four long (3 cm) source-detector pairs plus one short-separation (0.8 cm)
    pair that carries only systemic signal, and alternating Task/Control blocks so the
    GLM and short-channel regression have something to work with.
    """
    import mne

    n = int(seconds * sfreq)
    t = np.arange(n) / sfreq
    rng = np.random.RandomState(7)

    # Task design: alternating Task/Control blocks.
    block = 4.0
    onsets = np.arange(5.0, seconds - block, 7.0)
    labels = ["Task" if i % 2 == 0 else "Control" for i in range(len(onsets))]
    task = np.zeros(n)
    for onset, label in zip(onsets, labels):
        if label == "Task":
            task[int(onset * sfreq) : int((onset + block) * sfreq)] = 1.0

    # Systemic signal shared by all channels (slow Mayer wave + cardiac-band term).
    systemic = 0.5 * np.sin(2 * np.pi * 0.05 * t) + 0.1 * np.sin(2 * np.pi * 1.0 * t)

    geometry = {}
    names, rows = [], []
    for pair in range(1, 5):  # four long pairs (3 cm): systemic + task response
        key = f"S{pair}_D{pair}"
        source = np.array([0.03 * (pair - 1), 0.0, 0.0])
        detector = source + np.array([0.03, 0.0, 0.0])
        geometry[key] = (source, detector)
        base = 1.0 + 0.5 * systemic + 0.4 * task
        for wavelength in (760, 850):
            names.append(f"{key} {wavelength}")
            rows.append(base * 1e-3 + rng.standard_normal(n) * 1e-5)
    # One short pair (0.8 cm): systemic only, no task response.
    short_key = "S5_D5"
    short_source = np.array([0.0, 0.02, 0.0])
    geometry[short_key] = (short_source, short_source + np.array([0.008, 0.0, 0.0]))
    for wavelength in (760, 850):
        names.append(f"{short_key} {wavelength}")
        rows.append((1.0 + 0.5 * systemic) * 1e-3 + rng.standard_normal(n) * 1e-5)

    info = mne.create_info(names, sfreq, ch_types="fnirs_cw_amplitude")
    raw = mne.io.RawArray(np.vstack(rows), info, verbose=False)
    for ch in raw.info["chs"]:
        key, wavelength = ch["ch_name"].split(" ")
        source, detector = geometry[key]
        ch["loc"][0:3] = (source + detector) / 2
        ch["loc"][3:6] = source
        ch["loc"][6:9] = detector
        ch["loc"][9] = float(wavelength)  # MNE reads the wavelength from here
    raw.set_annotations(mne.Annotations(onsets, [block] * len(onsets), labels))
    return dataset_from_raw(raw)
