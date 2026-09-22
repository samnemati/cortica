"""Motion-artifact correction for fNIRS that MNE does not provide directly.

Two published methods, implemented on plain ``(n_channels, n_times)`` arrays so they
are testable without MNE and reused by the pipeline steps:

* wavelet correction (Molavi & Dumont, 2012): decompose each channel, zero the detail
  coefficients that are statistical outliers (motion spikes), then reconstruct.
* spline-interpolation correction (in the spirit of Scholkmann et al., 2010): detect
  motion samples from a robust threshold on the temporal derivative and replace them by
  cubic-spline interpolation from the surrounding clean signal.
"""
from __future__ import annotations

import numpy as np


def wavelet_motion_correction(data, wavelet="db2", level=None, iqr_factor=1.5):
    """Zero wavelet detail coefficients beyond ``iqr_factor`` IQRs of their
    distribution (motion spikes), per channel. ``data`` is ``(n_channels, n_times)``.
    """
    import pywt

    out = np.array(data, dtype=float)
    for ch in range(out.shape[0]):
        signal = out[ch]
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        cleaned = [coeffs[0]]  # keep the approximation (slow) coefficients
        for detail in coeffs[1:]:
            q1, q3 = np.percentile(detail, [25, 75])
            iqr = q3 - q1
            lo, hi = q1 - iqr_factor * iqr, q3 + iqr_factor * iqr
            trimmed = detail.copy()
            trimmed[(trimmed < lo) | (trimmed > hi)] = 0.0
            cleaned.append(trimmed)
        reconstructed = pywt.waverec(cleaned, wavelet)
        out[ch] = reconstructed[: signal.shape[0]]
    return out


def spline_motion_correction(data, threshold=5.0):
    """Detect motion samples (robust z-score of the temporal derivative above
    ``threshold``) and replace them by cubic-spline interpolation from clean samples,
    per channel. ``data`` is ``(n_channels, n_times)``.
    """
    from scipy.interpolate import CubicSpline

    out = np.array(data, dtype=float)
    n = out.shape[1]
    index = np.arange(n)
    for ch in range(out.shape[0]):
        signal = out[ch]
        derivative = np.abs(np.diff(signal, prepend=signal[0]))
        mad = np.median(np.abs(derivative - np.median(derivative))) * 1.4826 + 1e-15
        artifact = derivative > (np.median(derivative) + threshold * mad)
        clean = ~artifact
        if artifact.any() and clean.sum() > 3:
            spline = CubicSpline(index[clean], signal[clean])
            repaired = signal.copy()
            repaired[artifact] = spline(index[artifact])
            out[ch] = repaired
    return out
