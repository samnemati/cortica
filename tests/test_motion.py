"""Tests for the fNIRS motion-correction algorithms (pure array operations)."""
import numpy as np

from cortica._motion import spline_motion_correction, wavelet_motion_correction


def test_wavelet_attenuates_a_motion_spike():
    rng = np.random.RandomState(0)
    data = rng.standard_normal((2, 256)) * 0.1
    data[0, 128] += 10.0  # a large motion spike
    out = wavelet_motion_correction(data)
    assert out.shape == data.shape
    assert abs(out[0, 128]) < abs(data[0, 128])  # the spike is reduced


def test_spline_smooths_an_abrupt_jump():
    rng = np.random.RandomState(1)
    data = rng.standard_normal((1, 200)) * 0.02
    data[0, 100:] += 5.0  # a step (baseline shift) artifact
    out = spline_motion_correction(data)
    before = abs(data[0, 100] - data[0, 99])
    after = abs(out[0, 100] - out[0, 99])
    assert after < before  # the abrupt jump is smoothed


def test_correction_leaves_clean_data_close_to_unchanged():
    rng = np.random.RandomState(2)
    data = rng.standard_normal((3, 128)) * 0.05
    out = wavelet_motion_correction(data)
    assert out.shape == data.shape
    assert np.corrcoef(out[0], data[0])[0, 1] > 0.5  # broadly preserved, no spike removed
