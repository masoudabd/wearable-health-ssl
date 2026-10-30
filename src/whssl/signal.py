"""Signal utilities with no torch dependency (data loaders and analysis scripts import only this)."""
from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly


def resample(x: np.ndarray, fs_in: float, fs_out: float, axis: int = 0) -> np.ndarray:
    """Polyphase resampling with anti-aliasing (handles 64, 80, 85.7, 100, 103.4, 128 Hz inputs)."""
    if fs_in == fs_out:
        return x
    frac = Fraction(fs_out / fs_in).limit_denominator(1000)
    return resample_poly(x, frac.numerator, frac.denominator, axis=axis)


def gait_features(W: np.ndarray, fs: float, f_lo: float = 0.5, f_hi: float = 3.0, half_band: float = 0.3):
    """Orientation-free walking features of windows W (n, T, 3) in g, from the acceleration magnitude:
    sd   standard deviation of the magnitude (g)
    f    dominant frequency of the detrended magnitude in [f_lo, f_hi] Hz (step or stride rhythm)
    conc share of the 0.3 to 10 Hz power within +-half_band Hz of f and of its first harmonic (rhythmicity)"""
    m = np.linalg.norm(W, axis=2)
    sd = m.std(axis=1)
    x = (m - m.mean(axis=1, keepdims=True)) * np.hanning(m.shape[1])
    nfft = 512
    P = np.abs(np.fft.rfft(x, n=nfft, axis=1)) ** 2
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    band = np.flatnonzero((freqs >= f_lo) & (freqs <= f_hi))
    f = freqs[band[np.argmax(P[:, band], axis=1)]]
    total = P[:, (freqs >= 0.3) & (freqs <= 10.0)].sum(axis=1)
    near = (np.abs(freqs[None, :] - f[:, None]) <= half_band) | (np.abs(freqs[None, :] - 2 * f[:, None]) <= half_band)
    conc = (P * near).sum(axis=1) / np.maximum(total, 1e-12)
    return sd, f, conc


def gait_like(W: np.ndarray, fs: float, min_sd: float, min_conc: float) -> np.ndarray:
    """Boolean mask of walking-like windows: enough movement and a dominant rhythm (thresholds validated on HAR70+,
    scripts/validate_gait_filter.py)."""
    sd, _, conc = gait_features(W, fs)
    return (sd >= min_sd) & (conc >= min_conc)
