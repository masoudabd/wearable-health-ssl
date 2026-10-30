"""Handcrafted window features, vectorized over windows.

feat60: 60 window features, 20 per axis: mean, std, var, rms, median, min, max, ptp, q25, q75,
iqr, mad, skew, kurtosis, zcr, jerk_rms, dom_freq, spec_centroid, spec_bandwidth, spec_entropy.
Definitions match a scalar per-window reference implementation (checked in tests/test_features.py).

freeze_index_features: per axis, log power in the locomotor band (0.5-3 Hz), log power in the freeze
band (3-8 Hz), and log freeze index (their ratio), after Moore et al., J Neurosci Methods 2008 and
Baechlin et al., IEEE TITB 2010.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy import stats

FEAT20 = ["mean", "std", "var", "rms", "median", "min", "max", "ptp", "q25", "q75", "iqr", "mad",
          "skew", "kurtosis", "zcr", "jerk_rms", "dom_freq", "spec_centroid", "spec_bandwidth", "spec_entropy"]


def _feat20(x: np.ndarray, fs: float) -> np.ndarray:
    """x: (N, T) one axis. Returns (N, 20)."""
    n = x.shape[1]
    mean = x.mean(1)
    std = x.std(1, ddof=1)
    var = x.var(1, ddof=1)
    rms = np.sqrt((x ** 2).mean(1))
    med = np.median(x, 1)
    mn, mx = x.min(1), x.max(1)
    q25, q75 = np.percentile(x, 25, axis=1), np.percentile(x, 75, axis=1)
    mad = np.median(np.abs(x - med[:, None]), 1)
    const = std <= 0
    with np.errstate(all="ignore"):
        skew = np.where(const, 0.0, stats.skew(x, axis=1, bias=False))
        kurt = np.where(const, 0.0, stats.kurtosis(x, axis=1, fisher=True, bias=False))
    s = np.sign(x)
    s[s == 0] = 1
    zcr = (s[:, :-1] != s[:, 1:]).mean(1)
    jerk = np.sqrt(((np.diff(x, axis=1) * fs) ** 2).mean(1))

    mag = np.abs(np.fft.rfft(x * np.hanning(n)[None, :], axis=1))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    dom = freqs[1 + np.argmax(mag[:, 1:], axis=1)]
    p = mag / (mag.sum(1, keepdims=True) + 1e-12)
    ent = -(p * np.log2(p + 1e-12)).sum(1)
    cen = (freqs[None, :] * p).sum(1)
    bw = np.sqrt((((freqs[None, :] - cen[:, None]) ** 2) * p).sum(1))
    flat = np.isclose(x, x[:, :1]).all(1) | (mag.sum(1) == 0)
    dom, cen, bw, ent = [np.where(flat, 0.0, v) for v in (dom, cen, bw, ent)]
    out = np.stack([mean, std, var, rms, med, mn, mx, mx - mn, q25, q75, q75 - q25, mad,
                    skew, kurt, zcr, jerk, dom, cen, bw, ent], axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def feat60(X: np.ndarray, fs: float, axes: Tuple[str, ...] = ("V", "ML", "AP")) -> Tuple[np.ndarray, List[str]]:
    """X: (N, T, 3). Returns (N, 60) and column names."""
    X = np.asarray(X, dtype=np.float64)
    cols = [f"{a}_{f}" for a in axes for f in FEAT20]
    return np.concatenate([_feat20(X[:, :, i], fs) for i in range(X.shape[2])], axis=1), cols


def freeze_index_features(X: np.ndarray, fs: float, axes: Tuple[str, ...] = ("V", "ML", "AP"),
                          loco: Tuple[float, float] = (0.5, 3.0), freeze: Tuple[float, float] = (3.0, 8.0)):
    """X: (N, T, 3). Returns (N, 9): per axis log locomotor power, log freeze power, log freeze index."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[1]
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    lo = (freqs >= loco[0]) & (freqs < loco[1])
    fr = (freqs >= freeze[0]) & (freqs <= freeze[1])
    feats, cols = [], []
    for i, a in enumerate(axes):
        x = X[:, :, i] - X[:, :, i].mean(1, keepdims=True)
        P = np.abs(np.fft.rfft(x * np.hanning(n)[None, :], axis=1)) ** 2
        pl, pf = P[:, lo].sum(1) + 1e-10, P[:, fr].sum(1) + 1e-10
        feats += [np.log(pl), np.log(pf), np.log(pf / pl)]
        cols += [f"{a}_logpow_loco", f"{a}_logpow_freeze", f"{a}_log_freeze_index"]
    return np.stack(feats, axis=1), cols
