"""The vectorized feat60 must reproduce the scalar per-window feature code it was ported from."""
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whssl.features import FEAT20, feat60, freeze_index_features


def _reference_feat20(x, fs):
    """Scalar reference: the original per-window time and spectral feature code, condensed."""
    x = np.asarray(x, float)
    n = x.size
    std = float(np.std(x, ddof=1))
    med = float(np.median(x))
    q25, q75 = float(np.percentile(x, 25)), float(np.percentile(x, 75))
    s = np.sign(x)
    s[s == 0] = 1
    if np.allclose(x, x[0]):
        spec = (0.0, 0.0, 0.0, 0.0)
    else:
        mag = np.abs(np.fft.rfft(x * np.hanning(n)))
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        dom = freqs[1 + int(np.argmax(mag[1:]))]
        p = mag / (mag.sum() + 1e-12)
        ent = float(-(p * np.log2(p + 1e-12)).sum())
        cen = float((freqs * p).sum())
        spec = (dom, cen, float(np.sqrt(((freqs - cen) ** 2 * p).sum())), ent)
    return [float(np.mean(x)), std, float(np.var(x, ddof=1)), float(np.sqrt(np.mean(x ** 2))), med,
            float(np.min(x)), float(np.max(x)), float(np.ptp(x)), q25, q75, q75 - q25,
            float(np.median(np.abs(x - med))),
            float(stats.skew(x, bias=False)) if std > 0 else 0.0,
            float(stats.kurtosis(x, fisher=True, bias=False)) if std > 0 else 0.0,
            float(np.mean(s[:-1] != s[1:])), float(np.sqrt(np.mean((np.diff(x) * fs) ** 2))), *spec]


def test_feat60_matches_reference():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4, 150, 3))
    X[1, :, 0] += np.sin(np.arange(150) / 30 * 2 * math.pi * 5)  # a 5 Hz tremor-like component
    X[2, :, 1] = 0.7  # constant axis
    F, cols = feat60(X, 30.0)
    assert F.shape == (4, 60) and len(cols) == 60
    for w in range(4):
        for a in range(3):
            ref = np.nan_to_num(np.array(_reference_feat20(X[w, :, a], 30.0)))
            np.testing.assert_allclose(F[w, a * 20:(a + 1) * 20], ref, rtol=1e-9, atol=1e-9,
                                       err_msg=f"window {w} axis {a}: {FEAT20}")


def test_freeze_index_rises_with_tremor_band_power():
    t = np.arange(150) / 30.0
    walk = np.sin(2 * math.pi * 1.5 * t)          # locomotor band
    freeze = np.sin(2 * math.pi * 5.0 * t)        # freeze band
    X = np.stack([np.stack([walk] * 3, -1), np.stack([freeze] * 3, -1)])
    F, cols = freeze_index_features(X, 30.0)
    fi = F[:, cols.index("V_log_freeze_index")]
    assert fi[1] > fi[0] + 5
