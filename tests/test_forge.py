"""Rebuilt FORGE encoder: filters as designed, strict weight load, invariance to axis sign and scale."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
WEIGHTS = ROOT / "data" / "raw" / "forge" / "model" / "encoders" / "mc.safetensors"
pytestmark = pytest.mark.skipif(not WEIGHTS.exists(), reason="FORGE weights not downloaded")


def test_forge_filters_tokens_and_invariance():
    from scipy.signal import butter, sos2tf

    from whssl.models.forge import load_forge

    enc = load_forge("mc")
    for (b, a), cut, kind in [((enc.hp_b, enc.hp_a), 0.5, "high"), ((enc.lp_b, enc.lp_a), 10.0, "low")]:
        bb, aa = sos2tf(butter(4, cut, btype=kind, fs=100, output="sos"))
        np.testing.assert_allclose(b, bb, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(a, aa, rtol=1e-5, atol=1e-7)
    x = np.random.default_rng(0).normal(size=(4, 3, 500)).astype(np.float32)
    tok = enc.tokens(x)
    assert tuple(tok.shape) == (4, 50, 512)
    flipped = enc.tokens(np.ascontiguousarray(x * np.array([1, -1, -1], np.float32)[None, :, None]))
    assert float((flipped - tok).abs().max()) == 0.0
    scaled = enc.tokens(x * 9.80665)
    assert float((scaled - tok).abs().max()) < 1e-4
