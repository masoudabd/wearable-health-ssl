"""Oxford harnet wrapper: loads offline, embeds, and resampling keeps duration."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whssl.models import pretrained as P

needs_weights = pytest.mark.skipif(not P.OXFORD_REPO.exists(), reason="Oxford weights not downloaded")


def test_resample_keeps_duration():
    x = np.random.randn(1280, 3)  # 10 s at 128 Hz
    y = P.resample(x, 128, 30)
    assert y.shape == (300, 3)
    x = np.random.randn(1000, 3)  # 10 s at 100 Hz
    assert P.resample(x, 100, 30).shape == (300, 3)


@needs_weights
@pytest.mark.parametrize("name", ["harnet5", "harnet10"])
def test_harnet_embeds(name):
    secs, dim = P.HARNET[name]
    fe = P.load_harnet(name)
    Z = P.harnet_embed(fe, np.random.randn(5, secs * P.HARNET_FS, 3).astype(np.float32))
    assert Z.shape == (5, dim) and np.isfinite(Z).all()
