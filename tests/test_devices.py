"""Training and embedding run on every available device (CPU always; MPS on Apple GPUs; CUDA when present)."""
import numpy as np
import pytest
import torch

from whssl.models.encoders import ResNet1D, embed
from whssl.models.supervised import predict, train_supervised
from whssl.ssl.multitask import pretrain

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_pretrain_embed_supervised_on_device(device):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((600, 150, 3)).astype(np.float32)  # (N, T, C), as the scripts pass it
    y = (rng.random(600) < 0.3).astype(int)
    enc = ResNet1D()
    hist = pretrain(enc, X, epochs=1, batch_size=128, device=device, log=None)
    assert np.isfinite(hist["loss"][0])
    Z = embed(enc, X, device=device)
    assert Z.shape == (600, enc.emb_dim) and np.isfinite(Z).all()
    model = train_supervised(ResNet1D(), X[:480], y[:480], X[480:], y[480:], epochs=1, batch_size=128,
                             device=device, log=None)
    p = predict(model, X[480:], device=device)
    assert p.shape == (120,) and ((p >= 0) & (p <= 1)).all()
