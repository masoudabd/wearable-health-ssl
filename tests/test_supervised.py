"""train_supervised: early stopping and the default path."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whssl.models.encoders import ResNet1D  # noqa: E402
from whssl.models.supervised import predict, train_supervised  # noqa: E402


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 150, 3)).astype(np.float32)
    y = (rng.random(n) < 0.3).astype(int)
    X[y == 1, :, 0] += 1.0
    return X, y


def test_patience_stops_early_and_keeps_best_epoch():
    X, y = _data()
    torch.manual_seed(0)
    m = train_supervised(ResNet1D(), X[:200], y[:200], X[200:], y[200:], epochs=40, batch_size=64,
                         seed=0, log=None, patience=2)
    assert 1 <= m.best_epoch <= m.epochs_run <= 40
    assert m.epochs_run < 40 or m.best_epoch > 38  # stopped early unless still improving at the end
    assert m.epochs_run - m.best_epoch <= 2


def test_no_patience_trains_all_epochs_and_is_reproducible():
    X, y = _data()
    out = []
    for _ in range(2):
        torch.manual_seed(0)
        m = train_supervised(ResNet1D(), X[:200], y[:200], X[200:], y[200:], epochs=3, batch_size=64,
                             seed=0, log=None)
        assert m.epochs_run == 3
        out.append(predict(m, X[200:]))
    np.testing.assert_array_equal(out[0], out[1])


def test_linear_head_from_probe_reproduces_probe_probabilities():
    from whssl.eval.probe import fit_probe, probe_scores
    from whssl.models.pretrained import linear_head_from_probe

    rng = np.random.default_rng(1)
    Z = rng.normal(2.0, 3.0, size=(400, 16)).astype(np.float32)
    y = (Z[:, 0] + rng.normal(size=400) > 2.5).astype(int)
    clf = fit_probe(Z, y, C=0.1)
    head = linear_head_from_probe(clf, 16)
    with torch.no_grad():
        p = torch.sigmoid(head(torch.from_numpy(Z))).squeeze(-1).numpy()
    np.testing.assert_allclose(p, probe_scores(clf, Z), atol=1e-5)
