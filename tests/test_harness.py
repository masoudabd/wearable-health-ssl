"""Checks that the harness cannot leak subjects and that the pretext transforms behave."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whssl.eval.metrics import binary_report, subject_bootstrap_ci
from whssl.eval.splits import carve_validation, subject_kfold, subsample_subjects
from whssl.models.encoders import ResNet1D, embed
from whssl.ssl.multitask import make_pretext_batch


def _toy(n_sub=30, per=40, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat([f"S{i}" for i in range(n_sub)], per)
    y = rng.integers(0, 2, size=n_sub * per)
    return y, groups


def test_kfold_has_no_subject_overlap_and_covers_everything():
    y, g = _toy()
    seen = []
    for tr, te in subject_kfold(y, g, 5, 0):
        assert not set(g[tr]) & set(g[te])
        seen.append(te)
    assert sorted(np.concatenate(seen)) == list(range(len(y)))


def test_validation_is_disjoint_from_fit_and_test():
    y, g = _toy()
    tr, te = next(subject_kfold(y, g, 5, 0))
    fit, val = carve_validation(y, g, tr)
    assert not set(g[fit]) & set(g[val])
    assert not set(g[val]) & set(g[te])
    assert set(fit) | set(val) == set(tr)


def test_subsample_subjects_count():
    y, g = _toy()
    idx = np.arange(len(y))
    sub = subsample_subjects(y, g, idx, 7, seed=1)
    assert len(np.unique(g[sub])) == 7


def test_metrics_on_perfect_scores():
    y = np.array([0, 0, 1, 1])
    rep = binary_report(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert rep["auroc"] == 1.0 and rep["f1_pos"] == 1.0
    y, g = _toy()
    a, lo, hi = subject_bootstrap_ci(y, y.astype(float), g, n_boot=50)
    assert a == lo == hi == 1.0


def test_pretext_transforms_keep_shape_and_change_signal():
    torch.manual_seed(0)
    xin = torch.randn(64, 3, 200)
    xb, labels = make_pretext_batch(xin)
    assert xb.shape == xin.shape and set(labels) == {"aot", "perm", "warp"}
    transformed = (labels["aot"] + labels["perm"] + labels["warp"]) > 0
    for i in range(len(xin)):  # a window changes exactly when at least one task was applied to it
        assert torch.equal(xb[i], xin[i]) != bool(transformed[i])


def test_encoder_embedding_shape():
    Z = embed(ResNet1D(), np.random.randn(10, 200, 3).astype(np.float32))
    assert Z.shape == (10, 128)
