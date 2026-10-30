"""The weighted, vectorized AUROC must equal sklearn on explicitly resampled data, ties included."""
import numpy as np
from sklearn.metrics import roc_auc_score

from whssl.eval.paired import boot_auroc, holm, patient_counts, two_sided_p, weighted_auroc


def _data(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(int)
    scores = np.round(rng.normal(size=n) + y, 1)  # rounding creates many ties
    return scores, y


def test_unit_weights_equal_sklearn():
    s, y = _data()
    assert abs(weighted_auroc(s, y, np.ones(len(y))) - roc_auc_score(y, s)) < 1e-12


def test_integer_weights_equal_sklearn_on_repeated_windows():
    s, y = _data()
    w = np.random.default_rng(1).integers(0, 4, size=len(y))
    assert abs(weighted_auroc(s, y, w) - roc_auc_score(np.repeat(y, w), np.repeat(s, w))) < 1e-12


def test_patient_bootstrap_matches_explicit_resampling():
    s, y = _data(600)
    patient = np.repeat(np.arange(30), 20)  # 30 patients x 20 windows
    counts = patient_counts(30, 50, seed=3)
    fast = boot_auroc(s, y, patient, counts, chunk=7)
    for b in range(50):
        idx = np.concatenate([np.flatnonzero(patient == p) for p in np.repeat(np.arange(30), counts[b].astype(int))])
        assert abs(fast[b] - roc_auc_score(y[idx], s[idx])) < 1e-12


def test_identical_arms_give_zero_difference_and_helpers():
    s, y = _data(600)
    patient = np.repeat(np.arange(30), 20)
    counts = patient_counts(30, 200)
    d = boot_auroc(s, y, patient, counts) - boot_auroc(s.copy(), y, patient, counts)
    assert np.all(d == 0)
    assert two_sided_p(np.r_[np.full(990, 0.1), np.full(10, -0.1)]) == 0.02
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
