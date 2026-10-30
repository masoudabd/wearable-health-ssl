"""Paired patient-level bootstrap for differences in pooled AUROC between arms scored on the same windows.

A replicate draws patients with replacement (as `metrics.subject_bootstrap_ci` does). Every window carries its
patient's draw count as a weight, so all arms, budgets and seeds are scored on exactly the same resample (paired),
and AUROC is computed from the weights over one sort, vectorized across replicates.
"""
from __future__ import annotations

import numpy as np


def patient_counts(n_patients: int, n_boot: int, seed: int = 0) -> np.ndarray:
    """(n_boot, n_patients) draw counts; row b is bootstrap resample b of the patients."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_patients, np.full(n_patients, 1.0 / n_patients), size=n_boot).astype(np.float64)


def weighted_auroc(scores: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """AUROC with per-window weights. weights (n,) gives a float; (B, n) gives one AUROC per row. Ties count one
    half, as in sklearn. Equals roc_auc_score on the data with each window repeated `weight` times. A row without
    both classes gives NaN."""
    w = np.atleast_2d(np.asarray(weights, dtype=np.float64))
    order = np.argsort(scores, kind="mergesort")
    s, yy, w = np.asarray(scores)[order], np.asarray(y)[order].astype(bool), w[:, order]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])  # first position of each group of tied scores
    pos = np.add.reduceat(w * yy, starts, axis=1)
    neg = np.add.reduceat(w * ~yy, starts, axis=1)
    below = np.cumsum(neg, axis=1) - neg  # negative weight strictly below each tie group
    wp, wn = pos.sum(axis=1), neg.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = (pos * (below + 0.5 * neg)).sum(axis=1) / (wp * wn)
    auc[(wp == 0) | (wn == 0)] = np.nan
    return auc if np.ndim(weights) == 2 else float(auc[0])


def boot_auroc(scores: np.ndarray, y: np.ndarray, window_patient: np.ndarray, counts: np.ndarray,
               chunk: int = 200) -> np.ndarray:
    """AUROC in every bootstrap replicate. window_patient: (n,) column of `counts` for each window's patient.

    Every window of a patient carries the same weight, so the AUROC numerator is bilinear in the draw counts c:
    pos and neg per tie group are c times the patient's window counts, and

        numerator(c) = sum_g pos_g (below_g + neg_g / 2) = c' A c,  A = P (below + N / 2)'

    with P and N the positive and negative window counts per patient and tie group. A is (patients, patients) and
    depends only on the scores, so it is built once and each replicate costs a quadratic form instead of a pass over
    every window. Identical scores still give exactly equal results, as the paired tests need. `chunk` is accepted
    for compatibility and no longer used.
    """
    scores, yy = np.asarray(scores), np.asarray(y).astype(bool)
    col, counts = np.asarray(window_patient), np.asarray(counts, dtype=np.float64)
    n_pat = counts.shape[1]
    order = np.argsort(scores, kind="mergesort")
    s, yo, co = scores[order], yy[order], col[order]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    grp = np.zeros(len(s), dtype=np.int64)
    grp[starts] = 1
    grp = np.cumsum(grp) - 1  # tie-group index of each window, in score order
    n_grp = len(starts)
    flat = co.astype(np.int64) * n_grp + grp
    P = np.bincount(flat[yo], minlength=n_pat * n_grp).astype(np.float64).reshape(n_pat, n_grp)
    N = np.bincount(flat[~yo], minlength=n_pat * n_grp).astype(np.float64).reshape(n_pat, n_grp)
    A = P @ (np.cumsum(N, axis=1) - 0.5 * N).T  # negative windows strictly below, plus half the tie group
    num = np.einsum("bj,bj->b", counts @ A, counts, optimize=True)
    wp, wn = counts @ P.sum(axis=1), counts @ N.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = num / (wp * wn)
    auc[(wp == 0) | (wn == 0)] = np.nan
    return auc


def two_sided_p(d_boot: np.ndarray) -> float:
    """Bootstrap p-value for 'difference = 0': twice the smaller tail mass at zero, floored at 1 / replicates."""
    d = d_boot[np.isfinite(d_boot)]
    return float(min(1.0, max(2 * min(np.mean(d <= 0), np.mean(d >= 0)), 1.0 / len(d))))


def holm(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values (family-wise error)."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj
