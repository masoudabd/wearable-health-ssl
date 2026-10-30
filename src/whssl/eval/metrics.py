"""Metrics for imbalanced binary clinical tasks, with subject-level bootstrap intervals."""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Threshold maximizing positive-class F1. Call on VALIDATION data only."""
    grid = np.unique(np.quantile(y_score, np.linspace(0.01, 0.99, 99)))
    f1 = [f1_score(y_true, y_score >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1))])


def binary_report(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
        "f1_pos": float(f1_score(y_true, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "prevalence": float(np.mean(y_true)),
    }


def subject_bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float] = roc_auc_score,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Point estimate and percentile CI, resampling SUBJECTS with replacement
    (windows of one subject are not independent)."""
    rng = np.random.default_rng(seed)
    subjects = np.unique(groups)
    by_subject = {s: np.flatnonzero(groups == s) for s in subjects}
    stats = []
    for _ in range(n_boot):
        draw = rng.choice(subjects, size=len(subjects), replace=True)
        idx = np.concatenate([by_subject[s] for s in draw])
        if len(np.unique(y_true[idx])) < 2:
            continue
        stats.append(metric(y_true[idx], y_score[idx]))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(metric(y_true, y_score)), float(lo), float(hi)
