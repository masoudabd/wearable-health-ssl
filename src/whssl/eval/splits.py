"""Subject-wise splits. Every split in this project is made on subjects, never on windows."""
from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def subject_kfold(
    y: np.ndarray, groups: np.ndarray, n_splits: int = 5, seed: int = 0
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Stratified k-fold over subjects. Yields (train_idx, test_idx) window indices."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in cv.split(np.zeros(len(y)), y, groups):
        assert not set(groups[tr]) & set(groups[te]), "subject leakage between train and test"
        yield tr, te


def carve_validation(
    y: np.ndarray, groups: np.ndarray, train_idx: np.ndarray, frac: float = 0.2, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Split train_idx into (fit_idx, val_idx) by subject. Use val for every selection decision
    (checkpoint, threshold, hyperparameters) so the test subjects stay untouched."""
    n_splits = max(2, min(int(round(1.0 / frac)), len(np.unique(groups[train_idx]))))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    first = None
    for fit, val in cv.split(np.zeros(len(train_idx)), y[train_idx], groups[train_idx]):
        first = first or (fit, val)
        # prefer a split where both fit and validation contain both classes (threshold needs positives)
        if len(np.unique(y[train_idx][val])) > 1 and len(np.unique(y[train_idx][fit])) > 1:
            first = (fit, val)
            break
    fit, val = first
    fit_idx, val_idx = train_idx[fit], train_idx[val]
    assert not set(groups[fit_idx]) & set(groups[val_idx])
    return fit_idx, val_idx


def subsample_subjects(
    y: np.ndarray, groups: np.ndarray, idx: np.ndarray, n_subjects: int, seed: int = 0
) -> np.ndarray:
    """Keep windows of n_subjects randomly chosen subjects (for label-efficiency curves).
    Redraws until both classes are present."""
    rng = np.random.default_rng(seed)
    subjects = np.unique(groups[idx])
    if n_subjects >= len(subjects):
        return idx
    for _ in range(200):
        keep = rng.choice(subjects, size=n_subjects, replace=False)
        sub = idx[np.isin(groups[idx], keep)]
        if len(np.unique(y[sub])) > 1:
            return sub
    raise RuntimeError("could not draw a subject subset containing both classes")
