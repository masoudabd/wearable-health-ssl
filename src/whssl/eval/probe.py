"""Linear probe on frozen features. Scaler and classifier are fit on training subjects only."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_probe(Z: np.ndarray, y: np.ndarray, C: float = 1.0, seed: int = 0):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=seed),
    )
    return clf.fit(Z, y)


def probe_scores(clf, Z: np.ndarray) -> np.ndarray:
    return clf.predict_proba(Z)[:, 1]
