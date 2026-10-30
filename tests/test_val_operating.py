"""The elapsed-time burden threshold of scripts/paper_val_operating.py.

The threshold is found by bisection, which is only right if the predicted share of time falls monotonically as the
threshold rises and the closest share lies next to the crossing. Both are checked here against a brute-force search
over every validation score, on the cohorts' regular 5 s / 2.5 s window grid with gaps between recordings.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from paper_ptf_time import union_length  # noqa: E402
from paper_val_operating import burden_time_threshold, fog_seconds  # noqa: E402


def grid(n_per_rec=(40, 25, 60)):
    """Window lookup for consecutive recordings, indexed by window number, and each recording's analyzed time."""
    t0, rec, dur = [], [], []
    for r, n in enumerate(n_per_rec):
        t = np.arange(n) * 2.5
        t0.append(t); rec.append(np.full(n, r)); dur.append(np.full(n, t[-1] + 5.0))
    lookup = (np.concatenate(t0), np.concatenate(rec).astype(np.int64), np.concatenate(dur))
    analyzed = sum(union_length(t, t + 5.0) for t in t0)
    return lookup, analyzed


def brute(win, sc, lookup, analyzed):
    cand = np.unique(sc)
    return cand, np.array([fog_seconds(win, sc >= c, lookup) / analyzed for c in cand])


def test_share_is_monotone_and_counts_overlap_once():
    lookup, analyzed = grid()
    win = np.arange(len(lookup[0]))
    sc = np.random.default_rng(0).random(len(win))
    _, shares = brute(win, sc, lookup, analyzed)
    assert shares[0] == pytest.approx(1.0)          # every window predicted covers all analyzed time
    assert np.all(np.diff(shares) <= 1e-12)          # non-increasing as the threshold rises


@pytest.mark.parametrize("seed", range(20))
def test_bisection_finds_the_closest_share(seed):
    rng = np.random.default_rng(seed)
    lookup, analyzed = grid(tuple(rng.integers(5, 80, size=3)))
    win = np.arange(len(lookup[0]))
    # scores with runs, as detectors produce: smoothed noise, plus some ties
    sc = np.round(np.convolve(rng.random(len(win)), np.ones(3) / 3, mode="same"), 3)
    target = rng.uniform(0.02, 0.6)
    thr, got = burden_time_threshold(win, sc, target, analyzed, lookup)
    cand, shares = brute(win, sc, lookup, analyzed)
    assert got == pytest.approx(fog_seconds(win, sc >= thr, lookup) / analyzed)
    assert abs(got - target) == pytest.approx(np.min(np.abs(shares - target)))


def test_target_below_every_share_takes_the_highest_score():
    lookup, analyzed = grid((10,))
    win = np.arange(10)
    sc = np.linspace(0, 1, 10)
    thr, got = burden_time_threshold(win, sc, 1e-6, analyzed, lookup)
    assert thr == 1.0 and got == pytest.approx(5.0 / analyzed)
