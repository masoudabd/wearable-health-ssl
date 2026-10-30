"""The inclusion-exclusion identity behind episode coverage (scripts/paper_episode_strict.py).

Coverage of an episode is computed as the sum of each predicted window's overlap with the episode minus, for each
adjacent predicted pair, the part of the episode inside their shared span. That is exact only because a window is
exactly two hops long, so neighbours overlap and no three windows share any instant. Both facts are checked here,
and the identity itself is checked against interval merging on random window sets.

The analysis script also verifies the identity on real episodes at every run (`--verify`), which covers the real
link structure; these tests cover the geometry it relies on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from paper_ptf_time import intersect_length  # noqa: E402

WIN, HOP = 5.0, 2.5


def coverage_fast(t0: np.ndarray, e_start: float, e_end: float) -> float:
    """The script's identity, written out for one episode and a set of predicted windows (all on the cache grid)."""
    t0 = np.sort(t0)
    overlap = np.maximum(np.minimum(e_end, t0 + WIN) - np.maximum(e_start, t0), 0.0).sum()
    adjacent = np.flatnonzero(np.isclose(np.diff(t0), HOP))
    shared = np.maximum(np.minimum(e_end, t0[adjacent] + WIN) - np.maximum(e_start, t0[adjacent] + HOP), 0.0).sum()
    return overlap - shared


def test_window_is_exactly_two_hops():
    """The identity assumes this; if the cache ever changes it, the identity stops being exact."""
    assert WIN == pytest.approx(2 * HOP)


def test_no_three_windows_share_an_instant():
    # windows two apart start exactly where the first ends, so at most two windows cover any instant
    t0 = np.arange(0, 100, HOP)
    grid = np.arange(0, 100, 0.01) + 0.005
    covering = ((grid[:, None] >= t0[None, :]) & (grid[:, None] < t0[None, :] + WIN)).sum(axis=1)
    assert covering.max() == 2


def test_identity_on_a_contiguous_run():
    t0 = np.arange(5) * HOP                     # covers [0, 15)
    assert coverage_fast(t0, 0.0, 15.0) == pytest.approx(15.0)
    assert coverage_fast(t0, 2.0, 6.0) == pytest.approx(4.0)   # episode inside the run


def test_identity_with_a_gap():
    t0 = np.array([0.0, 2.5, 10.0])             # [0, 7.5) and [10, 15)
    assert coverage_fast(t0, 0.0, 15.0) == pytest.approx(7.5 + 5.0)


def test_identity_for_an_episode_outside_the_windows():
    assert coverage_fast(np.array([0.0, 2.5]), 20.0, 25.0) == pytest.approx(0.0)


@pytest.mark.parametrize("seed", range(40))
def test_identity_matches_interval_merging(seed):
    rng = np.random.default_rng(seed)
    t0 = np.unique(rng.choice(np.arange(0, 60, HOP), size=rng.integers(1, 14), replace=False))
    e_start = float(rng.uniform(0, 55))
    e_end = e_start + float(rng.uniform(0.2, 12))
    exact = intersect_length(t0, t0 + WIN, np.array([e_start]), np.array([e_end]))
    assert coverage_fast(t0, e_start, e_end) == pytest.approx(exact, abs=1e-9)


def test_coverage_never_exceeds_episode_duration():
    rng = np.random.default_rng(99)
    for _ in range(50):
        t0 = np.unique(rng.choice(np.arange(0, 60, HOP), size=rng.integers(1, 20), replace=False))
        e_start = float(rng.uniform(0, 50)); e_end = e_start + float(rng.uniform(0.2, 10))
        assert coverage_fast(t0, e_start, e_end) <= (e_end - e_start) + 1e-9


def test_stricter_rule_can_only_lower_detection():
    """Detection under a higher minimum coverage is a subset of detection under a lower one."""
    rng = np.random.default_rng(3)
    cov = rng.uniform(0, 1, 2000)
    for lo, hi in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.8)):
        assert (cov >= hi).sum() <= (cov > lo).sum()
