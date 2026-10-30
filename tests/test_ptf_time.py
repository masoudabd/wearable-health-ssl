"""The interval arithmetic behind the time-based percent-time-frozen endpoint (scripts/paper_ptf_time.py).

Every number of analysis h rests on these two functions, so they are checked against a brute-force mask on a fine
grid for random interval sets, and exactly on the regular 5 s / 2.5 s window grid the cohorts use.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from paper_ptf_time import intersect_length, union_length, window_intervals  # noqa: E402

RES = 1e-4


def brute_union(s, e):
    g = np.arange(min(s), max(e), RES)
    m = np.zeros(len(g), bool)
    for a, b in zip(s, e):
        m |= (g >= a) & (g < b)
    return m.sum() * RES


def brute_intersect(a_s, a_e, b_s, b_e):
    lo, hi = min([*a_s, *b_s]), max([*a_e, *b_e])
    g = np.arange(lo, hi, RES)
    ma, mb = np.zeros(len(g), bool), np.zeros(len(g), bool)
    for a, b in zip(a_s, a_e):
        ma |= (g >= a) & (g < b)
    for a, b in zip(b_s, b_e):
        mb |= (g >= a) & (g < b)
    return (ma & mb).sum() * RES


def test_union_empty_and_single():
    assert union_length(np.array([]), np.array([])) == 0.0
    assert union_length(np.array([3.0]), np.array([8.0])) == pytest.approx(5.0)


def test_union_counts_overlap_once():
    # two windows overlapping by half, as consecutive windows of the cache do
    assert union_length(np.array([0.0, 2.5]), np.array([5.0, 7.5])) == pytest.approx(7.5)
    # a contiguous run of k windows covers 2.5 k + 2.5 seconds
    for k in (1, 2, 5, 10, 37):
        t0 = np.arange(k) * 2.5
        assert union_length(t0, t0 + 5.0) == pytest.approx(2.5 * k + 2.5)


def test_union_separate_runs_add():
    # a run of two windows covers 2.5 * 2 + 2.5 = 7.5 s, a run of three covers 2.5 * 3 + 2.5 = 10 s
    t0 = np.array([0.0, 2.5, 100.0, 102.5, 105.0])
    assert union_length(t0, t0 + 5.0) == pytest.approx(7.5 + 10.0)


def test_union_touching_intervals_merge():
    # [0, 5) and [5, 10) touch without overlapping and must count as 10 s, not 10 s twice
    assert union_length(np.array([0.0, 5.0]), np.array([5.0, 10.0])) == pytest.approx(10.0)


def test_union_nested_interval():
    assert union_length(np.array([0.0, 1.0]), np.array([10.0, 2.0])) == pytest.approx(10.0)


def test_intersect_basics():
    assert intersect_length(np.array([]), np.array([]), np.array([0.0]), np.array([5.0])) == 0.0
    assert intersect_length(np.array([0.0]), np.array([10.0]),
                            np.array([4.0]), np.array([6.0])) == pytest.approx(2.0)
    # disjoint
    assert intersect_length(np.array([0.0]), np.array([2.0]),
                            np.array([5.0]), np.array([7.0])) == pytest.approx(0.0)


def test_intersect_handles_overlap_within_each_set():
    # the window set overlaps itself; the intersection must not double-count the shared time
    w0 = np.array([0.0, 2.5, 5.0])
    ep_s, ep_e = np.array([1.0, 3.0]), np.array([4.0, 9.0])   # these two episodes also overlap
    got = intersect_length(w0, w0 + 5.0, ep_s, ep_e)
    assert got == pytest.approx(brute_intersect(w0, w0 + 5.0, ep_s, ep_e), abs=1e-3)
    assert got == pytest.approx(8.0)  # windows cover [0, 10); episodes cover [1, 9)


@pytest.mark.parametrize("seed", range(25))
def test_union_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    s = np.sort(rng.uniform(0, 40, rng.integers(1, 10)))
    e = s + rng.uniform(0.2, 7, len(s))
    assert union_length(s, e) == pytest.approx(brute_union(s, e), abs=1e-3)


@pytest.mark.parametrize("seed", range(25))
def test_intersect_matches_brute_force(seed):
    rng = np.random.default_rng(1000 + seed)
    s1 = np.sort(rng.uniform(0, 40, rng.integers(1, 10)))
    e1 = s1 + rng.uniform(0.2, 7, len(s1))
    s2 = np.sort(rng.uniform(0, 40, rng.integers(1, 10)))
    e2 = s2 + rng.uniform(0.2, 7, len(s2))
    assert intersect_length(s1, e1, s2, e2) == pytest.approx(brute_intersect(s1, e1, s2, e2), abs=1e-3)


def test_intersect_is_symmetric():
    rng = np.random.default_rng(7)
    for _ in range(20):
        s1 = np.sort(rng.uniform(0, 30, 4)); e1 = s1 + rng.uniform(0.5, 5, 4)
        s2 = np.sort(rng.uniform(0, 30, 4)); e2 = s2 + rng.uniform(0.5, 5, 4)
        assert intersect_length(s1, e1, s2, e2) == pytest.approx(intersect_length(s2, e2, s1, e1))


def test_intersection_never_exceeds_either_union():
    rng = np.random.default_rng(11)
    for _ in range(50):
        s1 = np.sort(rng.uniform(0, 30, 5)); e1 = s1 + rng.uniform(0.5, 6, 5)
        s2 = np.sort(rng.uniform(0, 30, 5)); e2 = s2 + rng.uniform(0.5, 6, 5)
        i = intersect_length(s1, e1, s2, e2)
        assert i <= union_length(s1, e1) + 1e-9
        assert i <= union_length(s2, e2) + 1e-9


def test_window_intervals_clip_to_recording():
    t0 = np.array([0.0, 2.5, 5.0])
    s, e = window_intervals(t0, duration=8.0)
    assert np.allclose(s, t0)
    assert np.allclose(e, [5.0, 7.5, 8.0])          # the last window is cut at the recording's end
    assert union_length(s, e) == pytest.approx(8.0)
