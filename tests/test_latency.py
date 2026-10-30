"""Detection latency of scripts/paper_latency.py (analysis k).

The first flag of an episode is taken with one scatter-min over the episode-window links. Checked here against a
brute-force search over every window, on the cohorts' regular 5 s / 2.5 s window grid with several recordings, and
the references against what the grid allows.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from paper_latency import LatencyCohort, per_patient  # noqa: E402


def synthetic(seed: int = 0, n_per_rec=(60, 45, 80, 30)):
    """Window metadata, episodes and episode-window links for one cohort of two patients."""
    rng = np.random.default_rng(seed)
    t0, rec, subj = [], [], []
    for r, n in enumerate(n_per_rec):
        t0.append(np.arange(n) * 2.5); rec.append(np.full(n, f"r{r}")); subj.append(np.full(n, f"p{r % 2}"))
    t0, rec, subj = np.concatenate(t0), np.concatenate(rec), np.concatenate(subj)
    eps = []
    for r, n in enumerate(n_per_rec):
        end_rec = (n - 1) * 2.5 + 5.0
        start = 0.0
        while True:
            start += rng.uniform(1.0, 25.0)
            dur = float(np.exp(rng.uniform(np.log(0.3), np.log(40.0))))
            if start + dur > end_rec:
                break
            eps.append({"episode": f"r{r}_{len(eps):03d}", "rec": f"r{r}", "subject": f"p{r % 2}", "cohort": "c",
                        "start_s": start, "end_s": start + dur, "duration_s": dur, "valid_share": 1})
            start += dur
    ep = pd.DataFrame(eps)
    links, fog = [], np.zeros(len(t0))
    for e in ep.itertuples():
        on = np.flatnonzero((rec == e.rec) & (t0 < e.end_s) & (t0 + 5.0 > e.start_s))
        ov = np.minimum(t0[on] + 5.0, e.end_s) - np.maximum(t0[on], e.start_s)
        fog[on] += ov / 5.0
        links += [{"episode": e.episode, "win": int(w), "overlap_s": float(o)} for w, o in zip(on, ov)]
    W = {"cohort": np.full(len(t0), "c"), "rec": rec, "t0": t0, "subject": subj, "frac": fog,
         "y": (fog >= 0.5).astype(np.int8)}
    return W, ep, pd.DataFrame(links)


def brute_flag(S: LatencyCohort, pos: np.ndarray) -> np.ndarray:
    out = np.full(len(S.episodes), np.inf)
    for i, e in enumerate(S.episodes.itertuples()):
        on = (S.rec == e.rec) & (S.t0 < e.end_s) & (S.t0 + 5.0 > e.start_s) & pos
        if on.any():
            out[i] = (S.t0[on] + 5.0).min()
    return out


def test_first_flag_matches_brute_force():
    for seed in range(4):
        W, ep, epw = synthetic(seed)
        S = LatencyCohort("c", W, ep, epw)
        rng = np.random.default_rng(seed + 10)
        for share in (0.05, 0.3, 0.8):
            pos = rng.random(len(S.win)) < share
            got, want = S.flag(pos), brute_flag(S, pos)
            assert np.array_equal(np.isfinite(got), np.isfinite(want))
            assert np.allclose(got[np.isfinite(got)], want[np.isfinite(want)], rtol=0, atol=1e-12)
            # the episodes with a flag are exactly analysis e's detected episodes
            assert np.array_equal(np.isfinite(got), S.evaluate(pos)["det"])
            lat = got - S.ep_start
            assert (lat[np.isfinite(lat)] > 0).all()


def test_windows_reference_is_the_grid_floor():
    W, ep, epw = synthetic(1)
    S = LatencyCohort("c", W, ep, epw)
    lat = S.flag(np.ones(len(S.win), bool)) - S.ep_start
    inside = S.ep_start >= 2.5  # away from the recording start, some window starts 2.5 to 5 s before onset
    assert np.isfinite(lat).all()
    assert (lat[inside] > 0).all() and (lat[inside] <= 2.5 + 1e-9).all()
    # no detector flags earlier than the windowing allows
    rng = np.random.default_rng(3)
    other = S.flag(rng.random(len(S.win)) < 0.5) - S.ep_start
    ok = np.isfinite(other)
    assert (other[ok] >= lat[ok] - 1e-12).all()


def test_labels_reference_and_ratios():
    W, ep, epw = synthetic(2)
    S = LatencyCohort("c", W, ep, epw)
    f = S.flag(S.y)
    # an episode shorter than half a window cannot make a window FOG by itself; isolated ones stay unflagged
    short = S.episodes.duration_s.to_numpy() < 2.5
    alone = np.array([((S.episodes.rec == e.rec) & (S.episodes.end_s > e.start_s - 5) &
                       (S.episodes.start_s < e.end_s + 5)).sum() == 1 for e in S.episodes.itertuples()])
    assert not np.isfinite(f[short & alone]).any()
    p = per_patient(S, f)
    det, n = p["detected"]
    assert n.sum() == len(S.episodes) and det.sum() == np.isfinite(f).sum()
    in_time, det2 = p["in_time_of_detected"]
    assert in_time.sum() <= det2.sum()
    long_in, n_long = p["in_time_long"]
    assert n_long.sum() == (S.episodes.duration_s >= 5.0).sum() and long_in.sum() <= n_long.sum()
