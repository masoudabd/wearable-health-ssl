"""Percent time frozen in elapsed time for the FogAtHome external test (scripts/fogathome_time.py).

Synthetic sessions check the per-participant analyzed, annotated and predicted seconds against a brute-force mask on
a fine grid (clipping to the session's duration included). When the FogAtHome data are present, the annotated
intervals read from the raw files must reproduce every cached window's FOG fraction to the resampling grain.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import fogathome_time as FT  # noqa: E402
from paper_ptf_time import intersect_length  # noqa: E402
from whssl.data import fogathome as FH  # noqa: E402

RES = 1e-3


def synthetic(rng):
    """Three participants, five sessions with gaps, one session ending inside its last window."""
    t0, rec, subject, dur = [], [], [], {}
    for k, (p, n) in enumerate([("a", 12), ("a", 7), ("b", 20), ("c", 9), ("c", 15)]):
        t = np.arange(n) * 2.5
        t = np.delete(t, rng.choice(n, size=n // 5, replace=False))  # invalid stretches leave gaps
        t0.append(t); rec += [f"s{k}"] * len(t); subject += [p] * len(t)
        dur[f"s{k}"] = t[-1] + (3.0 if k == 1 else 5.0)                # s1 ends 2 s before its last window does
    iv = pd.DataFrame({"rec": ["s0", "s0", "s2", "s4"], "start_s": [3.1, 20.0, 0.0, 11.3], "end_s": [8.7, 26.4, 4.0, 30.0]})
    W = {"t0": np.concatenate(t0), "rec": np.array(rec), "subject": np.array(subject)}
    return W, iv, pd.Series(dur)


def brute(W, iv, dur, pred=None):
    out = {}
    for p in np.unique(W["subject"]):
        a = f = q = 0.0
        for r in np.unique(W["rec"][W["subject"] == p]):
            m = W["rec"] == r
            g = np.arange(0, dur[r], RES)
            cover = np.zeros(len(g), bool)
            for t in W["t0"][m]:
                cover |= (g >= t) & (g < min(t + 5, dur[r]))
            fog = np.zeros(len(g), bool)
            for s, e in iv[iv.rec == r][["start_s", "end_s"]].to_numpy():
                fog |= (g >= s) & (g < e)
            pc = np.zeros(len(g), bool)
            if pred is not None:
                for t in W["t0"][m & pred]:
                    pc |= (g >= t) & (g < min(t + 5, dur[r]))
            a, f, q = a + cover.sum() * RES, f + (cover & fog).sum() * RES, q + pc.sum() * RES
        out[p] = (a, f, q)
    return out


@pytest.mark.parametrize("seed", range(3))
def test_elapsed_time_matches_brute_force(seed, monkeypatch):
    rng = np.random.default_rng(seed)
    W, iv, dur = synthetic(rng)
    monkeypatch.setattr(FH, "fog_intervals", lambda split, log=None: (iv, dur))
    E = FT.ElapsedTime(W, "synthetic")
    pred = rng.random(len(W["t0"])) < 0.4
    ref = brute(W, iv, dur, pred)
    q = E.predicted_s(pred)
    for i, p in enumerate(E.pts):
        a, f, qq = ref[p]
        assert E.analyzed_s[i] == pytest.approx(a, abs=5 * RES)
        assert E.annotated_s[i] == pytest.approx(f, abs=5 * RES)
        assert q[i] == pytest.approx(qq, abs=5 * RES)
    assert np.allclose(E.ptf_pred(np.ones(len(pred), bool)), 100)   # every window predicted covers all analyzed time
    assert np.all(E.predicted_s(np.zeros(len(pred), bool)) == 0)


@pytest.mark.skipif(not (FH.RAW / "fogathome").exists(), reason="FogAtHome data not downloaded")
def test_intervals_reproduce_cached_window_fractions():
    W = FH.load_windows("fogathome", (-1.0, 1.0, 1.0))
    iv, _ = FH.fog_intervals("fogathome", log=None)
    g = {r: (d.start_s.to_numpy(), d.end_s.to_numpy()) for r, d in iv.groupby("rec")}
    frac = np.array([intersect_length(np.array([t]), np.array([t + 5.0]), *g[r]) / 5 if r in g else 0.0
                     for t, r in zip(W["t0"], W["rec"])])
    assert np.max(np.abs(frac - W["frac"])) < 2 / 150   # two samples of a 5 s window at 30 Hz
