"""Validate the walking filter used to pick gait-like daily-living windows for RQ4 (whssl.signal.gait_like), on HAR70+.

HAR70+ (UCI 780, CC BY 4.0): 18 older adults, Axivity sensor on the lower back, 50 Hz, g, video-annotated activities
(1 walking, 3 shuffling, 4 stairs up, 5 stairs down, 6 standing, 7 sitting, 8 lying). The back sensor is resampled to
30 Hz and cut into non-overlapping 5 s windows, as the FOG data; windows whose label covers at least 80% are kept.

Rhythmicity threshold: the 10th percentile of walking windows' rhythmicity, so 90% of walking passes by construction.
HAR70+ cannot tune it by specificity: it has no non-walking movement classes (only static postures besides gait),
and against static postures the movement threshold alone already separates (the grid then picks the lowest
rhythmicity, which in daily living would admit non-rhythmic movement: 56% of active windows instead of 32%).
Movement threshold: the grid value with the highest walking sensitivity at specificity >= 0.95 against standing,
sitting and lying, given that rhythmicity threshold. Honest estimate: 2-fold split by subject (both thresholds set on
one half, tested on the other, both ways); final thresholds: set on all 18. Writes results/gait_filter/ (grid.csv,
validation.json). Result 2026-10-05: SD >= 0.02 g and rhythmicity >= 0.33; held out, sensitivity 0.898 and
specificity 0.977 (whssl.data.fog_daily uses these thresholds).

Usage: python scripts/validate_gait_filter.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from whssl.signal import gait_features, resample  # noqa: E402

RAW = ROOT / "data" / "raw" / "har70plus" / "har70" / "har70plus"
OUT = ROOT / "results" / "gait_filter"
FS_IN, FS, T = 50.0, 30.0, 150
WALK, STATIC = {1}, {6, 7, 8}
SD_GRID = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
CONC_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
MIN_SPEC = 0.95


def windows():
    rows = []
    for f in sorted(RAW.glob("*.csv")):
        d = pd.read_csv(f)
        acc = resample(d[["back_x", "back_y", "back_z"]].to_numpy(np.float32), FS_IN, FS, axis=0)
        lab = d.label.to_numpy()[np.minimum((np.arange(len(acc)) * FS_IN / FS).round().astype(int), len(d) - 1)]
        n = len(acc) // T
        W, L = acc[: n * T].reshape(n, T, 3), lab[: n * T].reshape(n, T)
        maj = np.array([np.bincount(l).argmax() for l in L])
        pure = (L == maj[:, None]).mean(axis=1) >= 0.8
        sd, fdom, conc = gait_features(W[pure], FS)
        rows.append(pd.DataFrame({"subject": f.stem, "label": maj[pure], "sd": sd, "f": fdom, "conc": conc}))
    return pd.concat(rows, ignore_index=True)


def score(d, sd_t, conc_t):
    pred = (d.sd >= sd_t) & (d.conc >= conc_t)
    walk, static = d.label.isin(WALK), d.label.isin(STATIC)
    return {"sensitivity": float(pred[walk].mean()), "specificity": float(1 - pred[static].mean()),
            "shuffling_selected": float(pred[d.label == 3].mean()), "stairs_selected": float(pred[d.label.isin({4, 5})].mean())}


def tune(d):
    c = float(np.percentile(d[d.label.isin(WALK)].conc, 10))  # keeps 90% of walking by construction
    best = None
    for s in SD_GRID:
        r = score(d, s, c)
        if r["specificity"] >= MIN_SPEC and (best is None or r["sensitivity"] > best[2]["sensitivity"]):
            best = (s, c, r)
    return best


def main():
    d = windows()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"min_sd": s, "min_conc": c, **score(d, s, c)} for s in SD_GRID for c in CONC_GRID]).to_csv(
        OUT / "grid.csv", index=False)
    subjects = np.array(sorted(d.subject.unique()))
    halves = [subjects[::2], subjects[1::2]]
    held_out = []
    for a, b in [(0, 1), (1, 0)]:
        s, c, _ = tune(d[d.subject.isin(halves[a])])
        held_out.append({"tuned_on": list(halves[a]), "min_sd": s, "min_conc": c, **score(d[d.subject.isin(halves[b])], s, c)})
    s, c, r = tune(d)
    counts = d.label.value_counts().sort_index().to_dict()
    result = {"windows": int(len(d)), "windows_by_label": {int(k): int(v) for k, v in counts.items()},
              "final": {"min_sd": s, "min_conc": c, **r}, "held_out_2fold": held_out,
              "held_out_mean": {k: float(np.mean([h[k] for h in held_out])) for k in ["sensitivity", "specificity"]},
              "walking_f_median_hz": float(d[d.label == 1].f.median())}
    (OUT / "validation.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
