"""Post hoc data description (writing brief 8.5): axis sign conventions of the lower-back recordings.

The datasets name their axes V, ML and AP without documenting the signs. Per cohort: mean acceleration per axis over
all analysis windows, for FOG and non-FOG windows, and the number of patients whose mean AP is positive or negative;
for the FogAtHome recordings of the FORGE release, the same before any harmonization (the cache stores V negated,
which is undone here), daily living restricted to walking and standing windows. Read-only on the window caches.

Outputs (results/paper_derived/axis_conventions/): summary.json, report.md.
Usage: .venv/bin/python scripts/paper_axis_conventions.py   (seconds)
"""
from __future__ import annotations

import json
import time

import numpy as np

import paper_common as C
from whssl.data import fogathome as FH

NAME = "axis_conventions"


def describe(X, y, subject):
    m = X.mean(axis=1)  # per-window mean per axis
    pts = np.unique(subject)
    ap_pt = np.array([m[subject == p, 2].mean() for p in pts])
    out = {"windows": int(len(X)), "patients": int(len(pts)),
           "mean_v": float(m[:, 0].mean()), "mean_ml": float(m[:, 1].mean()), "mean_ap": float(m[:, 2].mean()),
           "patients_ap_positive": int((ap_pt > 0).sum()), "patients_ap_negative": int((ap_pt < 0).sum())}
    if y is not None and (y == 1).any():
        out.update({"mean_ap_fog": float(m[y == 1, 2].mean()), "mean_ap_nonfog": float(m[y == 0, 2].mean())})
    return out


def main():
    t0 = time.time()
    W = C.F.load_windows(30.0, 5.0, 2.5)
    res = {}
    for cohort in C.COHORTS:
        k = W["cohort"] == cohort
        res[cohort] = describe(W["X"][k], W["y"][k], W["subject"][k])
    undo = np.array([-1.0, 1.0, 1.0], np.float32)  # the FogAtHome cache stores V negated
    S = FH.load_windows("fogathome", (-1.0, 1.0, 1.0))
    res["fogathome_structured"] = describe(S["X"] * undo, S["y"], S["subject"])
    res["fogathome_structured"]["sessions_v_positive"] = int(sum(
        (S["X"][S["rec"] == r][:, :, 0] * -1).mean() > 0 for r in np.unique(S["rec"])))
    res["fogathome_structured"]["sessions"] = int(len(np.unique(S["rec"])))
    D = FH.load_windows("fogathome_dailyliving", (-1.0, 1.0, 1.0))
    ws = np.isin(D["activity"], FH.WALK_STAND_CODES)
    res["fogathome_daily_walkstand"] = describe(D["X"][ws] * undo, D["y"][ws], D["subject"][ws])
    summary = {"commit": C.git_commit(), "date": time.strftime("%Y-%m-%d"), "units": "g", **res,
               "seconds": round(time.time() - t0, 1)}
    rows = "\n".join(f"| {k} | {v['mean_v']:+.2f} | {v['mean_ml']:+.2f} | {v['mean_ap']:+.2f} | "
                     f"{v['patients_ap_positive']} of {v['patients']} |" for k, v in res.items())
    report = f"""# Axis sign conventions (post hoc data description)

Written by `scripts/paper_axis_conventions.py` on {summary['date']} at commit {summary['commit']}. Mean acceleration
(g) over windows, as recorded (before any harmonization).

| Recordings | V | ML | AP | Patients with mean AP > 0 |
|---|---|---|---|---|
{rows}
"""
    out = C.write_outputs(NAME, summary, {}, report)
    print(report)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
