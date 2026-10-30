"""Post hoc summary (writing brief 8.5) of the external test on FogAtHome (scripts/external_fogathome.py).

Reads the four runs (training cohort home or lab; axes harmonized by the main rule, Vflip or VAPflip, or by the
sensitivity rule that also negates ML) and
  * Holm-adjusts the 12 main paired comparisons (foundation model minus clinical features, default and tuned probes,
    for 2 training cohorts x 3 test sets, main axis rule) as one family; the sensitivity rule is reported unadjusted;
  * describes the test sets from the window caches: participants, windows, analyzed hours (the union of the valid
    windows of each session), the share of FOG windows and the share of analyzed time annotated FOG (elapsed time,
    scripts/fogathome_time.py, the reference for the percent-time-frozen errors).
Recomputes no score.

Outputs (results/paper_derived/external/): summary.csv, summary.json, report.md.
Usage: .venv/bin/python scripts/paper_external.py   (seconds)
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import paper_common as C
from fogathome_time import ElapsedTime
from whssl.data import fogathome as FH

NAME = "external"
RUNS = {("defog", "Vflip"): True, ("tdcsfog", "VAPflip"): True, ("defog", "VMLflip"): False,
        ("tdcsfog", "VAPMLflip"): False}


def hours(W: dict, hop: float = 2.5, win: float = 5.0) -> float:
    rec, t0 = W["rec"], W["t0"]
    cont = np.r_[False, (rec[1:] == rec[:-1]) & (np.abs(np.diff(t0) - hop) < 1e-6)]
    return float((np.full(len(t0), hop) + (~cont) * (win - hop)).sum() / 3600)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-suffix", default="", help="appended to each external run directory, e.g. _b20k")
    args = ap.parse_args()
    sfx = args.runs_suffix
    t0 = time.time()
    frames = []
    for (src, axes), main_rule in RUNS.items():
        d = pd.read_csv(C.ROOT / f"results/external_fogathome_{src}_{axes}{sfx}" / "summary.csv")
        frames.append(d.assign(main_rule=main_rule))
    d = pd.concat(frames, ignore_index=True)
    d["comparison"] = d.arm.str.contains(" - ")
    fam = d.comparison & d.main_rule
    d["p_holm_main12"] = np.nan
    d.loc[fam, "p_holm_main12"] = C.holm(d.loc[fam, "p"].to_numpy())
    sets = {}
    for split, name in [("fogathome", "structured"), ("fogathome_dailyliving", "daily")]:
        W = FH.load_windows(split, (-1.0, 1.0, 1.0))
        E = ElapsedTime(W, split)
        sets[name] = {"participants": int(len(np.unique(W["subject"]))), "windows": int(len(W["y"])),
                      "sessions": int(len(np.unique(W["rec"]))), "hours": hours(W),
                      "fog_windows_pct": float(100 * W["y"].mean()),
                      "fog_time_pct": float(100 * E.annotated_s.sum() / E.analyzed_s.sum())}
        if name == "daily":
            ws = np.isin(W["activity"], FH.WALK_STAND_CODES)
            sets["daily_walkstand"] = {"participants": int(len(np.unique(W["subject"][ws]))), "windows": int(ws.sum()),
                                       "windows_pct_of_daily": float(100 * ws.mean()),
                                       "fog_windows_pct": float(100 * W["y"][ws].mean())}
    summary = {"commit": C.git_commit(), "date": time.strftime("%Y-%m-%d"), "n_tests_main": int(fam.sum()),
               "test_sets": sets, "runs": {f"{s}_{x}": f"results/external_fogathome_{s}_{x}{sfx}" for s, x in RUNS},
               "seconds": round(time.time() - t0, 1)}
    show = d[fam][["source", "axes", "test", "arm", "auroc", "auroc_lo", "auroc_hi", "p", "p_holm_main12"]]
    report = f"""# External test on FogAtHome, summary (post hoc)

Written by `scripts/paper_external.py` on {summary['date']} at commit {summary['commit']}. Test sets: {sets}.

Main paired comparisons (foundation model minus clinical features), Holm over the {summary['n_tests_main']}:

{C.md_table(show.set_index(["source", "axes", "test", "arm"]))}
"""
    C.write_outputs(NAME, summary, {"summary.csv": d}, report)
    print(report)


if __name__ == "__main__":
    main()
