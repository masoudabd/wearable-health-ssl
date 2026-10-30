"""Post hoc analysis b (writing brief 8.5): agreement of predicted with annotated percent time frozen per patient.

Percent time frozen of a patient is the share of the patient's analyzed windows that are FOG: annotated (window
labels) and predicted (windows above the threshold chosen on validation patients, as stored in the out-of-fold
file). Each repetition tests every patient once, so per repetition there is one pair of values per patient.

Statistics per repetition, then averaged over repetitions: Spearman correlation; intraclass correlation ICC(A,1)
(two-way model, absolute agreement, single measures; McGraw and Wong 1996), the form the FOG contest paper reports
for this endpoint; mean absolute error and mean signed error (predicted minus annotated), in percentage points.
Intervals: 2,000 patient resamples per cohort shared by repetitions and approaches (as in scripts/paired_tests.py),
so the differences between approaches are paired. Outputs:

  agreement.csv     cohort, budget, approach, statistic: mean over repetitions, 95% interval, between-repetition SD
  paired.csv        differences between approaches (the six RQ1/RQ2 comparisons) for each statistic, two-sided
                    bootstrap P, Holm within each statistic
  per_patient.csv   annotated and predicted percent time frozen per patient, mean and SD over repetitions
  summary.json, report.md

Usage: .venv/bin/python scripts/paper_ptf_agreement.py [--runs RUN ...]   (about a minute on the CPU)
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import paper_common as C

NAME = "ptf_agreement"
STATS = ["rho", "icc", "mae", "bias"]


def stats(x: np.ndarray, z: np.ndarray) -> dict:
    """x annotated, z predicted, both (B, n): one value of each statistic per row."""
    n = x.shape[1]
    rx, rz = rankdata(x, axis=1), rankdata(z, axis=1)
    rx, rz = rx - rx.mean(1, keepdims=True), rz - rz.mean(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = (rx * rz).sum(1) / np.sqrt((rx ** 2).sum(1) * (rz ** 2).sum(1))
        g = (x.mean(1) + z.mean(1)) / 2
        m = (x + z) / 2
        msr = 2 * ((m - g[:, None]) ** 2).sum(1) / (n - 1)
        msc = n * ((x.mean(1) - g) ** 2 + (z.mean(1) - g) ** 2)
        sst = ((x - g[:, None]) ** 2).sum(1) + ((z - g[:, None]) ** 2).sum(1)
        mse = (sst - (n - 1) * msr - msc) / (n - 1)
        icc = (msr - mse) / (msr + mse + 2 / n * (msc - mse))
    return {"rho": rho, "icc": icc, "mae": np.abs(z - x).mean(1), "bias": (z - x).mean(1)}


def main():
    ap = C.common_args(argparse.ArgumentParser())
    ap.add_argument("--n-boot", type=int, default=C.N_BOOT)
    a = ap.parse_args()
    t_start = time.time()
    by_seed, labels, cfgs = C.load_runs(a.runs, a.allow_superseded,
                                        columns=["cohort", "fold", "budget", "arm", "subject", "y", "pred"])
    seeds = sorted(by_seed)
    point, boot, pp_rows, npat = {}, {}, [], {}
    for cohort in C.COHORTS:
        pts, idx = None, None
        for s in seeds:
            dc = by_seed[s][by_seed[s].cohort == cohort]
            per = dc.groupby(["budget", "arm", "subject"])[["y", "pred"]].mean() * 100
            p_s = np.array(sorted(dc.subject.unique()))
            if pts is None:
                pts = p_s
                counts = C.resamples(len(pts), a.n_boot, seed=0)
                idx = np.stack([np.repeat(np.arange(len(pts)), c.astype(int)) for c in counts])  # (B, n) patients
                npat[cohort] = len(pts)
            elif not np.array_equal(pts, p_s):
                raise SystemExit(f"seed {s}: patients of {cohort} differ")
            for (budget, arm), d in per.groupby(level=[0, 1]):
                d = d.droplevel([0, 1]).reindex(pts)
                x, z = d.y.to_numpy(), d.pred.to_numpy()
                st = stats(x[None, :], z[None, :])
                bt = stats(x[idx], z[idx])
                for k in STATS:
                    point[k, s, cohort, budget, arm] = float(st[k][0])
                    boot[k, s, cohort, budget, arm] = bt[k]
                pp_rows.append(pd.DataFrame({"cohort": cohort, "budget": budget, "arm": arm, "seed": s,
                                             "subject": pts, "ptf_true": x, "ptf_pred": z}))
        print(f"{cohort} done ({time.time() - t_start:.0f} s)", flush=True)

    def mp(k, c, b, arm):
        return float(np.mean([point[k, s, c, b, arm] for s in seeds]))

    def mb(k, c, b, arm):
        return np.mean([boot[k, s, c, b, arm] for s in seeds], axis=0)

    cells = sorted({key[2:] for key in point})
    rows = []
    for c, b, arm in cells:
        for k in STATS:
            lo, hi = C.interval(mb(k, c, b, arm))
            vals = [point[k, s, c, b, arm] for s in seeds]
            rows.append({"cohort": c, "budget": b, "arm": arm, "statistic": k, "value": mp(k, c, b, arm), "lo": lo,
                         "hi": hi, "seed_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
                         "n_patients": npat[c], "n_seeds": len(seeds)})
    agreement = pd.DataFrame(rows)

    prs = []
    for k in STATS:
        part = []
        for c in C.COHORTS:
            for A, B in C.COMPARISONS:
                for b in C.BUDGETS:
                    if (c, b, A) not in {x for x in cells} or (c, b, B) not in {x for x in cells}:
                        continue
                    d_seed = [point[k, s, c, b, A] - point[k, s, c, b, B] for s in seeds]
                    d_boot = mb(k, c, b, A) - mb(k, c, b, B)
                    lo, hi = C.interval(d_boot)
                    part.append({"cohort": c, "budget": b, "comparison": f"{A} - {B}", "statistic": k,
                                 "d_mean": float(np.mean(d_seed)), "ci_lo": lo, "ci_hi": hi, "p": C.two_sided_p(d_boot),
                                 "seeds_positive": int(np.sum(np.array(d_seed) > 0)), "n_seeds": len(seeds),
                                 "n_patients": npat[c]})
        part = pd.DataFrame(part)
        part["p_holm"] = C.holm(part.p.to_numpy())
        prs.append(part)
    paired = pd.concat(prs, ignore_index=True)

    pp = pd.concat(pp_rows, ignore_index=True)
    per_patient = (pp.groupby(["cohort", "budget", "arm", "subject"])
                   .agg(ptf_true=("ptf_true", "mean"), ptf_pred=("ptf_pred", "mean"), ptf_pred_sd=("ptf_pred", "std"),
                        n_seeds=("seed", "nunique")).reset_index())

    summary = {"commit": C.git_commit(), "date": time.strftime("%Y-%m-%d"), "labels": labels,
               "runs": {str(s): cfgs[s]["run"] for s in seeds}, "run_commits": {str(s): cfgs[s]["commit"] for s in seeds},
               "n_boot": a.n_boot, "seeds": seeds, "seconds": round(time.time() - t_start, 1)}
    show = agreement[(agreement.budget == "all")].pivot_table(index=["cohort", "arm"], columns="statistic",
                                                              values="value").round(3)
    report = f"""# Agreement of predicted with annotated percent time frozen (analysis b, post hoc)

Written by `scripts/paper_ptf_agreement.py` on {summary['date']} at commit {summary['commit']}; labels: {labels};
{len(seeds)} repetitions; {a.n_boot:,} patient resamples per cohort. ICC is ICC(A,1); MAE and bias in percentage
points (bias = predicted minus annotated).

## All training patients, mean over repetitions

{C.md_table(show)}

Intervals, other budgets and paired differences: `agreement.csv`, `paired.csv`.
"""
    out = C.write_outputs(NAME, summary, {"agreement.csv": agreement, "paired.csv": paired,
                                          "per_patient.csv": per_patient}, report, base=a.out_base)
    print(report)
    print(f"wrote {out} in {summary['seconds']} s")


if __name__ == "__main__":
    main()
