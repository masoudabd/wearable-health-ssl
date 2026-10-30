"""Post hoc analysis d (writing brief 8.5): calibration of the window probabilities of RQ1/RQ2.

The probes are class-balanced logistic regressions and the supervised CNN ends in a sigmoid, so every approach
outputs a probability; none was recalibrated. Per repetition, on the pooled out-of-fold windows of a cohort:
Brier score; calibration in the large (mean predicted probability minus prevalence); calibration intercept and slope
(logistic regression of the label on the logit of the probability, clipped to [1e-6, 1 - 1e-6]); a reliability table
over ten equal-width probability bins. Averaged over repetitions; the Brier score and calibration in the large get
patient-bootstrap intervals (2,000 resamples shared by repetitions and approaches). Outputs:

  calibration.csv   cohort, budget, approach: Brier (interval), calibration in the large (interval), intercept and
                    slope (mean and SD over repetitions)
  reliability.csv   cohort, budget, approach, bin: mean predicted probability, observed FOG share, windows (sums over
                    repetitions)
  summary.json, report.md

Usage: .venv/bin/python scripts/paper_calibration.py [--runs RUN ...]   (about a minute on the CPU)
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import paper_common as C

NAME = "calibration"
BUDGETS = ["10", "all"]


def main():
    ap = C.common_args(argparse.ArgumentParser())
    ap.add_argument("--n-boot", type=int, default=C.N_BOOT)
    a = ap.parse_args()
    t_start = time.time()
    by_seed, labels, cfgs = C.load_runs(a.runs, a.allow_superseded,
                                        columns=["cohort", "fold", "budget", "arm", "subject", "y", "score"])
    seeds = sorted(by_seed)
    res, rel = {}, []
    for cohort in C.COHORTS:
        pts = None
        for s in seeds:
            dc = by_seed[s][(by_seed[s].cohort == cohort) & by_seed[s].budget.isin(BUDGETS)]
            if pts is None:
                pts = np.array(sorted(dc.subject.unique()))
                counts = C.resamples(len(pts), a.n_boot, seed=0)
            for (budget, arm), d in dc.groupby(["budget", "arm"]):
                y, p = d.y.to_numpy().astype(float), np.clip(d.score.to_numpy(), 1e-6, 1 - 1e-6)
                col = np.searchsorted(pts, d.subject.to_numpy())
                w = counts[:, col]
                wsum = w.sum(1)
                brier_b = (w * (p - y) ** 2).sum(1) / wsum
                citl_b = (w * (p - y)).sum(1) / wsum
                lr = LogisticRegression(C=1e6, max_iter=1000).fit(np.log(p / (1 - p))[:, None], y)
                res[s, cohort, budget, arm] = {"brier": float(np.mean((p - y) ** 2)), "citl": float(np.mean(p - y)),
                                               "slope": float(lr.coef_[0, 0]), "intercept": float(lr.intercept_[0]),
                                               "brier_b": brier_b, "citl_b": citl_b, "prevalence": float(y.mean())}
                bins = np.minimum((p * 10).astype(int), 9)
                for b in range(10):
                    m = bins == b
                    rel.append({"cohort": cohort, "budget": budget, "arm": arm, "seed": s, "bin": b,
                                "p_sum": p[m].sum(), "y_sum": y[m].sum(), "n": int(m.sum())})
        print(f"{cohort} done ({time.time() - t_start:.0f} s)", flush=True)

    rows = []
    for c, b, arm in sorted({k[1:] for k in res}):
        r = [res[s, c, b, arm] for s in seeds]
        row = {"cohort": c, "budget": b, "arm": arm, "prevalence": r[0]["prevalence"], "n_seeds": len(seeds)}
        for k in ("brier", "citl"):
            lo, hi = C.interval(np.mean([x[f"{k}_b"] for x in r], axis=0))
            row.update({k: float(np.mean([x[k] for x in r])), f"{k}_lo": lo, f"{k}_hi": hi})
        for k in ("slope", "intercept"):
            row.update({k: float(np.mean([x[k] for x in r])), f"{k}_sd": float(np.std([x[k] for x in r], ddof=1))
                        if len(r) > 1 else np.nan})
        rows.append(row)
    cal = pd.DataFrame(rows)
    rl = pd.DataFrame(rel).groupby(["cohort", "budget", "arm", "bin"]).sum(numeric_only=True).reset_index()
    rl = rl.assign(mean_predicted=rl.p_sum / rl.n.where(rl.n > 0), observed=rl.y_sum / rl.n.where(rl.n > 0))
    rl = rl[["cohort", "budget", "arm", "bin", "mean_predicted", "observed", "n"]]

    summary = {"commit": C.git_commit(), "date": time.strftime("%Y-%m-%d"), "labels": labels,
               "runs": {str(s): cfgs[s]["run"] for s in seeds}, "run_commits": {str(s): cfgs[s]["commit"] for s in seeds},
               "n_boot": a.n_boot, "seeds": seeds, "seconds": round(time.time() - t_start, 1)}
    show = cal[cal.budget == "all"][["cohort", "arm", "prevalence", "brier", "citl", "intercept", "slope"]]
    report = f"""# Calibration of window probabilities (analysis d, post hoc)

Written by `scripts/paper_calibration.py` on {summary['date']} at commit {summary['commit']}; labels: {labels};
{len(seeds)} repetitions. Calibration in the large is the mean predicted probability minus the prevalence.

## All training patients, mean over repetitions

{C.md_table(show.set_index(["cohort", "arm"]))}

Intervals and the budget of 10 patients: `calibration.csv`; reliability bins: `reliability.csv`.
"""
    out = C.write_outputs(NAME, summary, {"calibration.csv": cal, "reliability.csv": rl}, report, base=a.out_base)
    print(report)
    print(f"wrote {out} in {summary['seconds']} s")


if __name__ == "__main__":
    main()
