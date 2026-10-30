"""Post hoc analysis a (writing brief 8.5): patient-bootstrap intervals over the repetitions of RQ1/RQ2.

For each cohort, one set of 2,000 patient resamples (the same as scripts/paired_tests.py: sorted patients,
whssl.eval.paired.patient_counts with seed 0) is shared by every repetition, approach and budget. In each resample
a statistic is computed within each repetition on its out-of-fold windows and averaged over repetitions; the 2.5 and
97.5 percentiles give the 95% interval. Outputs:

  metrics.csv              pooled AUROC and pooled AUPRC (average precision over all out-of-fold windows of a
                           cohort), mean over repetitions, with intervals; between-repetition SD of the point values
  paired_auprc.csv         paired AUPRC differences for the six RQ1/RQ2 comparisons (two-sided bootstrap P, Holm over
                           the AUPRC family; secondary to the AUROC tests in paired_tests.csv)
  paired_auroc_check.csv   the AUROC differences recomputed here, to check them against paired_tests.csv
  budget_differences.csv   within an approach, all training patients minus 20, 10 and 5 labeled patients (same test
                           windows and resamples, so paired), AUROC and AUPRC
  cohort_gap.csv           lab minus home cohort for every approach and budget: independent patient resamples (the
                           home resamples are permuted so that they are not paired with the lab ones)
  summary.json, report.md

Usage: .venv/bin/python scripts/paper_intervals.py [--runs RUN ...]   (about five minutes on the CPU)
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import paper_common as C

NAME = "intervals"
METRICS = {"auroc": C.weighted_auroc, "auprc": C.weighted_auprc}
CONTRASTS = [("all", "20"), ("all", "10"), ("all", "5")]


def main():
    ap = C.common_args(argparse.ArgumentParser())
    ap.add_argument("--n-boot", type=int, default=C.N_BOOT)
    a = ap.parse_args()
    t_start = time.time()
    by_seed, labels, cfgs = C.load_runs(a.runs, a.allow_superseded,
                                        columns=["cohort", "fold", "budget", "arm", "subject", "y", "score"])
    seeds = sorted(by_seed)
    point, boot, info = {}, {}, {}
    for cohort in C.COHORTS:
        pts = None
        for s in seeds:
            dc = by_seed[s][by_seed[s].cohort == cohort]
            p_s, _ = C.patient_index(dc.subject.to_numpy())
            if pts is None:
                pts, counts = p_s, C.resamples(len(p_s), a.n_boot, seed=0)
            elif not np.array_equal(pts, p_s):
                raise SystemExit(f"seed {s}: patients of {cohort} differ")
            for (budget, arm), d in dc.groupby(["budget", "arm"]):
                y, sc = d.y.to_numpy(), d.score.to_numpy()
                col = np.searchsorted(pts, d.subject.to_numpy())
                for m, fn in METRICS.items():
                    point[m, s, cohort, budget, arm] = float(fn(sc, y, np.ones(len(y))))
                    boot[m, s, cohort, budget, arm] = C.boot_metric(fn, sc, y, col, counts)
                info[cohort, budget, arm] = (len(pts), float(y.mean()))
        print(f"{cohort}: bootstrap done ({time.time() - t_start:.0f} s)", flush=True)

    def mean_point(m, c, b, arm):
        return float(np.mean([point[m, s, c, b, arm] for s in seeds]))

    def mean_boot(m, c, b, arm):
        return np.mean([boot[m, s, c, b, arm] for s in seeds], axis=0)

    keys = sorted({k[2:] for k in point})
    rows = []
    for c, b, arm in keys:
        r = {"cohort": c, "budget": b, "arm": arm, "n_patients": info[c, b, arm][0], "prevalence": info[c, b, arm][1],
             "n_seeds": len(seeds)}
        for m in METRICS:
            lo, hi = C.interval(mean_boot(m, c, b, arm))
            r.update({m: mean_point(m, c, b, arm), f"{m}_lo": lo, f"{m}_hi": hi,
                      f"{m}_seed_sd": float(np.std([point[m, s, c, b, arm] for s in seeds], ddof=1))
                      if len(seeds) > 1 else np.nan})
        rows.append(r)
    metrics = pd.DataFrame(rows)

    def paired(m, comparisons):
        out = []
        for c in C.COHORTS:
            for A, B in comparisons:
                for b in C.BUDGETS:
                    if (c, b, A) not in info or (c, b, B) not in info:
                        continue
                    d_seed = [point[m, s, c, b, A] - point[m, s, c, b, B] for s in seeds]
                    d_boot = mean_boot(m, c, b, A) - mean_boot(m, c, b, B)
                    lo, hi = C.interval(d_boot)
                    out.append({"cohort": c, "budget": b, "comparison": f"{A} - {B}", "metric": m,
                                "d_mean": float(np.mean(d_seed)), "ci_lo": lo, "ci_hi": hi, "p": C.two_sided_p(d_boot),
                                "seeds_positive": int(np.sum(np.array(d_seed) > 0)), "n_seeds": len(seeds),
                                "n_patients": info[c, b, A][0]})
        df = pd.DataFrame(out)
        df["p_holm"] = C.holm(df.p.to_numpy())
        return df

    paired_auprc = paired("auprc", C.COMPARISONS)
    check = paired("auroc", C.COMPARISONS)
    ref_dir = "rq12_seeds_mps_fixed" if labels == "fixed" else "rq12_seeds_mps"  # where each run set's tests are
    ref_path = C.ROOT / "results" / ref_dir / "paired_tests.csv"
    check_note = "no paired_tests.csv found to compare with"
    if ref_path.exists():
        ref = pd.read_csv(ref_path, dtype={"budget": str})
        mg = check.merge(ref, on=["cohort", "budget", "comparison"], suffixes=("", "_ref"))
        dev = float(np.max(np.abs(mg[["d_mean", "ci_lo", "ci_hi"]].to_numpy()
                                  - mg[["d_mean_ref", "ci_lo_ref", "ci_hi_ref"]].to_numpy())))
        check_note = f"{len(mg)} AUROC tests compared with {ref_path.relative_to(C.ROOT)}: largest deviation {dev:.2e}"

    bud = []
    for c in C.COHORTS:
        for arm in C.ARMS:
            for hi_b, lo_b in CONTRASTS:
                if (c, hi_b, arm) not in info or (c, lo_b, arm) not in info:
                    continue
                for m in METRICS:
                    d_seed = [point[m, s, c, hi_b, arm] - point[m, s, c, lo_b, arm] for s in seeds]
                    d_boot = mean_boot(m, c, hi_b, arm) - mean_boot(m, c, lo_b, arm)
                    lo, hi = C.interval(d_boot)
                    bud.append({"cohort": c, "arm": arm, "contrast": f"{hi_b} - {lo_b}", "metric": m,
                                "d_mean": float(np.mean(d_seed)), "ci_lo": lo, "ci_hi": hi,
                                "p": C.two_sided_p(d_boot), "seeds_positive": int(np.sum(np.array(d_seed) > 0)),
                                "n_seeds": len(seeds)})
    budget_diff = pd.DataFrame(bud)

    perm = np.random.default_rng(1).permutation(a.n_boot)  # unpair the home resamples from the lab ones
    gap = []
    for b in C.BUDGETS:
        for arm in C.ARMS:
            if ("tdcsfog", b, arm) not in info or ("defog", b, arm) not in info:
                continue
            for m in METRICS:
                lab, home = mean_point(m, "tdcsfog", b, arm), mean_point(m, "defog", b, arm)
                d_boot = mean_boot(m, "tdcsfog", b, arm) - mean_boot(m, "defog", b, arm)[perm]
                lo, hi = C.interval(d_boot)
                gap.append({"budget": b, "arm": arm, "metric": m, "lab": lab, "home": home, "d_lab_minus_home": lab - home,
                            "ci_lo": lo, "ci_hi": hi, "n_seeds": len(seeds)})
    cohort_gap = pd.DataFrame(gap)

    summary = {"commit": C.git_commit(), "date": time.strftime("%Y-%m-%d"), "labels": labels,
               "runs": {str(s): cfgs[s]["run"] for s in seeds}, "run_commits": {str(s): cfgs[s]["commit"] for s in seeds},
               "n_boot": a.n_boot, "seeds": seeds, "auroc_check": check_note,
               "seconds": round(time.time() - t_start, 1)}
    show = metrics[metrics.budget.isin(["10", "all"])].sort_values(["cohort", "budget", "auroc"], ascending=[True, True, False])
    lines = "\n".join(f"| {r.cohort} | {r.budget} | {r.arm} | {r.auroc:.3f} ({r.auroc_lo:.3f} to {r.auroc_hi:.3f}) | "
                      f"{r.auprc:.3f} ({r.auprc_lo:.3f} to {r.auprc_hi:.3f}) | {r.prevalence:.3f} |" for r in show.itertuples())
    gl = "\n".join(f"| {r.budget} | {r.arm} | {r.d_lab_minus_home:+.3f} ({r.ci_lo:+.3f} to {r.ci_hi:+.3f}) |"
                   for r in cohort_gap[cohort_gap.metric == "auroc"].itertuples())
    report = f"""# Patient-bootstrap intervals over the repetitions (analysis a, post hoc)

Written by `scripts/paper_intervals.py` on {summary['date']} at commit {summary['commit']}; labels: {labels};
{len(seeds)} repetitions; {a.n_boot:,} patient resamples per cohort shared by repetitions, approaches and budgets.
Check: {check_note}.

## Pooled AUROC and AUPRC, mean over repetitions (95% patient-bootstrap interval), budgets 10 and all

| Cohort | Budget | Approach | AUROC | AUPRC | Prevalence |
|---|---|---|---|---|---|
{lines}

## Lab minus home cohort, AUROC (independent resamples)

| Budget | Approach | Difference (95% interval) |
|---|---|---|
{gl}

Other tables: `paired_auprc.csv`, `budget_differences.csv`, `paired_auroc_check.csv`, `cohort_gap.csv` (AUPRC rows).
"""
    out = C.write_outputs(NAME, summary, {"metrics.csv": metrics, "paired_auprc.csv": paired_auprc,
                                          "paired_auroc_check.csv": check, "budget_differences.csv": budget_diff,
                                          "cohort_gap.csv": cohort_gap}, report, base=a.out_base)
    print(report)
    print(f"wrote {out} in {summary['seconds']} s")


if __name__ == "__main__":
    main()
