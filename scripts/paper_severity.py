"""Post hoc analysis f (plan v0 section 10, approved 2026-10-04): severity across the five repetitions, and by sex.

Extends scripts/rq3_severity_fog.py (one repetition) to all repetitions with paired patient-bootstrap intervals.
Same definitions as RQ3: patient severity is the mean over the patient's visits; tertiles of MDS-UPDRS Part III ON
and of NFOG-Q are cut within each cohort (lowest t1 to highest t3); per-patient AUROC needs at least 10 FOG and 10
non-FOG windows. All analyses are exploratory.

In each of 2,000 patient resamples per cohort (shared by repetitions and approaches; the patients of a stratum keep
their stratum) a statistic is computed within each repetition and averaged over repetitions. Contrasts:
  t1 minus t3          pooled AUROC of the lowest minus the highest tertile (different patients)
  ON minus OFF         pooled AUROC ON minus OFF medication among patients recorded in both states (within patient)
  female minus male    pooled AUROC by sex
Correlations: per-patient AUROC (mean over repetitions) against severity, Spearman with a patient-bootstrap interval;
also for the per-patient difference foundation model minus clinical features, and for the absolute error of
percent time frozen. Outputs: strata.csv, contrasts.csv, correlations.csv, per_patient.csv, summary.json, report.md.

Usage: .venv/bin/python scripts/paper_severity.py [--runs RUN ...]   (a few minutes on the CPU)
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

import paper_common as C

NAME = "severity"
BUDGETS = ["all", "10"]
MIN_PER_CLASS = 10
SEV = ["updrs3_on", "nfogq"]
CORR_SEV = ["updrs3_on", "nfogq", "years_dx"]


def strata_definitions(cohort_pts: np.ndarray, pt: pd.DataFrame, med_both: set) -> dict:
    """{(severity, level): set of patients} plus the score range of each tertile."""
    p = pt.loc[cohort_pts]
    out, ranges = {}, {}
    for sev in SEV:
        q = p[sev].quantile([1 / 3, 2 / 3]).to_numpy()
        tert = pd.cut(p[sev], [-np.inf, q[0], q[1], np.inf], labels=["low", "mid", "high"])
        for level in ["low", "mid", "high"]:
            members = set(p.index[tert == level])
            out[sev, level] = members
            v = p.loc[list(members), sev]
            ranges[sev, level] = f"{v.min():.0f}-{v.max():.0f}" if len(v) else ""
    for sex, level in (("F", "female"), ("M", "male")):
        out["sex", level] = set(p.index[p.sex == sex])
        ranges["sex", level] = ""
    out["medication_within", "both"] = med_both
    return out, ranges


def spearman_boot(x: np.ndarray, y: np.ndarray, idx: np.ndarray) -> np.ndarray:
    rx, ry = rankdata(x[idx], axis=1), rankdata(y[idx], axis=1)
    rx, ry = rx - rx.mean(1, keepdims=True), ry - ry.mean(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (rx * ry).sum(1) / np.sqrt((rx ** 2).sum(1) * (ry ** 2).sum(1))


def main():
    ap = C.common_args(argparse.ArgumentParser())
    ap.add_argument("--n-boot", type=int, default=C.N_BOOT)
    a = ap.parse_args()
    t_start = time.time()
    by_seed, labels, cfgs = C.load_runs(a.runs, a.allow_superseded,
                                        columns=["cohort", "fold", "budget", "arm", "subject", "y", "score", "pred"])
    W = C.window_meta(labels)
    pt = C.patient_table()
    seeds = sorted(by_seed)
    strata_rows, contrast_rows, corr_rows, pp_rows = [], [], [], []
    rng = np.random.default_rng(0)
    for cohort in C.COHORTS:
        m = W["cohort"] == cohort
        meds = pd.Series(W["medication"][m]).groupby(W["subject"][m]).agg(set)
        pts = np.array(sorted(set(W["subject"][m])))
        counts = C.resamples(len(pts), a.n_boot, seed=0)
        defs, ranges = strata_definitions(pts, pt, set(meds.index[meds.apply(lambda x: {"on", "off"} <= x)]))
        point, boot, ppa = {}, {}, {}
        for s in seeds:
            dc = by_seed[s][(by_seed[s].cohort == cohort) & by_seed[s].budget.isin(BUDGETS)]
            for (budget, arm), d in dc.groupby(["budget", "arm"]):
                y, sc, subj = d.y.to_numpy(), d.score.to_numpy(), d.subject.to_numpy()
                med = W["medication"][d.win.to_numpy()]
                col = np.searchsorted(pts, subj)
                groups = {key: np.isin(subj, list(members)) for key, members in defs.items()
                          if key != ("medication_within", "both")}
                within = np.isin(subj, list(defs["medication_within", "both"]))
                groups["medication_within", "on"] = within & (med == "on")
                groups["medication_within", "off"] = within & (med == "off")
                groups["medication_all", "on"], groups["medication_all", "off"] = med == "on", med == "off"
                for key, g in groups.items():
                    if g.sum() == 0 or len(np.unique(y[g])) < 2:
                        continue
                    point[s, budget, arm, key] = C.auroc(sc[g], y[g])
                    boot[s, budget, arm, key] = C.boot_metric(C.weighted_auroc, sc[g], y[g], col[g], counts)
                    if s == seeds[0]:
                        strata_rows.append({"cohort": cohort, "budget": budget, "severity": key[0], "level": key[1],
                                            "range": ranges.get(key, ""), "arm": arm, "patients": len(np.unique(subj[g])),
                                            "windows": int(g.sum()), "prevalence": float(y[g].mean())})
                per = pd.DataFrame({"subject": subj, "y": y, "score": sc, "pred": d.pred.to_numpy()})
                for subject, p in per.groupby("subject"):
                    npos, nneg = int(p.y.sum()), int((1 - p.y).sum())
                    ppa[s, budget, arm, subject] = (
                        C.auroc(p.score.to_numpy(), p.y.to_numpy()) if min(npos, nneg) >= MIN_PER_CLASS else np.nan,
                        100 * p.y.mean(), 100 * p.pred.mean(), npos, len(p))
        # strata: means over repetitions
        for r in strata_rows:
            if r["cohort"] != cohort:
                continue
            key = (r["severity"], r["level"])
            ks = [(s, r["budget"], r["arm"], key) for s in seeds]
            if not all(k in point for k in ks):
                continue
            lo, hi = C.interval(np.mean([boot[k] for k in ks], axis=0))
            vals = [point[k] for k in ks]
            r.update({"auroc": float(np.mean(vals)), "auroc_lo": lo, "auroc_hi": hi,
                      "auroc_seed_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan, "n_seeds": len(seeds)})
        contrasts = {"nfogq_t1_minus_t3": (("nfogq", "low"), ("nfogq", "high")),
                     "updrs3on_t1_minus_t3": (("updrs3_on", "low"), ("updrs3_on", "high")),
                     "med_on_minus_off": (("medication_within", "on"), ("medication_within", "off")),
                     "sex_female_minus_male": (("sex", "female"), ("sex", "male"))}
        arms_b = sorted({(k[1], k[2]) for k in point})
        for budget, arm in arms_b:
            for name, (ka, kb) in contrasts.items():
                ka_s, kb_s = [(s, budget, arm, ka) for s in seeds], [(s, budget, arm, kb) for s in seeds]
                if not all(k in point for k in ka_s + kb_s):
                    continue
                d_seed = [point[x] - point[z] for x, z in zip(ka_s, kb_s)]
                d_boot = np.mean([boot[x] - boot[z] for x, z in zip(ka_s, kb_s)], axis=0)
                lo, hi = C.interval(d_boot)
                contrast_rows.append({"cohort": cohort, "budget": budget, "arm": arm, "contrast": name,
                                      "d_mean": float(np.mean(d_seed)), "ci_lo": lo, "ci_hi": hi,
                                      "p": C.two_sided_p(d_boot), "seeds_positive": int(np.sum(np.array(d_seed) > 0)),
                                      "n_seeds": len(seeds)})
        # per-patient values, mean over repetitions, and their correlations with severity
        ppd = pd.DataFrame([{"budget": k[1], "arm": k[2], "subject": k[3], "auroc_patient": v[0], "ptf_true": v[1],
                             "ptf_pred": v[2], "n_fog": v[3], "n_windows": v[4]} for k, v in ppa.items()])
        ppm = ppd.groupby(["budget", "arm", "subject"]).agg(
            auroc_patient=("auroc_patient", "mean"), auroc_patient_sd=("auroc_patient", "std"),
            ptf_true=("ptf_true", "mean"), ptf_pred=("ptf_pred", "mean"), n_fog=("n_fog", "first"),
            n_windows=("n_windows", "first")).reset_index()
        ppm["ptf_abs_err"] = (ppm.ptf_pred - ppm.ptf_true).abs()
        ppm = ppm.join(pt[CORR_SEV + ["sex"]], on="subject")
        pp_rows.append(ppm.assign(cohort=cohort))
        for budget in BUDGETS:
            pb = ppm[ppm.budget == budget]
            targets = [(arm, "auroc_patient", g.set_index("subject").auroc_patient) for arm, g in pb.groupby("arm")]
            targets += [(arm, "ptf_abs_err", g.set_index("subject").ptf_abs_err) for arm, g in pb.groupby("arm")]
            h = pb[pb.arm == "harnet5"].set_index("subject").auroc_patient
            f = pb[pb.arm == "feat60_fi"].set_index("subject").auroc_patient
            if len(h) and len(f):
                targets.append(("harnet5 minus feat60_fi", "auroc_patient_diff", h - f))
            for arm, outcome, v in targets:
                for sev in CORR_SEV:
                    xy = pd.concat([pt.loc[v.index, sev], v], axis=1).dropna().to_numpy()
                    if len(xy) < 8:
                        continue
                    idx = rng.integers(0, len(xy), (a.n_boot, len(xy)))
                    bt = spearman_boot(xy[:, 0], xy[:, 1], idx)
                    lo, hi = C.interval(bt)
                    corr_rows.append({"cohort": cohort, "budget": budget, "arm": arm, "outcome": outcome,
                                      "severity": sev, "spearman": float(spearmanr(xy[:, 0], xy[:, 1]).statistic),
                                      "lo": lo, "hi": hi, "patients": len(xy), "n_seeds": len(seeds)})
        print(f"{cohort} done ({time.time() - t_start:.0f} s)", flush=True)

    strata = pd.DataFrame(strata_rows).dropna(subset=["auroc"])
    contrasts = pd.DataFrame(contrast_rows)
    corr = pd.DataFrame(corr_rows)
    summary = {"commit": C.git_commit(), "date": time.strftime("%Y-%m-%d"), "labels": labels,
               "runs": {str(s): cfgs[s]["run"] for s in seeds}, "run_commits": {str(s): cfgs[s]["commit"] for s in seeds},
               "n_boot": a.n_boot, "seeds": seeds, "min_per_class": MIN_PER_CLASS,
               "seconds": round(time.time() - t_start, 1)}
    show = contrasts[contrasts.budget == "all"].pivot_table(index=["cohort", "arm"], columns="contrast",
                                                            values="d_mean")
    report = f"""# Severity across repetitions and by sex (analysis f, post hoc, exploratory)

Written by `scripts/paper_severity.py` on {summary['date']} at commit {summary['commit']}; labels: {labels};
{len(seeds)} repetitions; {a.n_boot:,} patient resamples per cohort shared by repetitions and approaches.

## Contrasts in pooled AUROC, all training patients (mean over repetitions)

{C.md_table(show)}

Intervals: `contrasts.csv`; strata: `strata.csv`; correlations: `correlations.csv`; per patient: `per_patient.csv`.
"""
    out = C.write_outputs(NAME, summary, {"strata.csv": strata, "contrasts.csv": contrasts, "correlations.csv": corr,
                                          "per_patient.csv": pd.concat(pp_rows, ignore_index=True)},
                          report, base=a.out_base)
    print(report)
    print(f"wrote {out} in {summary['seconds']} s")


if __name__ == "__main__":
    main()
