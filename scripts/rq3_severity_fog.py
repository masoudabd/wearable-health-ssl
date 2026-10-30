"""RQ3 on the FOG contest data: does detection get worse for more impaired patients, and does the gap
between a frozen foundation model and handcrafted features depend on severity?

Reads the out-of-fold scores of scripts/rq12_fog_transfer.py. The run did not store window indices,
so they are rebuilt from the same deterministic subject split (same function, seed, fold count) and
checked row by row against the stored patient IDs before window metadata (recording, medication state,
visit) and clinical scores are attached.

Severity measures (no Hoehn and Yahr stage exists in this dataset): MDS-UPDRS III in the ON state
(available for nearly all patients), NFOG-Q (FOG severity questionnaire), years since diagnosis.
Patient-level values are means over the patient's visits. Medication state is per recording.

All analyses are exploratory; intervals resample patients; no multiplicity correction is applied.

Usage: python scripts/rq3_severity_fog.py [--run results/rq12_fog_transfer] [--quick]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from whssl.data import fog_contest as F  # noqa: E402
from whssl.eval.metrics import subject_bootstrap_ci  # noqa: E402
from whssl.eval.splits import subject_kfold  # noqa: E402

ARMS = ["harnet5", "feat60_fi", "supervised", "ssl_indomain", "feat60", "harnet5_rand"]
MIN_PER_CLASS = 10  # windows of each class a patient needs for a per-patient AUROC


def attach_metadata(oof: pd.DataFrame, W: dict, folds: int, seed: int, quick: bool) -> pd.DataFrame:
    parts = []
    for cohort, oc in oof.groupby("cohort", sort=False):
        m = np.flatnonzero(W["cohort"] == cohort)
        y, g = W["y"][m].astype(int), W["subject"][m]
        if quick:  # mirror the --quick subset of rq12
            keep = np.isin(g, np.unique(g)[:24])
            m, y, g = m[keep], y[keep], g[keep]
        te_order = np.concatenate([te for _, te in subject_kfold(y, g, folds, seed)])
        win = m[te_order]  # global window index for each out-of-fold row, in stored order
        for (budget, arm), d in oc.groupby(["budget", "arm"], sort=False):
            assert len(d) == len(win), f"{cohort}/{budget}/{arm}: {len(d)} rows vs {len(win)} rebuilt"
            assert (d.subject.to_numpy() == W["subject"][win]).all(), f"{cohort}/{budget}/{arm}: patient order mismatch"
            assert (d.y.to_numpy() == W["y"][win]).all(), f"{cohort}/{budget}/{arm}: label mismatch"
            parts.append(d.assign(win=win, rec=W["rec"][win], medication=W["medication"][win],
                                  visit=W["visit"][win], t0=W["t0"][win]))
    return pd.concat(parts, ignore_index=True)


def patient_table() -> pd.DataFrame:
    s = F.subjects()
    return s.groupby("Subject").agg(updrs3_on=("UPDRSIII_On", "mean"), updrs3_off=("UPDRSIII_Off", "mean"),
                                    nfogq=("NFOGQ", "mean"), years_dx=("YearsSinceDx", "mean"),
                                    age=("Age", "mean")).rename_axis("subject")


def per_patient(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subj, p in d.groupby("subject"):
        n_pos, n_neg = int(p.y.sum()), int((1 - p.y).sum())
        auc = roc_auc_score(p.y, p.score) if min(n_pos, n_neg) >= MIN_PER_CLASS else np.nan
        rows.append({"subject": subj, "n_windows": len(p), "n_fog": n_pos, "auroc_patient": auc,
                     "ptf_true": p.y.mean() * 100, "ptf_pred": p.pred.mean() * 100})
    out = pd.DataFrame(rows).set_index("subject")
    out["ptf_abs_err"] = (out.ptf_pred - out.ptf_true).abs()
    return out


def spearman_ci(x, y, n_boot=2000, seed=0):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[ok], np.asarray(y)[ok]
    if len(x) < 8:
        return np.nan, np.nan, np.nan, len(x)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if np.unique(x[i]).size > 2 and np.unique(y[i]).size > 2:
            boots.append(spearmanr(x[i], y[i]).statistic)
    lo, hi = np.nanquantile(boots, [0.025, 0.975])
    return spearmanr(x, y).statistic, lo, hi, len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "results" / "rq12_fog_transfer"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--budgets", nargs="+", default=["all", "10"])
    args = ap.parse_args()
    run = Path(args.run)
    folds, seed = (2, 0) if args.quick else (5, 0)
    cfg_path = run / "config.json"
    if cfg_path.exists():  # take the split from the run itself (seeds other than 0 reshuffle the folds)
        cfg = json.loads(cfg_path.read_text())
        folds, seed, args.quick = cfg["folds"], cfg["seed"], cfg["quick"]
    out = run / "rq3"
    out.mkdir(exist_ok=True)

    oof = pd.read_parquet(run / "out_of_fold_scores.parquet")
    W = F.load_windows(30.0, 5.0, 2.5)
    oof = attach_metadata(oof, W, folds, seed, args.quick)
    pt = patient_table()
    print(f"rebuilt window indices for {len(oof):,} out-of-fold rows; all patient IDs and labels match")

    strata_rows, corr_rows, pp_all = [], [], []
    for cohort in oof.cohort.unique():
        for budget in [b for b in args.budgets if b in set(oof.budget)]:
            d0 = oof[(oof.cohort == cohort) & (oof.budget == budget)]
            # tertiles of patient severity, cut on this cohort's patients
            pats = pt.loc[pt.index.isin(d0.subject.unique())]
            for sev in ["updrs3_on", "nfogq"]:
                q = pats[sev].quantile([1 / 3, 2 / 3]).to_numpy()
                tert = pd.cut(pats[sev], [-np.inf, q[0], q[1], np.inf], labels=["low", "mid", "high"])
                for arm in [a for a in ARMS if a in set(d0.arm)]:
                    d = d0[d0.arm == arm]
                    for level in ["low", "mid", "high"]:
                        s = d[d.subject.map(tert) == level]
                        if s.y.nunique() < 2:
                            continue
                        auc, lo, hi = subject_bootstrap_ci(s.y.to_numpy(), s.score.to_numpy(), s.subject.to_numpy(), n_boot=1000)
                        strata_rows.append({"cohort": cohort, "budget": budget, "severity": sev, "level": level,
                                            "range": f"{pats[sev][tert == level].min():.0f}-{pats[sev][tert == level].max():.0f}",
                                            "arm": arm, "patients": s.subject.nunique(), "windows": len(s),
                                            "prevalence": s.y.mean(), "auroc": auc, "auroc_lo": lo, "auroc_hi": hi})
            for arm in [a for a in ARMS if a in set(d0.arm)]:
                d = d0[d0.arm == arm]
                # medication state is per recording, so it is a within-patient comparison
                for med in ["on", "off"]:
                    s = d[d.medication == med]
                    if s.y.nunique() == 2:
                        auc, lo, hi = subject_bootstrap_ci(s.y.to_numpy(), s.score.to_numpy(), s.subject.to_numpy(), n_boot=1000)
                        strata_rows.append({"cohort": cohort, "budget": budget, "severity": "medication", "level": med,
                                            "range": "", "arm": arm, "patients": s.subject.nunique(), "windows": len(s),
                                            "prevalence": s.y.mean(), "auroc": auc, "auroc_lo": lo, "auroc_hi": hi})
                pp = per_patient(d).join(pt, how="left")
                pp_all.append(pp.assign(cohort=cohort, budget=budget, arm=arm).reset_index())
                for target in ["auroc_patient", "ptf_abs_err"]:
                    for sev in ["updrs3_on", "nfogq", "years_dx"]:
                        r, lo, hi, n = spearman_ci(pp[sev].to_numpy(), pp[target].to_numpy())
                        corr_rows.append({"cohort": cohort, "budget": budget, "arm": arm, "outcome": target,
                                          "severity": sev, "spearman": r, "lo": lo, "hi": hi, "patients": n})
            # does the foundation-model advantage over features change with severity?
            a = per_patient(d0[d0.arm == "harnet5"]).auroc_patient
            b = per_patient(d0[d0.arm == "feat60_fi"]).auroc_patient
            diff = (a - b).rename("auroc_diff").to_frame().join(pt, how="left")
            for sev in ["updrs3_on", "nfogq", "years_dx"]:
                r, lo, hi, n = spearman_ci(diff[sev].to_numpy(), diff.auroc_diff.to_numpy())
                corr_rows.append({"cohort": cohort, "budget": budget, "arm": "harnet5 minus feat60_fi",
                                  "outcome": "auroc_patient_diff", "severity": sev, "spearman": r, "lo": lo, "hi": hi,
                                  "patients": n})

    strata = pd.DataFrame(strata_rows)
    corr = pd.DataFrame(corr_rows)
    strata.to_csv(out / "strata_auroc.csv", index=False)
    corr.to_csv(out / "severity_correlations.csv", index=False)
    pd.concat(pp_all, ignore_index=True).to_csv(out / "per_patient.csv", index=False)
    pd.set_option("display.width", 220)
    show = strata[strata.budget == "all"].pivot_table(index=["cohort", "severity", "level"], columns="arm",
                                                      values="auroc").round(3)
    print("\nAUROC by severity stratum (budget all):\n", show.to_string())
    print("\nSpearman correlations, budget all:\n",
          corr[(corr.budget == "all")].round(3).to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
