"""Compare two RQ1/RQ2 runs, e.g. the laptop run and its rerun on a new machine (HANDOFF section 6, step 1).

Levels, strictest first:
  1. Out-of-fold scores row by row. Arms without training (feat60, feat60_fi, harnet5, harnet5_rand probes)
     should match to float precision when data, folds, features and embeddings reproduce. Trained arms
     (ssl_indomain, supervised) can drift with the platform's floating-point order.
  2. Per-fold AUROC (and whether the label-budget subsampling drew the same number of patients).
  3. Pooled AUROC with its patient-bootstrap interval: is each estimate inside the other run's interval?
  4. RQ3 tables (rq3/*.csv), when both runs have them: largest absolute difference per numeric column.

Usage: python scripts/compare_runs.py results/rq12_fog_transfer_laptop results/rq12_fog_transfer
Writes <second run>/comparison_vs_<first run name>.csv (the level 3 table).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROW_KEYS = ["cohort", "fold", "budget", "arm", "subject", "y"]
BUDGET_ORDER = {"5": 0, "10": 1, "20": 2, "all": 3}


def describe(run: Path) -> str:
    c = json.loads((run / "config.json").read_text())
    return f"{run}  commit {c['commit']}{' (dirty)' if c['dirty'] else ''}"


def level1(a: Path, b: Path) -> None:
    oa, ob = pd.read_parquet(a / "out_of_fold_scores.parquet"), pd.read_parquet(b / "out_of_fold_scores.parquet")
    # compare values, not dtypes: Windows numpy stores astype(int) labels as int32, macOS and Linux as int64
    same = len(oa) == len(ob) and all(np.array_equal(oa[c].to_numpy(), ob[c].to_numpy()) for c in ROW_KEYS)
    print(f"\n[1] out-of-fold rows: {len(oa):,} vs {len(ob):,}; same windows, folds, labels and order: {same}")
    if not same:
        print("    row sets differ, so windowing or fold assignment changed: investigate before anything else")
        return
    d = oa.assign(score_b=ob["score"].to_numpy(), pred_b=ob["pred"].to_numpy())
    t = d.groupby(["cohort", "arm"])[["score", "score_b", "pred", "pred_b"]].apply(lambda g: pd.Series({
        "max_abs_score_diff": float(np.abs(g.score - g.score_b).max()),
        "score_corr": float(np.corrcoef(g.score, g.score_b)[0, 1]),
        "pred_agreement": float((g.pred == g.pred_b).mean())}))
    print(t.to_string(float_format=lambda x: f"{x:.3g}"))


def level2(a: Path, b: Path) -> None:
    fa = pd.read_csv(a / "per_fold.csv", dtype={"budget": str})
    fb = pd.read_csv(b / "per_fold.csv", dtype={"budget": str})
    f = fa.merge(fb, on=["cohort", "fold", "budget", "arm"], suffixes=("_a", "_b"))
    same_n = bool((f.n_train_patients_a == f.n_train_patients_b).all())
    f["abs_d"] = (f.auroc_b - f.auroc_a).abs()
    print(f"\n[2] per-fold AUROC, {len(f)} matched rows; same patients drawn per budget: {same_n}")
    print("    largest |difference| per arm:")
    print(f.groupby(["cohort", "arm"]).abs_d.max().unstack().round(4).to_string())


def level3(a: Path, b: Path) -> pd.DataFrame:
    keys = ["cohort", "budget", "arm"]
    sa = pd.read_csv(a / "summary.csv", dtype={"budget": str})
    sb = pd.read_csv(b / "summary.csv", dtype={"budget": str})
    s = sa.merge(sb, on=keys, suffixes=("_a", "_b"))
    s["d_auroc"] = s.auroc_b - s.auroc_a
    s["b_in_a_ci"] = s.auroc_b.between(s.auroc_lo_a, s.auroc_hi_a)
    s["a_in_b_ci"] = s.auroc_a.between(s.auroc_lo_b, s.auroc_hi_b)
    s["d_auprc_mean"] = s.auprc_mean_b - s.auprc_mean_a
    s["d_ptf_spearman_mean"] = s.ptf_spearman_mean_b - s.ptf_spearman_mean_a
    s = s.sort_values(keys, key=lambda c: c.map(BUDGET_ORDER) if c.name == "budget" else c)
    cols = keys + ["auroc_a", "auroc_lo_a", "auroc_hi_a", "auroc_b", "auroc_lo_b", "auroc_hi_b", "d_auroc",
                   "b_in_a_ci", "a_in_b_ci", "d_auprc_mean", "d_ptf_spearman_mean"]
    out = s[cols].reset_index(drop=True)
    inside = int(out.b_in_a_ci.sum())
    print(f"\n[3] pooled AUROC, {len(out)} cohort x budget x arm cells: largest |difference| "
          f"{out.d_auroc.abs().max():.4f}, mean {out.d_auroc.abs().mean():.4f}; "
          f"new estimate inside the old 95% interval in {inside} of {len(out)}")
    print(out[keys + ["auroc_a", "auroc_b", "d_auroc", "b_in_a_ci"]].round(3).to_string(index=False))
    return out


def level4(a: Path, b: Path) -> None:
    for name in ("strata_auroc.csv", "severity_correlations.csv", "per_patient.csv"):
        pa, pb = a / "rq3" / name, b / "rq3" / name
        if not (pa.exists() and pb.exists()):
            continue
        ta, tb = pd.read_csv(pa, dtype={"budget": str}), pd.read_csv(pb, dtype={"budget": str})
        keys = [c for c in ta.columns if ta[c].dtype == object]
        m = ta.merge(tb, on=keys, suffixes=("_a", "_b"))
        diffs = {c: float((m[f"{c}_b"] - m[f"{c}_a"]).abs().max()) for c in ta.columns if c not in keys}
        print(f"\n[4] rq3/{name}: {len(ta)} vs {len(tb)} rows, {len(m)} matched on {keys}")
        print("    largest |difference|: " + ", ".join(f"{c} {v:.4g}" for c, v in diffs.items()))


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    a, b = Path(sys.argv[1]), Path(sys.argv[2])
    pd.set_option("display.width", 220)
    print(f"A: {describe(a)}\nB: {describe(b)}")
    level1(a, b)
    level2(a, b)
    out = level3(a, b)
    level4(a, b)
    path = b / f"comparison_vs_{a.name}.csv"
    out.to_csv(path, index=False)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
