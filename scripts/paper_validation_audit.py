"""Post hoc analysis j: what the validation patients carried, and what choosing on them cost.

With a budget of 5 labeled patients, one fifth of the budget is a single validation patient, which chooses every
decision threshold and the classifier's regularization. The external review asked how the pipeline behaves when that
patient has few or no FOG windows, how variable the chosen thresholds are, and how much the choice costs against a
threshold chosen with full knowledge of the test patients.

Three things are measured, from the stored validation and out-of-fold scores of the equal-effort runs (no retraining):

  composition   per cohort, budget and repetition: validation patients per fold, their windows, their FOG windows,
                and how many validation sets contained no FOG window at all
  thresholds    the spread of the chosen threshold over folds and repetitions, as a share of the score range, so
                arms on different score scales can be compared
  cost          positive-class F1 on the test patients at the chosen threshold, against the best F1 any single
                threshold could have reached on those same test windows (an oracle that cannot be used in practice);
                the gap is what choosing on validation patients cost, and it is reported by budget

The oracle is an upper bound computed on the test patients and is not an alternative method; it exists only to size
the gap. Thresholds are chosen per fold, so F1 and the oracle are computed per fold and then averaged.

  validation_composition.csv, threshold_spread.csv, threshold_cost.csv, summary.json, report.md

Usage: .venv/bin/python scripts/paper_validation_audit.py [--runs RUN ...]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import paper_common as C

NAME = "validation_audit"
RUNS = [f"results/rq12_tuned_ft_seed{s}" for s in range(5)]
ARMS = ["harnet5", "feat60_fi", "ssl_indomain", "supervised", "harnet5_rand"]


def f1_at(y: np.ndarray, pred: np.ndarray) -> float:
    tp = float((pred & (y == 1)).sum())
    if tp == 0:
        return 0.0
    return 2 * tp / (2 * tp + float((pred & (y == 0)).sum()) + float((~pred & (y == 1)).sum()))


def best_f1(y: np.ndarray, score: np.ndarray) -> float:
    """Highest positive-class F1 reachable by any single threshold on these windows (an oracle upper bound)."""
    o = np.argsort(-score, kind="mergesort")
    ys = y[o] == 1
    tp = np.cumsum(ys)
    fp = np.cumsum(~ys)
    fn = ys.sum() - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = 2 * tp / (2 * tp + fp + fn)
    return float(np.nanmax(f1)) if len(f1) else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=RUNS)
    ap.add_argument("--out-base", default=None)
    a = ap.parse_args()
    t_start = time.time()

    comp, spread, cost = [], [], []
    for run in a.runs:
        seed = int(str(run).rstrip("/").split("seed")[-1])
        v = pd.read_parquet(C.ROOT / run / "validation_scores.parquet")
        o = pd.read_parquet(C.ROOT / run / "out_of_fold_scores.parquet",
                            columns=["cohort", "fold", "budget", "arm", "y", "score", "pred"])
        v = v[v["arm"].isin(ARMS)]
        o = o[o["arm"].isin(ARMS)]

        g = (v.groupby(["cohort", "budget", "fold", "arm"])
             .agg(n_val_patients=("subject", "nunique"), n_windows=("y", "size"), n_fog=("y", "sum"),
                  threshold=("threshold", "first")).reset_index())
        g["seed"] = seed
        comp.append(g)

        for (cohort, budget, arm), d in g.groupby(["cohort", "budget", "arm"]):
            # the score scale differs by arm, so the spread is given relative to that arm's own score range
            sc = v[(v.cohort == cohort) & (v["budget"] == budget) & (v["arm"] == arm)].score
            rng = float(sc.max() - sc.min()) or 1.0
            spread.append({"cohort": cohort, "budget": budget, "arm": arm, "seed": seed,
                           "threshold_min": float(d.threshold.min()), "threshold_max": float(d.threshold.max()),
                           "threshold_range_rel": float((d.threshold.max() - d.threshold.min()) / rng),
                           "n_folds": int(len(d))})

        for (cohort, budget, arm), d in o.groupby(["cohort", "budget", "arm"]):
            chosen, oracle = [], []
            for _, df in d.groupby("fold"):
                y = df.y.to_numpy().astype(int)
                chosen.append(f1_at(y, df.pred.to_numpy().astype(bool)))
                oracle.append(best_f1(y, df.score.to_numpy()))
            cost.append({"cohort": cohort, "budget": budget, "arm": arm, "seed": seed,
                         "f1_chosen": float(np.mean(chosen)), "f1_oracle": float(np.mean(oracle)),
                         "f1_gap": float(np.mean(oracle) - np.mean(chosen)), "n_folds": len(chosen)})
        print(f"seed {seed} done ({time.time() - t_start:.0f} s)", flush=True)

    composition = pd.concat(comp, ignore_index=True)
    spread_df = pd.DataFrame(spread)
    cost_df = pd.DataFrame(cost)

    # one row per cohort and budget, averaged over arms, folds and repetitions
    comp_sum = (composition.groupby(["cohort", "budget"])
                .agg(val_patients=("n_val_patients", "median"), windows_median=("n_windows", "median"),
                     fog_windows_min=("n_fog", "min"), fog_windows_median=("n_fog", "median"),
                     sets_without_fog=("n_fog", lambda s: int((s == 0).sum())),
                     n_sets=("n_fog", "size")).reset_index())
    cost_sum = (cost_df.groupby(["cohort", "budget"])
                .agg(f1_chosen=("f1_chosen", "mean"), f1_oracle=("f1_oracle", "mean"),
                     f1_gap=("f1_gap", "mean"), gap_sd=("f1_gap", "std")).reset_index())
    spread_sum = (spread_df.groupby(["cohort", "budget"])
                  .agg(threshold_range_rel=("threshold_range_rel", "median")).reset_index())

    summary = {"commit": C.git_commit(), "date": time.strftime("%Y-%m-%d"), "runs": a.runs, "arms": ARMS,
               "validation_sets_without_fog": int((composition.n_fog == 0).sum()),
               "validation_sets": int(len(composition)),
               "note": "oracle F1 is an upper bound computed on the test patients, not a usable method",
               "seconds": round(time.time() - t_start, 1)}
    report = f"""# Validation patients: what they carried and what choosing on them cost (analysis j, post hoc)

Written by `scripts/paper_validation_audit.py` on {summary['date']} at commit {summary['commit']};
{len(a.runs)} repetitions; arms {', '.join(ARMS)}.

**No validation set was without a FOG window**: {summary['validation_sets_without_fog']} of
{summary['validation_sets']} validation sets across every cohort, budget, fold, arm and repetition contained at
least one. The label budget is drawn until both classes are present, and a validation patient is carved from it
stratified on the label, which is why.

## What the validation patients carried

{C.md_table(comp_sum, floatfmt=".0f")}

## Cost of choosing the threshold on them

Positive-class F1 on the test patients at the chosen threshold, against the best any single threshold could have
reached on those test windows. The oracle is not attainable in practice; the gap sizes what the choice cost.

{C.md_table(cost_sum, floatfmt=".3f")}

## Spread of the chosen threshold over folds, as a share of the arm's score range

{C.md_table(spread_sum, floatfmt=".3f")}
"""
    out = C.write_outputs(NAME, summary,
                          {"validation_composition.csv": composition, "threshold_spread.csv": spread_df,
                           "threshold_cost.csv": cost_df, "composition_summary.csv": comp_sum,
                           "cost_summary.csv": cost_sum}, report, base=a.out_base)
    print(report)
    print(f"wrote {out} in {summary['seconds']} s")


if __name__ == "__main__":
    main()
