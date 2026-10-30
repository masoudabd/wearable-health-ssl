"""Post hoc summary (writing brief 8.5) of the cross-cohort transfer runs (scripts/cross_cohort.py).

For each axis mode (none, ap: AP harmonized, apml: AP and ML harmonized), direction (lab to home, home to lab), label
budget and representation: the pooled AUROC on the target cohort, mean and SD over the five repetitions; beside it the
within-cohort AUROC of the same representation and budget from the RQ1/RQ2 runs of the same seeds (fixed-label runs
for the default probes, equal-effort runs for the tuned probes), which the paired tests compare (transfer gap).
Read-only on results; recomputes nothing but the means.

Outputs (results/paper_derived/cross_cohort/): summary.csv, summary.json, report.md.
Usage: .venv/bin/python scripts/paper_cross_cohort.py   (seconds)
"""
from __future__ import annotations

import time

import pandas as pd

import paper_common as C

NAME = "cross_cohort"
MODES = {"none": "_x", "ap": "_xap", "apml": "_xrot"}
ARMS = ["feat60_fi", "feat60_fi_tc", "harnet5", "harnet5_tc", "feat60", "harnet5_rand"]


def main():
    t0 = time.time()
    rows = []
    within = []
    for s in range(5):
        for run in (f"results/rq12_fog_transfer_mps_fixed_seed{s}", f"results/rq12_tuned_ft_seed{s}"):
            d = pd.read_csv(C.ROOT / run / "summary.csv", dtype={"budget": str})
            within.append(d[d.arm.isin(ARMS)][["cohort", "budget", "arm", "auroc"]].assign(seed=s))
    within = pd.concat(within).drop_duplicates(["cohort", "budget", "arm", "seed"])
    w = within.groupby(["cohort", "budget", "arm"]).auroc.agg(["mean", "std"]).rename(
        columns={"mean": "within_auroc_mean", "std": "within_auroc_sd"}).reset_index()
    for mode, suffix in MODES.items():
        for s in range(5):
            d = pd.read_csv(C.ROOT / f"results/cross_cohort_{mode}_seed{s}" / "per_budget.csv", dtype={"budget": str})
            d = d[d.arm.str.endswith(suffix)].copy()
            d["base_arm"] = d.arm.str[: -len(suffix)]
            rows.append(d.assign(mode=mode, seed=s))
    x = pd.concat(rows)
    out = (x.groupby(["mode", "source", "cohort", "budget", "base_arm"])
           .agg(transfer_auroc_mean=("auroc", "mean"), transfer_auroc_sd=("auroc", "std"), n_seeds=("seed", "nunique"),
                n_source_patients=("n_source_patients", "mean")).reset_index())
    out = out.merge(w, left_on=["cohort", "budget", "base_arm"], right_on=["cohort", "budget", "arm"], how="left")
    out = out.drop(columns="arm").rename(columns={"cohort": "target"})
    out["within_minus_transfer"] = out.within_auroc_mean - out.transfer_auroc_mean
    summary = {"commit": C.git_commit(), "date": time.strftime("%Y-%m-%d"), "modes": list(MODES),
               "seeds": sorted(x.seed.unique().tolist()), "seconds": round(time.time() - t0, 1)}
    show = out[(out.base_arm.isin(["harnet5", "feat60_fi"])) & (out["mode"].isin(["none", "ap"]))]
    report = f"""# Cross-cohort transfer, summary (post hoc)

Written by `scripts/paper_cross_cohort.py` on {summary['date']} at commit {summary['commit']}. Pooled AUROC on the
target cohort, mean over five repetitions; within-cohort reference from the RQ1/RQ2 runs of the same seeds.

{C.md_table(show.set_index(["mode", "source", "budget", "base_arm"])[["transfer_auroc_mean", "within_auroc_mean", "within_minus_transfer"]])}
"""
    C.write_outputs(NAME, summary, {"summary.csv": out}, report)
    print(report)


if __name__ == "__main__":
    main()
