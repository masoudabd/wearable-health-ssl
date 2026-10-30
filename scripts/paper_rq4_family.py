"""Post hoc (writing brief 8.5): one Holm family over every RQ4 paired test.

RQ4 ran as four experiments, each with its own paired tests and its own Holm adjustment: daily-living pretraining
(results/rq4_seeds_mps_fixed/rq4_paired_tests.csv, 28 tests), the equal-compute control
(results/rq4_equal_compute_seeds/paired_tests.csv, 24), the walking-like subset (results/rq4_gait_seeds/
paired_tests.csv, 24) and the longer adaptation of the foundation model (results/rq4_long_adapt_seeds/paired_tests.csv,
20; three repetitions, added 2026-10-13). The later three were designed after the first had been seen, so a claim from
any of them is also judged against the whole RQ4 family. All three use the same patient resamples (paired_tests.py: sorted
patients, patient_counts seed 0, the same --n-boot) and runs that reproduce each other's shared arms exactly, so a
comparison tested in two files has the same P in both; it enters the family once. Nothing is recomputed from
scores: the unadjusted P values are read from the three files and Holm-adjusted together.

Outputs (results/paper_derived/rq4_family/): holm_all.csv (every unique test with its experiment, its P adjusted
within that experiment and over all RQ4 tests), summary.json, report.md.

Usage: .venv/bin/python scripts/paper_rq4_family.py   (seconds)
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import paper_common as C

NAME = "rq4_family"
FILES = {"rq4": "results/rq4_seeds_mps_fixed/rq4_paired_tests.csv",
         "equal_compute": "results/rq4_equal_compute_seeds/paired_tests.csv",
         "walking": "results/rq4_gait_seeds/paired_tests.csv",
         "long_adaptation": "results/rq4_long_adapt_seeds/paired_tests.csv"}
KEY = ["cohort", "budget", "comparison"]


def with_suffix_dir(path: str, suffix: str) -> str:
    """results/<run>/<file>.csv -> results/<run><suffix>/<file>.csv"""
    parent, _, name = path.rpartition("/")
    return f"{parent}{suffix}/{name}" if suffix else path


def n_boot_of(path: str) -> int:
    """Resamples recorded by paired_tests.py next to its table; 2,000 for tables written before it recorded them."""
    cfg = (C.ROOT / path).with_name((C.ROOT / path).stem + "_config.json")
    return int(json.loads(cfg.read_text())["n_boot"]) if cfg.exists() else 2000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="", help="appended to each input run directory, e.g. _b20k")
    ap.add_argument("--out", default=NAME, help="output directory under results/paper_derived/")
    a = ap.parse_args()
    files = {exp: with_suffix_dir(path, a.suffix) for exp, path in FILES.items()}
    n_boots = {exp: n_boot_of(path) for exp, path in files.items()}
    if len(set(n_boots.values())) != 1:
        raise SystemExit(f"inputs do not share one resample count: {n_boots}")
    n_boot = next(iter(n_boots.values()))
    frames = []
    for exp, path in files.items():
        d = pd.read_csv(C.ROOT / path, dtype={"budget": str})
        frames.append(d.assign(experiment=exp, p_holm_experiment=d.p_holm, n_tests_experiment=len(d)))
    allt = pd.concat(frames, ignore_index=True)
    dup = allt[allt.duplicated(KEY, keep=False)]
    for _, g in dup.groupby(KEY):  # a comparison in two files must be the same test
        for col in ("d_mean", "ci_lo", "ci_hi", "p"):
            if not np.allclose(g[col], g[col].iloc[0], rtol=0, atol=1e-12):
                raise SystemExit(f"{tuple(g[KEY].iloc[0])}: {col} differs between experiments")
    first = allt.drop_duplicates(KEY, keep="first").copy()
    also = dup.groupby(KEY).experiment.apply(lambda e: ";".join(e.iloc[1:])).rename("also_in")
    first = first.merge(also, on=KEY, how="left")
    first["p_holm_all"] = C.holm(first.p.to_numpy())
    cols = KEY + ["experiment", "also_in", "d_mean", "ci_lo", "ci_hi", "p", "p_holm_experiment",
                  "n_tests_experiment", "p_holm_all", "seeds_positive", "n_seeds", "n_patients"]
    first = first[cols]
    out = C.DERIVED / a.out
    out.mkdir(parents=True, exist_ok=True)
    first.to_csv(out / "holm_all.csv", index=False)
    n = len(first)
    surv = first[first.p_holm_all < 0.05]
    summary = {"commit": C.git_commit(), "inputs": files, "n_tests_listed": int(len(allt)), "n_tests_unique": n,
               "duplicates": int(len(allt) - n), "n_boot": n_boot,
               "min_attainable_p_holm_all": min(1.0, n / n_boot),
               "surviving_holm_all": surv[KEY + ["d_mean", "ci_lo", "ci_hi", "p_holm_all"]].to_dict("records")}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    lines = [f"# RQ4 tests under one Holm family\n", f"Commit {summary['commit']}. {len(allt)} tests listed in the "
             f"three files, {n} unique ({summary['duplicates']} listed twice, identical). Smallest attainable "
             f"Holm-adjusted P with {n_boot:,} resamples: {summary['min_attainable_p_holm_all']:.3f}.\n",
             "Tests with Holm-adjusted P below 0.05 over all RQ4 tests:\n",
             "```", surv.round(4).to_string(index=False), "```\n\nAll tests:\n", "```", first.round(4).to_string(index=False), "```"]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    pd.set_option("display.width", 220)
    print(surv.round(4).to_string(index=False))
    print(f"{n} unique tests; wrote {out}")


if __name__ == "__main__":
    main()
