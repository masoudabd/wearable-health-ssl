"""Paired patient-bootstrap tests of pooled-AUROC differences between arms, averaged over seeds (RQ1/RQ2).

For each cohort, one set of patient resamples (draws with replacement) is shared by every seed, arm and budget, so
each difference is paired. Per replicate the difference (arm A minus arm B) is computed within each seed on that
seed's out-of-fold scores and averaged over seeds; the 2.5 and 97.5 percentiles give the 95% interval and the tail
mass at zero a two-sided p-value. Holm adjusts over all tests. Primary, pre-specified: harnet5 minus feat60_fi.
Seeds average out training and partition randomness; the interval reflects patient sampling.

Usage: python scripts/paired_tests.py RUN [RUN ...] [--n-boot 2000] [--out results/rq12_seeds_mps]
Writes <out>/paired_tests.csv.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from whssl.eval.paired import boot_auroc, holm, patient_counts, two_sided_p, weighted_auroc  # noqa: E402

BUDGETS = ["5", "10", "20", "all"]
PRIMARY = ("harnet5", "feat60_fi")
COMPARISONS = [PRIMARY, ("ssl_indomain", "feat60_fi"), ("harnet5", "harnet5_rand"), ("harnet5", "supervised"),
               ("ssl_indomain", "harnet5"), ("supervised", "feat60_fi")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="results/rq12_seeds_mps")
    ap.add_argument("--comparisons", nargs="+", default=None, metavar="A:B",
                    help="pairs to test as A:B (A minus B); default: the six RQ1/RQ2 comparisons")
    ap.add_argument("--name", default="paired_tests", help="output file results/<out>/<name>.csv")
    a = ap.parse_args()
    comparisons = [tuple(c.split(":")) for c in a.comparisons] if a.comparisons else COMPARISONS
    arms = sorted({x for c in comparisons for x in c})
    patients, counts, point, boot, seeds = {}, {}, {}, {}, []
    for spec in a.runs:
        # "RUN_A+RUN_B" merges runs of the same seed (arms already in an earlier run are taken from it); the window
        # alignment check below then verifies, arm by arm, that the merged runs scored the same windows in the same order
        parts = spec.split("+")
        part_seeds = {json.loads((Path(r) / "config.json").read_text())["seed"] for r in parts}
        if len(part_seeds) != 1:
            raise SystemExit(f"{spec}: merged runs have different seeds {sorted(part_seeds)}")
        seed = part_seeds.pop()
        seeds.append(seed)
        frames, have = [], set()
        for r in parts:
            o = pd.read_parquet(Path(r) / "out_of_fold_scores.parquet", columns=["cohort", "budget", "arm", "subject", "y", "score"])
            frames.append(o[~o.arm.isin(have)])
            have |= set(o.arm.unique())
        oof = pd.concat(frames, ignore_index=True)
        run = spec
        for cohort, dc in oof.groupby("cohort"):
            pts = np.array(sorted(dc.subject.unique()))
            if cohort not in patients:  # one set of resamples per cohort, shared by all seeds, arms and budgets
                patients[cohort], counts[cohort] = pts, patient_counts(len(pts), a.n_boot, seed=0)
            elif not np.array_equal(pts, patients[cohort]):
                raise SystemExit(f"{run}: patients of {cohort} differ from the first run")
            for budget in [b for b in BUDGETS if (dc.budget == b).any()]:  # quick runs have only 5 and all
                db = dc[dc.budget == budget]
                ref = db[db.arm == arms[0]]
                for arm in arms:
                    d = db[db.arm == arm]
                    if not (np.array_equal(d.subject.to_numpy(), ref.subject.to_numpy())
                            and np.array_equal(d.y.to_numpy(), ref.y.to_numpy())):
                        raise SystemExit(f"{run}: windows of {arm} do not line up with {arms[0]} ({cohort}, {budget})")
                    y, s = d.y.to_numpy(), d.score.to_numpy()
                    wp = np.searchsorted(patients[cohort], d.subject.to_numpy())
                    point[seed, cohort, budget, arm] = weighted_auroc(s, y, np.ones(len(y)))
                    boot[seed, cohort, budget, arm] = boot_auroc(s, y, wp, counts[cohort])
        print(f"seed {seed} done", flush=True)

    rows = []
    for cohort in sorted(patients):
        for A, B in comparisons:
            for budget in [b for b in BUDGETS if all((s, cohort, b, A) in point for s in seeds)]:
                d_seed = [point[s, cohort, budget, A] - point[s, cohort, budget, B] for s in seeds]
                d_boot = np.mean([boot[s, cohort, budget, A] - boot[s, cohort, budget, B] for s in seeds], axis=0)
                lo, hi = np.nanpercentile(d_boot, [2.5, 97.5])
                rows.append({"cohort": cohort, "budget": budget, "comparison": f"{A} - {B}",
                             "primary": (A, B) == PRIMARY, "d_mean": np.mean(d_seed), "ci_lo": lo, "ci_hi": hi,
                             "p": two_sided_p(d_boot), "seeds_positive": int(np.sum(np.array(d_seed) > 0)),
                             "n_seeds": len(seeds), "n_patients": len(patients[cohort]),
                             "d_per_seed": ";".join(f"{x:.4f}" for x in d_seed)})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p.to_numpy())
    dest = Path(a.out)
    dest.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest / f"{a.name}.csv", index=False)
    (dest / f"{a.name}_config.json").write_text(json.dumps(
        {"n_boot": a.n_boot, "runs": a.runs, "seeds": sorted(seeds), "comparisons": [list(c) for c in comparisons],
         "n_tests": len(out), "n_patients": {k: len(v) for k, v in patients.items()}}, indent=1))
    pd.set_option("display.width", 200)
    print(f"{len(seeds)} seeds {sorted(seeds)}, {a.n_boot} patient resamples, {len(out)} tests (Holm over all)")
    print(out[["cohort", "comparison", "budget", "d_mean", "ci_lo", "ci_hi", "p", "p_holm", "seeds_positive"]]
          .round(4).to_string(index=False))
    print(f"wrote {dest / (a.name + '.csv')}")


if __name__ == "__main__":
    main()
