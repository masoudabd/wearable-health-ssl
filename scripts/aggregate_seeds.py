"""Aggregate RQ1/RQ2 runs that differ only in --seed (repeated cross-validation: each seed redraws the patient folds,
the label-budget subsampling, training initialisation and the random-weights control).

Usage: python scripts/aggregate_seeds.py RUN [RUN ...] [--out results/rq12_seeds] [--ref feat60_fi]
Writes <out>/summary_seeds.csv, per cohort x budget x arm: pooled AUROC mean, SD, min and max over seeds; the mean
of the per-seed patient-bootstrap bounds; AUPRC, F1 and percent-time-frozen agreement averaged over seeds; and the
paired difference from the reference arm within each seed (mean, SD, seeds where the arm is higher).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

BUDGET_ORDER = {"5": 0, "10": 1, "20": 2, "all": 3}
IGNORE = {"seed", "commit", "dirty", "out", "threads"}  # may differ between runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="results/rq12_seeds")
    ap.add_argument("--ref", default="feat60_fi", help="arm the paired differences are taken against")
    a = ap.parse_args()

    configs = {r: json.loads((Path(r) / "config.json").read_text()) for r in a.runs}
    base = next(iter(configs.values()))
    for r, c in configs.items():
        diff = {k: (base.get(k), c.get(k)) for k in set(base) | set(c) if k not in IGNORE and base.get(k) != c.get(k)}
        if diff:
            raise SystemExit(f"{r} differs from {a.runs[0]} in more than the seed: {diff}")
    seeds = [c["seed"] for c in configs.values()]
    if len(set(seeds)) != len(seeds):
        raise SystemExit(f"duplicate seeds: {seeds}")

    s = pd.concat([pd.read_csv(Path(r) / "summary.csv", dtype={"budget": str}).assign(seed=configs[r]["seed"])
                   for r in a.runs], ignore_index=True)
    ref = s[s.arm == a.ref][["cohort", "budget", "seed", "auroc"]].rename(columns={"auroc": "auroc_ref"})
    s = s.merge(ref, on=["cohort", "budget", "seed"])
    s["d_ref"] = s.auroc - s.auroc_ref
    g = s.groupby(["cohort", "budget", "arm"])
    out = pd.DataFrame({
        "n_seeds": g.seed.nunique(),
        "auroc_mean": g.auroc.mean(), "auroc_sd": g.auroc.std(), "auroc_min": g.auroc.min(), "auroc_max": g.auroc.max(),
        "auroc_lo_mean": g.auroc_lo.mean(), "auroc_hi_mean": g.auroc_hi.mean(),
        f"d_vs_{a.ref}_mean": g.d_ref.mean(), f"d_vs_{a.ref}_sd": g.d_ref.std(),
        f"seeds_above_{a.ref}": g.d_ref.apply(lambda d: int((d > 0).sum())),
        "auprc_mean": g.auprc_mean.mean(), "f1_pos_mean": g.f1_pos_mean.mean(),
        "ptf_spearman_mean": g.ptf_spearman_mean.mean(), "ptf_mae_pp_mean": g.ptf_mae_pp_mean.mean(),
    }).reset_index()
    out = out.sort_values(["cohort", "budget", "auroc_mean"], ascending=[True, True, False],
                          key=lambda c: c.map(BUDGET_ORDER) if c.name == "budget" else c)
    dest = Path(a.out)
    dest.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest / "summary_seeds.csv", index=False)
    (dest / "runs.json").write_text(json.dumps({r: {k: configs[r].get(k) for k in ("seed", "commit", "dirty", "device")}
                                                for r in a.runs}, indent=1))
    pd.set_option("display.width", 220)
    show = out[["cohort", "budget", "arm", "auroc_mean", "auroc_sd", "auroc_min", "auroc_max", "auroc_lo_mean",
                "auroc_hi_mean", f"d_vs_{a.ref}_mean", f"d_vs_{a.ref}_sd", f"seeds_above_{a.ref}"]]
    print(f"{len(a.runs)} seeds {sorted(seeds)}; device {sorted({c.get('device', 'cpu') for c in configs.values()})}")
    print(show.round(3).to_string(index=False))
    print(f"wrote {dest / 'summary_seeds.csv'}")


if __name__ == "__main__":
    main()
