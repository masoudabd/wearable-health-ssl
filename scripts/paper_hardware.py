"""Post hoc (writing brief 8.5): reproducibility of the RQ1/RQ2 protocol across hardware.

The same protocol and seed ran on three setups before the home labels were corrected: a Windows laptop CPU
(results/rq12_fog_transfer_laptop, commit c03cf41), the Mac CPU (results/rq12_fog_transfer, f31b7fd) and the Mac GPU
(results/rq12_fog_transfer_mps, 0a4f88d). Nothing but the arithmetic differs between them, so the comparison is valid
whatever the label version. For every approach this reports the largest absolute difference in pooled AUROC over both
cohorts and four budgets, for laptop CPU against Mac CPU and Mac CPU against Mac GPU. Recomputes nothing.

Outputs (results/paper_derived/hardware/): summary.csv, summary.json, report.md.
Usage: .venv/bin/python scripts/paper_hardware.py   (seconds)
"""
from __future__ import annotations

import time

import pandas as pd

import paper_common as C

NAME = "hardware"
RUNS = {"laptop_cpu": "results/rq12_fog_transfer_laptop/summary.csv", "mac_cpu": "results/rq12_fog_transfer/summary.csv",
        "mac_gpu": "results/rq12_fog_transfer_mps/summary.csv"}
FROZEN = {"harnet5", "harnet5_rand", "feat60", "feat60_fi"}


def main():
    t0 = time.time()
    d = {k: pd.read_csv(C.ROOT / v, dtype={"budget": str}).set_index(["cohort", "budget", "arm"]).auroc for k, v in RUNS.items()}
    rows = []
    for arm in sorted(set(d["mac_cpu"].index.get_level_values("arm"))):
        r = {"arm": arm, "kind": "frozen" if arm in FROZEN else "trained"}
        for a, b in (("laptop_cpu", "mac_cpu"), ("mac_cpu", "mac_gpu")):
            x = d[a].xs(arm, level="arm")
            y = d[b].xs(arm, level="arm").reindex(x.index)
            r[f"max_abs_diff_{a}_vs_{b}"] = float((x - y).abs().max())
            r[f"cells_{a}_vs_{b}"] = int(len(x))
        rows.append(r)
    out = pd.DataFrame(rows)
    frozen = out[out.kind == "frozen"]
    summary = {"commit": C.git_commit(), "date": time.strftime("%Y-%m-%d"), "runs": RUNS,
               "frozen_max_abs_diff": float(frozen.filter(like="max_abs_diff").max().max()),
               "trained_max_abs_diff_laptop_vs_mac_cpu": float(out[out.kind == "trained"].max_abs_diff_laptop_cpu_vs_mac_cpu.max()),
               "trained_max_abs_diff_mac_cpu_vs_gpu": float(out[out.kind == "trained"].max_abs_diff_mac_cpu_vs_mac_gpu.max()),
               "seconds": round(time.time() - t0, 1)}
    report = f"""# Reproducibility across hardware (post hoc)

Written by `scripts/paper_hardware.py` on {summary['date']} at commit {summary['commit']}.

{C.md_table(out.set_index('arm'))}
"""
    C.write_outputs(NAME, summary, {"summary.csv": out}, report)
    print(report)


if __name__ == "__main__":
    main()
