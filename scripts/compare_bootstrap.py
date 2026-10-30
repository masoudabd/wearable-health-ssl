"""Compares a re-bootstrapped tree (results/*<suffix>/) with the registered one, table by table.

Only the resample count differs between the two trees, so every point estimate must come out identical and only
interval bounds, standard deviations and bootstrap P-values may move. The report therefore separates three things:
any point estimate that moved (a bug, not a result), how far the intervals and P-values moved, and whether any
Holm decision at 0.05 changed, which is the only thing that can change a claim in the paper.

    .venv/bin/python scripts/compare_bootstrap.py --suffix _b20k
    .venv/bin/python scripts/compare_bootstrap.py --suffix _b20k --alpha 0.05 --out results/paper_derived/rebootstrap

Since 2026-10-30 the analyses that are not part of the paper (Daphnet, WearGait-PD, PADS) are left out unless
--include-held is given, so the counts quoted in the Methods cover only the paper's own analyses; the pattern must
stay identical to the one the public release uses to leave those analyses out. A rebuilt table that gained columns after the audit was
registered (the external summaries gained elapsed-time columns on 2026-10-29) is compared on the registered columns.
With --out the script also writes crossing_counts.json, the counts the Methods quote.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
# columns that a different resample count is allowed to change; everything else numeric must be bit-identical
RESAMPLED = ("_lo", "_hi", "_sd")
P_COLS = ("p", "p_holm", "p_adj", "pval", "p_value")
# analyses not part of the paper (2026-10-30); identical to the public release's pattern
HELD = re.compile(r"(?i)(weargait|daphnet|synapse_setup|(^|[/_])pads)")


def is_p(col: str) -> bool:
    """True for a P-value column: a plain one, a per-contrast "<something>_p", or a Holm column named for its
    family ("p_holm_all", "p_holm_main12", "p_holm_4")."""
    c = col.lower()
    return c in P_COLS or c.endswith("_p") or c.startswith(("p_holm", "p_adj"))


def resampled(col: str) -> bool:
    """True for a column a resample count may change: an interval bound ("lo", "ci_lo", "auroc_lo"), a bootstrap
    standard deviation, or any P-value."""
    c = col.lower()
    return is_p(c) or c in ("lo", "hi") or any(k in c for k in RESAMPLED)


def pairs(suffix: str, include_held: bool = False) -> list[tuple[Path, Path]]:
    """Every CSV under results/*<suffix>/ next to its registered counterpart, where that counterpart exists (held
    analyses left out unless asked for)."""
    out = []
    for new in sorted(RESULTS.glob(f"*{suffix}/**/*.csv")):
        if not include_held and HELD.search(str(new.relative_to(RESULTS))):
            continue
        rel = new.relative_to(RESULTS).parts
        old = RESULTS.joinpath(rel[0].removesuffix(suffix), *rel[1:])
        if old.exists():
            out.append((old, new))
    return out


def key_of(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if df[c].dtype == object or not np.issubdtype(df[c].dtype, np.number)]


def auroc_width_changes(a: pd.DataFrame, b: pd.DataFrame) -> list[float]:
    """How much AUROC interval widths moved, for the fragility statement in the Methods. Only AUROC intervals count,
    so a generic ci_lo/ci_hi pair is included only where the table's metric column says the row is an AUROC (the
    agreement tables name theirs `statistic`, and none of its values is an AUROC: those are ICC, bias, MAE and rho)."""
    out = []
    for lo in [c for c in a.columns if c.endswith("_lo") and "auroc" in c.lower()]:
        if (hi := lo[:-3] + "_hi") in a.columns:
            out.append(((b[hi] - b[lo]) - (a[hi] - a[lo])).abs().max())
    if {"ci_lo", "ci_hi"} <= set(a.columns):
        mcol = next((c for c in ("metric", "measure", "statistic") if c in a.columns), None)
        sel = a[mcol].astype(str).str.lower().str.contains("auroc") if mcol else pd.Series(True, index=a.index)
        if sel.any():
            out.append(((b.ci_hi[sel] - b.ci_lo[sel]) - (a.ci_hi[sel] - a.ci_lo[sel])).abs().max())
    return [float(x) for x in out if pd.notna(x)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_b20k")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default=None, help="directory for the three CSV reports (default: print only)")
    ap.add_argument("--include-held", action="store_true", help="also compare the analyses that are not part of the paper")
    args = ap.parse_args()

    moved, flips, summary = [], [], []
    p_cells = 0
    for old, new in pairs(args.suffix, args.include_held):
        name = str(new.relative_to(RESULTS))
        a, b = pd.read_csv(old), pd.read_csv(new)
        added = [c for c in b.columns if c not in a.columns]
        if len(a) == len(b) and added and set(a.columns) <= set(b.columns):
            b = b[list(a.columns)]  # columns added to the rebuilt table later: compare on the registered ones
        if a.shape != b.shape or list(a.columns) != list(b.columns):
            summary.append({"table": name, "status": f"shape/columns differ {a.shape} vs {b.shape}"})
            continue
        key = key_of(a)
        if not a[key].equals(b[key]):
            summary.append({"table": name, "status": "row labels differ"})
            continue
        num = [c for c in a.columns if c not in key and np.issubdtype(a[c].dtype, np.number)]
        fixed = [c for c in num if not resampled(c)]
        changed_fixed = [c for c in fixed if not np.allclose(a[c], b[c], rtol=0, atol=0, equal_nan=True)]
        for c in changed_fixed:  # a point estimate that moved means the two trees are not the same computation
            d = (b[c] - a[c]).abs()
            i = int(d.idxmax())
            moved.append({"table": name, "column": c, "n_rows": int((d > 0).sum()), "max_abs_diff": float(d.max()),
                          "row": " | ".join(str(a.loc[i, k]) for k in key)})
        # interval and P movement, and Holm decisions crossing alpha
        widths = [((b[hi] - b[lo]) - (a[hi] - a[lo])).abs().max()
                  for lo, hi in [(c, c[:-3] + "_hi") for c in num if c.endswith("_lo")] + [("lo", "hi")]
                  if lo in num and hi in num]
        pcols = [c for c in num if is_p(c)]
        p_cells += sum(int(a[c].notna().sum()) for c in pcols)
        dp = max((float((b[c] - a[c]).abs().max()) for c in pcols), default=np.nan)
        for c in pcols:
            cross = (a[c] < args.alpha) != (b[c] < args.alpha)
            for i in np.flatnonzero(cross.to_numpy()):
                flips.append({"table": name, "column": c, "row": " | ".join(str(a.loc[i, k]) for k in key),
                              "p_registered": float(a.loc[i, c]), "p_rebuilt": float(b.loc[i, c])})
        floor_old = sum(int((np.isclose(a[c], 1 / 2000)).sum()) for c in pcols if c.lower() == "p")
        floor_new = sum(int((np.isclose(b[c], 1 / 20000)).sum()) for c in pcols if c.lower() == "p")
        auroc_w = auroc_width_changes(a, b)
        summary.append({"table": name, "status": ("point estimates identical" + (f" (on the {a.shape[1]} registered columns)" if added else ""))
                        if not changed_fixed else "MOVED",
                        "n_rows": len(a), "max_ci_width_change": float(max(widths)) if widths else np.nan,
                        "max_auroc_ci_width_change": max(auroc_w) if auroc_w else np.nan,
                        "n_auroc_interval_sets": len(auroc_w),
                        "max_p_change": dp, "at_old_floor": floor_old, "at_new_floor": floor_new})

    S, M, F = pd.DataFrame(summary), pd.DataFrame(moved), pd.DataFrame(flips)
    ok = int(S.status.str.startswith("point estimates identical").sum())
    print(f"{len(S)} tables compared, {ok} with identical point estimates, {len(S) - ok} not")
    for r in S[~S.status.str.startswith("point estimates identical")].itertuples(index=False):
        print(f"  {r.status}: {r.table}")
    if not M.empty:
        print("\npoint estimates that moved (must be none):")
        print(M.to_string(index=False))
    print(f"\nlargest interval width change: {S.max_ci_width_change.max():.4f}")
    print(f"largest AUROC interval width change: {S.max_auroc_ci_width_change.max():.4f} "
          f"over {int(S.n_auroc_interval_sets.sum())} AUROC interval sets (the figure quoted in the Methods)")
    print(f"largest P change: {S.max_p_change.max():.4f}")
    print(f"rows at the 2,000 P floor (registered): {int(S.at_old_floor.sum())}; "
          f"at the 20,000 floor (rebuilt): {int(S.at_new_floor.sum())}")
    print(f"\nHolm/P decisions crossing alpha={args.alpha}: {len(F)}")
    if not F.empty:
        print(F.to_string(index=False))

    if args.out:
        d = ROOT / args.out
        d.mkdir(parents=True, exist_ok=True)
        S.to_csv(d / "tables.csv", index=False)
        M.to_csv(d / "point_estimates_moved.csv", index=False)
        F.to_csv(d / "decision_changes.csv", index=False)
        gained = int(((F.p_rebuilt < args.alpha) & (F.p_registered >= args.alpha)).sum()) if len(F) else 0
        counts = {"crossings": len(F), "gained": gained, "lost": len(F) - gained, "p_cells": p_cells,
                  "tables_compared": ok, "auroc_interval_sets": int(S.n_auroc_interval_sets.sum()),
                  "max_auroc_ci_width_change": float(S.max_auroc_ci_width_change.max()),
                  "held_included": bool(args.include_held)}
        (d / "crossing_counts.json").write_text(json.dumps(counts, indent=1) + "\n")
        print(f"\nwritten to {d}")


if __name__ == "__main__":
    main()
