"""Combines the per-participant free-living rows of the registered external specs into the derived tree.

scripts/external_fogathome.py writes one per_participant.csv per source and axis convention. The paper reads only
the two registered specs, the home-trained and the lab-trained detector, so this puts their rows in one table beside
the pooled external summary, where numbers/sources.yaml points at it as `external_participants`.

  .venv/bin/python scripts/combine_external_participants.py [--suffix _b20k]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPECS = [("defog", "Vflip"), ("tdcsfog", "VAPflip")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="", help="run-directory suffix, e.g. _b20k")
    a = ap.parse_args()
    frames = []
    for source, axes in SPECS:
        p = ROOT / f"results/external_fogathome_{source}_{axes}{a.suffix}" / "per_participant.csv"
        if not p.exists():
            raise SystemExit(f"missing {p.relative_to(ROOT)}: rerun external_fogathome.py for this spec")
        frames.append(pd.read_csv(p))
    out = ROOT / f"results/paper_derived{a.suffix}" / "external" / "per_participant.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(out, index=False)
    print(f"wrote {out.relative_to(ROOT)}: {len(combined)} rows from {len(frames)} specs")


if __name__ == "__main__":
    main()
