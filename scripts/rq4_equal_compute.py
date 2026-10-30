"""RQ4 equal-compute control: does the daily-living gain come from more data or from more optimizer steps?

In RQ4 the ResNet1D pretrained on daily living took about 21 times more optimizer steps (10 epochs over about
550,000 windows) than ssl_indomain (10 epochs over the fold's about 25,000 DeFOG training windows). DeFOG (home) is
evaluated exactly as RQ1/RQ2 and RQ4 (same folds, label budgets, probes and seeds) with three extra arms:

  ssl_dailyliving        as in RQ4: 10 SSL epochs on the fold's daily-living windows (test patients excluded),
                         same per-seed daily-living sample, so it reproduces RQ4's arm (GPU runs are deterministic)
  ssl_indomain_long      the fold's DeFOG training windows for as many optimizer steps as ssl_dailyliving: same
                         initialisation and RNG stream, so its first 10 epochs are ssl_indomain's
  ssl_dailyliving_short  daily-living windows for as many optimizer steps as ssl_indomain: one pass over a random
                         subset of that many batches

Reading: ssl_dailyliving against ssl_indomain_long compares data at equal compute (more and more varied windows
against more passes over the same windows); ssl_dailyliving_short against ssl_indomain compares the data source at
equal, small compute.

Usage:
  python scripts/rq4_equal_compute.py --device mps --seed 0     about 75 min per seed on the M5 GPU
  python scripts/rq4_equal_compute.py --quick --device mps      smoke test
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
_spec = importlib.util.spec_from_file_location("rq12_fog_transfer", ROOT / "scripts" / "rq12_fog_transfer.py")
rq12 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rq12)

from whssl.data import fog_daily as D  # noqa: E402
from whssl.models.encoders import ResNet1D, embed  # noqa: E402
from whssl.ssl.multitask import pretrain  # noqa: E402

BATCH = 256  # pretrain's default batch size; steps per epoch = windows // BATCH


def make_hook(args, log):
    dl = D.load(min_sd=args.dl_min_sd, max_windows=args.dl_max_windows, seed=args.seed)  # RQ4's sample for this seed
    log(f"daily living: {len(dl['X']):,} windows from {len(np.unique(dl['subject']))} patients")

    def hook(cohort, fold, fs, X, Xn, mu, sd, test_subjects, tr, log, **_):
        keep = ~np.isin(dl["subject"], test_subjects)  # never pretrain on this fold's test patients
        Xdl = ((dl["X"][keep] - mu) / sd).astype(np.float32)
        if args.quick:
            Xdl = Xdl[np.sort(np.random.default_rng(fs).choice(len(Xdl), min(len(Xdl), 3000), replace=False))]

        def ssl(X_train, epochs):
            torch.manual_seed(fs)  # same initialisation as ssl_indomain in run_cohort
            enc = ResNet1D()
            pretrain(enc, X_train, epochs=epochs, seed=fs, device=args.device, log=None)
            return embed(enc, Xn, device=args.device)

        t0 = time.time()
        steps_dl = args.dl_epochs * (len(Xdl) // BATCH)
        steps_in = args.ssl_epochs * (len(tr) // BATCH)
        long_epochs = max(args.ssl_epochs, round(steps_dl / max(len(tr) // BATCH, 1)))
        sub = np.sort(np.random.default_rng(fs).choice(len(Xdl), min(len(Xdl), steps_in * BATCH), replace=False))
        reps = {"ssl_dailyliving": ssl(Xdl, args.dl_epochs),
                "ssl_indomain_long": ssl(Xn[tr], long_epochs),
                "ssl_dailyliving_short": ssl(Xdl[sub], 1)}
        log(f"    optimizer steps: dailyliving {steps_dl:,}; indomain_long {long_epochs * (len(tr) // BATCH):,} "
            f"({long_epochs} epochs over {len(tr):,} windows); dailyliving_short {len(sub) // BATCH:,} "
            f"({len(sub):,} windows, one pass); indomain {steps_in:,}; {time.time() - t0:.0f}s")
        return reps

    return hook


def main():
    ap = rq12.base_parser(cohorts=("defog",))
    ap.add_argument("--dl-min-sd", type=float, default=0.01)
    ap.add_argument("--dl-max-windows", type=int, default=10_000)
    ap.add_argument("--dl-epochs", type=int, default=10)
    args = ap.parse_args()
    if args.quick:
        args.dl_epochs = 1
    rq12.run_experiment(args, "rq4_equal_compute", make_hook=make_hook)


if __name__ == "__main__":
    main()
