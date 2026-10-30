"""RQ4, longer adaptation of the foundation model to daily-living recordings.

RQ4 continued harnet5's self-supervised pretraining on the daily-living windows for 2 epochs at lr 1e-4 (harnet5_dl,
scripts/rq4_daily_living.py) and found no change. Here the same continued pretraining runs for --epochs epochs (10 by
default, five times the optimizer steps) and the home-cohort windows are embedded at the checkpoints of the same run
(--checkpoints, default 2, 4, 6, 10), giving one frozen arm per checkpoint (harnet5_dl_e2, ..., harnet5_dl_e10), each
probed as in RQ1/RQ2. Same folds, budgets, seeds, daily-living sample and initialisation as RQ4, and each fold's test
patients are excluded from pretraining; the epoch-2 arm therefore repeats RQ4's harnet5_dl (a check). The from-scratch
daily-living CNN is not retrained. Paired tests merge each seed with its RQ4 run (scripts/paired_tests.py "A+B").

Usage: python scripts/rq4_long_adapt.py --device mps --seed 0   (about 5 h per seed on the M5 GPU)
"""
from __future__ import annotations

import importlib.util
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rq12 = _load("rq12_fog_transfer")
rq4 = _load("rq4_daily_living")  # HarnetEncoder

from whssl.data import fog_daily as D  # noqa: E402
from whssl.models import pretrained as P  # noqa: E402
from whssl.ssl.multitask import pretrain  # noqa: E402


def make_hook(args, log):
    dl = D.load(min_sd=args.dl_min_sd, max_windows=args.dl_max_windows, seed=args.seed)  # RQ4's sample of this seed
    log(f"daily living: {len(dl['X']):,} windows from {len(np.unique(dl['subject']))} patients")

    def hook(cohort, fold, fs, X, Xn, mu, sd, test_subjects, log, **_):
        keep = ~np.isin(dl["subject"], test_subjects)  # never pretrain on this fold's test patients
        Xdl = dl["X"][keep]
        if args.quick:
            Xdl = Xdl[np.random.default_rng(fs).choice(len(Xdl), min(len(Xdl), 3000), replace=False)]
        reps, t0 = {}, time.time()
        torch.manual_seed(fs)
        h = rq4.HarnetEncoder("harnet5")

        def at_epoch_end(msg):  # pretrain logs once per epoch: embed the cohort at the checkpoints
            m = re.match(r"\s*ssl epoch (\d+)/", msg)
            if m and int(m.group(1)) in args.checkpoints:
                h.eval()
                reps[f"harnet5_dl_e{int(m.group(1))}"] = P.harnet_embed(h.fe, X, device=args.device)
                h.train()

        pretrain(h, Xdl, epochs=args.epochs, lr=args.lr, seed=fs, device=args.device, log=at_epoch_end)
        log(f"    {len(Xdl):,} daily-living windows, {args.epochs} epochs, checkpoints {sorted(reps)} "
            f"({time.time() - t0:.0f}s)")
        return reps

    return hook


def main():
    ap = rq12.base_parser(cohorts=("defog",))
    ap.add_argument("--dl-min-sd", type=float, default=0.01)
    ap.add_argument("--dl-max-windows", type=int, default=10_000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--checkpoints", type=int, nargs="+", default=[2, 4, 6, 10])
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()
    if args.quick:
        args.epochs, args.checkpoints = 2, [1, 2]
    rq12.run_experiment(args, "rq4_long_adapt", make_hook=make_hook)


if __name__ == "__main__":
    main()
