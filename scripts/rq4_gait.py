"""RQ4, gait-focused variant: does pretraining on walking-like daily-living windows beat pretraining on all active ones?

ElderNet adapted the Oxford model on gait bouts; RQ4 used all active daily-living windows (magnitude SD >= 0.01 g).
Here the same daily-living recordings are restricted to walking-like windows (whssl.data.fog_daily, gait=True:
magnitude SD >= 0.02 g and rhythmicity >= 0.33, validated on HAR70+ by scripts/validate_gait_filter.py; at most
10,000 per patient, own sample per seed; about 541,000 windows for seed 0 against 645,000 active ones). DeFOG (home)
is evaluated exactly as RQ1/RQ2 and RQ4 (same folds, budgets, probes, seeds), with two arms, each given as many
optimizer steps as its RQ4 counterpart (scripts/rq4_daily_living.py) in the same fold:

  ssl_dailygait   ResNet1D from scratch with the multi-task SSL on the fold's walking-like windows (test patients
                  excluded); steps = those of ssl_dailyliving (10 epochs over the fold's active windows)
  harnet5_dlgait  harnet5 with continued SSL on the walking-like windows in g (lr 1e-4); steps = those of harnet5_dl
                  (2 epochs over the fold's active windows)

Paired against the RQ4 arms by merging each seed's run with the RQ4 run of the same seed (scripts/paired_tests.py
"RUN_A+RUN_B"; GPU runs are deterministic, so the shared arms are identical).

Usage:
  python scripts/rq4_gait.py --device mps --seed 0     about 90 min per seed on the M5 GPU
  python scripts/rq4_gait.py --quick --device mps      smoke test
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


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rq12 = _load("rq12_fog_transfer")
rq4 = _load("rq4_daily_living")  # HarnetEncoder

from whssl.data import fog_daily as D  # noqa: E402
from whssl.models import pretrained as P  # noqa: E402
from whssl.models.encoders import ResNet1D, embed  # noqa: E402
from whssl.ssl.multitask import pretrain  # noqa: E402

BATCH = 256  # pretrain's default batch size; steps per epoch = windows // BATCH


def make_hook(args, log):
    active = D.load(min_sd=args.dl_min_sd, max_windows=args.dl_max_windows, seed=args.seed)  # RQ4's sample: step budgets
    gait = D.load(min_sd=args.dl_min_sd, max_windows=args.dl_max_windows, seed=args.seed, gait=True)
    log(f"daily living: {len(gait['X']):,} walking-like windows (against {len(active['X']):,} active) from "
        f"{len(np.unique(gait['subject']))} patients")

    def hook(cohort, fold, fs, X, Xn, mu, sd, test_subjects, log, **_):
        n_active = int((~np.isin(active["subject"], test_subjects)).sum())  # the fold's RQ4 pretraining set size
        keep = ~np.isin(gait["subject"], test_subjects)  # never pretrain on this fold's test patients
        Xg = gait["X"][keep]
        if args.quick:
            Xg = Xg[np.sort(np.random.default_rng(fs).choice(len(Xg), min(len(Xg), 3000), replace=False))]
            n_active = min(n_active, 3000)
        per_epoch = max(len(Xg) // BATCH, 1)
        ssl_epochs = max(1, round(args.dl_epochs * (n_active // BATCH) / per_epoch))
        h_epochs = max(1, round(args.harnet_dl_epochs * (n_active // BATCH) / per_epoch))
        t0 = time.time()
        torch.manual_seed(fs)
        enc = ResNet1D()
        pretrain(enc, ((Xg - mu) / sd).astype(np.float32), epochs=ssl_epochs, seed=fs, device=args.device, log=None)
        reps = {"ssl_dailygait": embed(enc, Xn, device=args.device)}
        t1 = time.time()
        torch.manual_seed(fs)
        h = rq4.HarnetEncoder("harnet5")
        pretrain(h, Xg, epochs=h_epochs, lr=args.harnet_dl_lr, seed=fs, device=args.device, log=None)
        h.fe.eval()
        reps["harnet5_dlgait"] = P.harnet_embed(h.fe, X, device=args.device)
        log(f"    walking-like windows {len(Xg):,} (fold's active windows {n_active:,}); ssl_dailygait {ssl_epochs} epochs "
            f"= {ssl_epochs * per_epoch:,} steps (RQ4 {args.dl_epochs * (n_active // BATCH):,}); harnet5_dlgait "
            f"{h_epochs} epochs = {h_epochs * per_epoch:,} steps (RQ4 {args.harnet_dl_epochs * (n_active // BATCH):,}); "
            f"{t1 - t0:.0f}s + {time.time() - t1:.0f}s")
        return reps

    return hook


def main():
    ap = rq12.base_parser(cohorts=("defog",))
    ap.add_argument("--dl-min-sd", type=float, default=0.01)
    ap.add_argument("--dl-max-windows", type=int, default=10_000)
    ap.add_argument("--dl-epochs", type=int, default=10, help="RQ4's epochs for ssl_dailyliving (sets the step budget)")
    ap.add_argument("--harnet-dl-epochs", type=int, default=2, help="RQ4's epochs for harnet5_dl (sets the step budget)")
    ap.add_argument("--harnet-dl-lr", type=float, default=1e-4)
    args = ap.parse_args()
    if args.quick:
        args.dl_epochs, args.harnet_dl_epochs = 1, 1
    rq12.run_experiment(args, "rq4_gait", make_hook=make_hook)


if __name__ == "__main__":
    main()
