"""RQ4: does self-supervised pretraining on unlabelled daily-living data help at home?

Cohort: DeFOG (home protocol, 57 patients), evaluated exactly as RQ1/RQ2 (scripts/rq12_fog_transfer.py: same folds,
label budgets, validation carving, probes and seeds, so every arm is paired with the RQ1/RQ2 arms), plus two arms
pretrained on the FOG daily-living recordings (whssl.data.fog_daily: 65 patients, about 8 days each; active 5 s
windows at 30 Hz, at most 10,000 per patient, 645,164 in all):

  ssl_dailyliving  ResNet1D pretrained from scratch with the multi-task SSL on daily-living windows, standardized
                   with the fold's training statistics (as ssl_indomain, which pretrains on DeFOG training windows)
  harnet5_dl       Oxford harnet5 with continued multi-task SSL on daily-living windows in g (lr 1e-4, the ElderNet
                   recipe), then frozen like harnet5

Leakage: inside each fold, the daily-living data of that fold's DeFOG test patients is excluded from pretraining (45
of the 65 daily-living patients are DeFOG patients). Training patients' daily-living data is used, as their
unlabelled DeFOG windows are for ssl_indomain.

Usage:
  python scripts/rq4_daily_living.py --device mps            seed 0, about 2 h on the M5 GPU
  python scripts/rq4_daily_living.py --quick --device mps    smoke test (minutes)
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
from whssl.models import pretrained as P  # noqa: E402
from whssl.models.encoders import ResNet1D, embed  # noqa: E402
from whssl.ssl.multitask import pretrain  # noqa: E402


class HarnetEncoder(torch.nn.Module):
    """Oxford harnet feature extractor as a trainable encoder: (B, 3, T) in g -> (B, D)."""

    def __init__(self, name: str = "harnet5"):
        super().__init__()
        self.fe = P.load_harnet(name, True)
        for p in self.fe.parameters():
            p.requires_grad_(True)
        self.emb_dim = P.HARNET[name][1]

    def forward(self, x):
        return self.fe(x).flatten(1)


def make_hook(args, log):
    # Each seed draws its own daily-living sample (at most max_windows per patient), as every seed redraws folds,
    # subsampling and initialisation; seed 0 keeps the sample of the first run.
    dl = D.load(min_sd=args.dl_min_sd, max_windows=args.dl_max_windows, seed=args.seed)
    log(f"daily living: {len(dl['X']):,} windows from {len(np.unique(dl['subject']))} patients "
        f"(SD >= {args.dl_min_sd:g} g, at most {args.dl_max_windows:,} per patient)")

    def hook(cohort, fold, fs, X, Xn, mu, sd, test_subjects, log, **_):
        keep = ~np.isin(dl["subject"], test_subjects)  # never pretrain on this fold's test patients
        Xdl = dl["X"][keep]
        if args.quick:
            Xdl = Xdl[np.random.default_rng(fs).choice(len(Xdl), min(len(Xdl), 3000), replace=False)]
        excluded = len(np.intersect1d(np.unique(dl["subject"]), test_subjects))
        t0 = time.time()
        torch.manual_seed(fs)
        enc = ResNet1D()
        pretrain(enc, ((Xdl - mu) / sd).astype(np.float32), epochs=args.dl_epochs, seed=fs, device=args.device,
                 log=None)
        reps = {"ssl_dailyliving": embed(enc, Xn, device=args.device)}
        t1 = time.time()
        torch.manual_seed(fs)
        h = HarnetEncoder("harnet5")
        pretrain(h, Xdl, epochs=args.harnet_dl_epochs, lr=args.harnet_dl_lr, seed=fs, device=args.device, log=None)
        h.fe.eval()
        reps["harnet5_dl"] = P.harnet_embed(h.fe, X, device=args.device)
        log(f"    daily living: {len(Xdl):,} windows from {len(np.unique(dl['subject'][keep]))} patients, {excluded} test "
            f"patients excluded; scratch {t1 - t0:.0f}s, harnet5 continued {time.time() - t1:.0f}s")
        return reps

    return hook


def main():
    ap = rq12.base_parser(cohorts=("defog",))
    ap.add_argument("--dl-min-sd", type=float, default=0.01, help="keep daily-living windows with magnitude SD >= this (g)")
    ap.add_argument("--dl-max-windows", type=int, default=10_000, help="daily-living windows per patient, at most")
    ap.add_argument("--dl-epochs", type=int, default=10, help="SSL epochs, ResNet1D from scratch on daily living")
    ap.add_argument("--harnet-dl-epochs", type=int, default=2, help="continued SSL epochs for harnet5 on daily living")
    ap.add_argument("--harnet-dl-lr", type=float, default=1e-4)
    args = ap.parse_args()
    if args.quick:
        args.dl_epochs, args.harnet_dl_epochs = 1, 1
    rq12.run_experiment(args, "rq4_daily_living", make_hook=make_hook)


if __name__ == "__main__":
    main()
