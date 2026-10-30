"""Cross-cohort transfer (paper plan v0 section 11 item 2): train in one cohort, test in the other.

Lab to home and home to lab, for the frozen representations of RQ1/RQ2 (no training of networks, so CPU minutes):
feat60, feat60_fi, harnet5, harnet5_rand, each with the default probe and with the tuned probe (C chosen on validation
AUPRC, as --extra tuned_probe). Per repetition (seed) and label budget (5, 10, 20 or all patients of the source
cohort), labeled patients are drawn from the source cohort, validation patients are carved from them and choose the
threshold (and the tuned C), and every patient of the target cohort is scored once. Nothing from the target cohort is
used for fitting or selection.

Arms are written with the suffix _x (harnet5_x, feat60_fi_tc_x, ...) and the target windows in the order of the
RQ1/RQ2 run of the same seed (folds concatenated, as subject_kfold yields them), so each seed's run can be merged with
that seed's within-cohort run for paired tests ("RUN_A+RUN_B" in scripts/paired_tests.py): transfer gap (harnet5 minus
harnet5_x: the same number of labeled patients from the target cohort against from the other cohort; with all, the
within-cohort model trains on about four fifths of the target patients, the transfer model on every source patient)
and the foundation model against the features under transfer (harnet5_x minus feat60_fi_x).

Axis conventions. The two cohorts record the anteroposterior (AP) axis with opposite signs (mean AP +0.20 g in the
lab cohort, positive in 66 of 71 patients; -0.19 g at home, negative in 49 of 57; vertical and mediolateral agree), which
the dataset does not document. --flip harmonizes the target cohort to the source's convention before any
representation is computed: ap negates AP; apml negates AP and mediolateral (a 180 degree rotation about the vertical
axis, the physical alternative that gravity cannot distinguish). Arms are suffixed _x (no flip), _xap or _xrot.
Flipped modes cover the features and harnet5 (the random-weight control needs new embeddings per seed and is left out);
harnet5 embeddings of the flipped windows are cached in data/processed/fog_contest/.

Usage: python scripts/cross_cohort.py --seed 0 [--flip none|ap|apml] [--out ...]   (minutes on the CPU; a flipped
mode first embeds the flipped windows, about 12 minutes on the CPU or under a minute on the GPU)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rq12 = _load("rq12_fog_transfer")

from whssl.data import fog_contest as F  # noqa: E402
from whssl.eval.metrics import best_f1_threshold, binary_report  # noqa: E402
from whssl.eval.probe import fit_probe, probe_scores  # noqa: E402
from whssl.eval.splits import carve_validation, subject_kfold, subsample_subjects  # noqa: E402
from whssl.features import feat60, freeze_index_features  # noqa: E402
from whssl.models import pretrained as P  # noqa: E402

COHORTS = ("tdcsfog", "defog")
TC_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
FLIPS = {"none": (np.array([1, 1, 1], np.float32), "_x"), "ap": (np.array([1, 1, -1], np.float32), "_xap"),
         "apml": (np.array([1, -1, -1], np.float32), "_xrot")}  # axes V, ML, AP


def flipped_reps(W, cohort, mode, device, log):
    """Representations of one cohort's windows (cache order) after the axis flip of `mode`."""
    sign, _ = FLIPS[mode]
    m = W["cohort"] == cohort
    X = W["X"][m] * sign
    proc = F.PROC
    f60 = rq12.cached(proc / f"feat60_flip{mode}_{cohort}.npz", lambda: feat60(X, rq12.FS)[0])
    ffi = rq12.cached(proc / f"freeze_index_flip{mode}_{cohort}.npz", lambda: freeze_index_features(X, rq12.FS)[0])
    t = time.time()
    h = rq12.cached(proc / f"emb_harnet5_flip{mode}_{cohort}.npz",
                    lambda: P.harnet_embed(P.load_harnet("harnet5", True), X, device=device))
    log(f"  {cohort} flipped ({mode}) representations ready ({time.time() - t:.0f}s)")
    return {"feat60": f60, "feat60_fi": np.concatenate([f60, ffi], 1), "harnet5": h}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budgets", type=int, nargs="+", default=[5, 10, 20, rq12.ALL])
    ap.add_argument("--folds", type=int, default=5, help="fold structure of the RQ1/RQ2 run (window order only)")
    ap.add_argument("--flip", choices=list(FLIPS), default="none", help="axis harmonization of the target cohort")
    ap.add_argument("--device", default="cpu", help="for the flipped harnet5 embeddings only")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    suffix = FLIPS[args.flip][1]
    out = ROOT / "results" / (args.out or f"cross_cohort_{args.flip}_seed{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "src", "scripts"], cwd=ROOT, capture_output=True,
                                text=True).stdout.strip())
    log(f"=== {time.strftime('%Y-%m-%d %H:%M')} commit {commit}{' (uncommitted changes)' if dirty else ''} args {vars(args)}")
    W = F.load_windows(rq12.FS, rq12.WIN, rq12.HOP)
    reps = rq12.frozen_representations(W["X"], log, "cpu", args.seed)
    y_all, g_all = W["y"].astype(int), W["subject"]
    rows, oof = [], []
    for src, tgt in [("tdcsfog", "defog"), ("defog", "tdcsfog")]:
        ms, mt = np.flatnonzero(W["cohort"] == src), np.flatnonzero(W["cohort"] == tgt)
        ys, gs, yt, gt = y_all[ms], g_all[ms], y_all[mt], g_all[mt]
        tgt_reps = None if args.flip == "none" else flipped_reps(W, tgt, args.flip, args.device, log)
        # target windows in the order of the RQ1/RQ2 run of this seed (its folds concatenated)
        order = np.concatenate([te for _, te in subject_kfold(yt, gt, args.folds, args.seed)])
        tw = mt[order]
        log(f"[{src} -> {tgt}] source {len(np.unique(gs))} patients, target {len(np.unique(gt))} patients, "
            f"{len(tw)} target windows")
        for n in args.budgets:
            fs = args.seed * 1000 + 500 + (0 if src == "tdcsfog" else 1)  # own stream, distinct from the RQ1/RQ2 folds
            sub = subsample_subjects(ys, gs, np.arange(len(ms)), n, seed=fs)
            fit, val = carve_validation(ys, gs, sub, frac=0.2, seed=fs)
            budget = "all" if n >= rq12.ALL else str(n)
            n_eff = len(np.unique(gs[sub]))
            line = []
            arms = ["feat60", "feat60_fi", "harnet5"] + (["harnet5_rand"] if args.flip == "none" else [])
            for arm in arms:
                Zs = reps[arm][ms]
                Zt = reps[arm][tw] if tgt_reps is None else tgt_reps[arm][order]  # flipped reps are in cohort order
                clf = fit_probe(Zs[fit], ys[fit], seed=args.seed)
                sv = probe_scores(clf, Zs[val])
                variants = {arm + suffix: (clf, sv, 1.0)}
                best = None
                for C in TC_GRID:  # tuned probe, C chosen on the source validation patients
                    c = fit_probe(Zs[fit], ys[fit], C=C, seed=args.seed)
                    v = probe_scores(c, Zs[val])
                    ap_ = average_precision_score(ys[val], v)
                    if best is None or ap_ > best[0]:
                        best = (ap_, c, v, C)
                variants[arm + "_tc" + suffix] = (best[1], best[2], best[3])
                for name, (c, v, C) in variants.items():
                    s = probe_scores(c, Zt)
                    thr = best_f1_threshold(ys[val], v)
                    rep = binary_report(yt[order], s, thr)
                    rows.append({"source": src, "cohort": tgt, "budget": budget, "n_source_patients": n_eff,
                                 "arm": name, "C": C, **rep, "threshold": thr})
                    oof.append(pd.DataFrame({"cohort": tgt, "budget": budget, "arm": name, "subject": gt[order],
                                             "y": yt[order], "score": s, "pred": (s >= thr).astype(int),
                                             "window": tw}))
                    line.append(f"{name}={roc_auc_score(yt[order], s):.3f}")
            log(f"  budget {budget:>3} (n={n_eff}): " + " ".join(line))
    pd.DataFrame(rows).to_csv(out / "per_budget.csv", index=False)
    pd.concat(oof, ignore_index=True).to_parquet(out / "out_of_fold_scores.parquet")
    (out / "config.json").write_text(json.dumps({**vars(args), "commit": commit, "dirty": dirty, "fs": rq12.FS,
                                                 "win_s": rq12.WIN, "hop_s": rq12.HOP, "device": "cpu"}, indent=2))
    log("DONE")


if __name__ == "__main__":
    main()
