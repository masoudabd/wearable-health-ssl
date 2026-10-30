"""RQ1 + RQ2 on the FOG contest data (128 patients, two cohorts).

RQ1: do frozen public encoders beat handcrafted features and supervised training on FOG detection?
RQ2: how many labelled patients does each approach need?

Arms (frozen representation + logistic-regression probe unless noted):
  feat60        60 window features (20 per axis; whssl.features)
  feat60_fi     feat60 + locomotor power, freeze power, freeze index per axis (Moore 2008)
  harnet5       Oxford SSL encoder (UK Biobank wrist, 30 Hz, 5 s), pretrained weights, frozen
  harnet5_rand  same architecture with random weights (separates pretraining from architecture)
  ssl_indomain  ResNet1D pretrained with multi-task SSL on the fold's training patients (unlabelled)
  supervised    ResNet1D trained end to end on the labelled budget

Optional arms (--extra; plan v0 section 11 item 1: equal effort for the baselines), computed after the arms above in
each budget so those stay bit-identical:
  tuned_probe    <arm>_tc for every frozen arm: the probe's C chosen from --tc-grid by validation AUPRC (fitted on the
                 fit patients, as the default probe; the chosen C is in per_fold.csv)
  supervised_es  ResNet1D from scratch as supervised, but up to --es-max-epochs with early stopping (--patience) on
                 validation AUPRC instead of a fixed 6 epochs
  harnet5_ft     harnet5 fine-tuned end to end, after Yuan et al. 2024: all layers, their evaluation head (two
                 linear layers), Adam (amsgrad) at 1e-4, class-weighted loss, up to --ft-max-epochs with early
                 stopping (--patience) on validation AUPRC; raw g input as the frozen arm
  harnet5_lpft   linear probing, then fine-tuning (LP-FT): the head is the tuned harnet5 probe (C chosen on validation
                 AUPRC, as harnet5_tc) folded into one linear layer, then every layer is trained at --lpft-lr (10 times
                 below harnet5_ft) with Adam (amsgrad), up to --ft-max-epochs, early stopping (--patience) on validation
                 AUPRC; the untrained warm start counts as epoch 0, so it is kept when training never beats it

Every run also stores the global window index of each out-of-fold row and the validation patients' scores
(validation_scores.parquet), so operating points can be chosen on validation patients afterwards.

Protocol, per cohort (tdcsfog, defog), never pooled:
  * 5-fold cross-validation over patients, stratified on window labels.
  * Label budget: n in {5, 10, 20, all} training patients (redrawn until both classes present).
  * Validation patients carved from the budget choose the threshold and the supervised checkpoint.
  * Test patients are scored once. Intervals resample patients.
  * Self-supervised pretraining sees all training patients of the fold (no labels), never test patients.

Input: 5 s windows at 30 Hz in g (axes V, ML, AP), hop 2.5 s, label = FOG in at least half the window.
harnet5 gets raw g (as in its pretraining); in-domain models get channel-standardized input
(statistics from training patients only).

Usage:
  python scripts/rq12_fog_transfer.py --quick     smoke test (minutes)
  python scripts/rq12_fog_transfer.py             full run (hours on CPU)
  python scripts/rq12_fog_transfer.py --device auto --out rq12_fog_transfer_gpu
      on the GPU (cuda, else mps) into its own folder. CPU stays the default, so CPU results reproduce exactly;
      GPU arithmetic differs slightly, which moves the trained arms (as a different CPU does).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from whssl.data import fog_contest as F  # noqa: E402
from whssl.eval.metrics import best_f1_threshold, binary_report, subject_bootstrap_ci  # noqa: E402
from whssl.eval.probe import fit_probe, probe_scores  # noqa: E402
from whssl.eval.splits import carve_validation, subject_kfold, subsample_subjects  # noqa: E402
from whssl.features import feat60, freeze_index_features  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from whssl.models import pretrained as P  # noqa: E402
from whssl.models.encoders import ResNet1D, embed  # noqa: E402
from whssl.models.supervised import predict, train_supervised  # noqa: E402
from whssl.ssl.multitask import pretrain  # noqa: E402

FS, WIN, HOP = 30.0, 5.0, 2.5
ALL = 10_000  # budget meaning "all training patients"


def cached(path: Path, fn):
    if path.exists():
        return np.load(path)["Z"]
    Z = fn()
    np.savez(path, Z=Z)
    return Z


def pick_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return name


def frozen_representations(X: np.ndarray, log, device: str = "cpu", seed: int = 0) -> dict:
    """Fold-independent representations, computed once and cached (the cache does not depend on the device).
    The random-weights control gets fresh weights for every seed; seed 0 keeps its original cache file."""
    proc = F.PROC
    t = time.time()
    f60 = cached(proc / "feat60.npz", lambda: feat60(X, FS)[0])
    ffi = cached(proc / "freeze_index.npz", lambda: freeze_index_features(X, FS)[0])
    log(f"features ready ({time.time() - t:.0f}s)")
    t = time.time()
    h = cached(proc / "emb_harnet5.npz", lambda: P.harnet_embed(P.load_harnet("harnet5", True), X, device=device))
    torch.manual_seed(seed)
    hr = cached(proc / ("emb_harnet5_rand.npz" if seed == 0 else f"emb_harnet5_rand_seed{seed}.npz"),
                lambda: P.harnet_embed(P.load_harnet("harnet5", False), X, device=device))
    log(f"harnet5 embeddings ready ({time.time() - t:.0f}s)")
    return {"feat60": f60, "feat60_fi": np.concatenate([f60, ffi], 1), "harnet5": h, "harnet5_rand": hr}


def pct_time_frozen(y, pred, groups):
    """Per-patient fraction of FOG windows: truth vs prediction (a clinical-unit summary)."""
    df = pd.DataFrame({"g": groups, "y": y, "p": pred}).groupby("g").mean()
    rho = spearmanr(df.y, df.p).statistic if df.y.nunique() > 1 else np.nan
    return float(rho), float((df.y - df.p).abs().mean() * 100)


def run_cohort(cohort, W, reps, args, log, fold_hook=None):
    """fold_hook (optional, used by RQ4) returns extra arms {name: embeddings of all cohort windows} for a fold.
    It gets the fold's test patients so that anything it pretrains can exclude them."""
    m = W["cohort"] == cohort
    X, y, g, wi = W["X"][m], W["y"][m].astype(int), W["subject"][m], np.flatnonzero(m)
    reps = {k: v[m] for k, v in reps.items()}
    budgets = args.budgets
    if args.quick:
        keep = np.isin(g, np.unique(g)[:24])
        X, y, g, wi = X[keep], y[keep], g[keep], wi[keep]
        reps = {k: v[keep] for k, v in reps.items()}
    log(f"[{cohort}] windows={len(y)} patients={len(np.unique(g))} FOG prevalence={y.mean():.3f}")
    extra = set(getattr(args, "extra", None) or [])
    rows, oof, vals = [], [], []
    for fold, (tr, te) in enumerate(subject_kfold(y, g, args.folds, args.seed)):
        t0 = time.time()
        # Per-fold seed. seed * 1000 + fold equals the original seed + fold for seed 0 (so seed-0 results are
        # unchanged) and, unlike seed + fold, never gives two (seed, fold) pairs the same initialisation.
        fs = args.seed * 1000 + fold
        mu, sd = X[tr].mean(axis=(0, 1)), X[tr].std(axis=(0, 1)) + 1e-8
        Xn = ((X - mu) / sd).astype(np.float32)
        torch.manual_seed(fs)
        enc = ResNet1D()
        pretrain(enc, Xn[tr], epochs=args.ssl_epochs, seed=fs, device=args.device, log=None)
        fold_reps = dict(reps, ssl_indomain=embed(enc, Xn, device=args.device))
        if fold_hook is not None:
            fold_reps.update(fold_hook(cohort=cohort, fold=fold, fs=fs, X=X, Xn=Xn, mu=mu, sd=sd,
                                       test_subjects=np.unique(g[te]), tr=tr, log=log))
        log(f"  fold {fold}: train={len(np.unique(g[tr]))} test={len(np.unique(g[te]))} patients; "
            f"SSL done ({time.time() - t0:.0f}s)")
        for n in budgets:
            sub = subsample_subjects(y, g, tr, n, seed=fs)
            fit, val = carve_validation(y, g, sub, frac=0.2, seed=fs)
            n_eff = len(np.unique(g[sub]))
            scores, info, tuned = {}, {}, {}  # arm -> (test, threshold, validation scores); extra columns; tuned probes
            for arm, Z in fold_reps.items():
                clf = fit_probe(Z[fit], y[fit], seed=args.seed)
                sv = probe_scores(clf, Z[val])
                scores[arm] = (probe_scores(clf, Z[te]), best_f1_threshold(y[val], sv), sv)
            torch.manual_seed(fs)
            sup = train_supervised(ResNet1D(), Xn[fit], y[fit], Xn[val], y[val], epochs=args.sup_epochs,
                                   seed=fs, device=args.device, log=None)
            sv = predict(sup, Xn[val], device=args.device)
            scores["supervised"] = (predict(sup, Xn[te], device=args.device), best_f1_threshold(y[val], sv), sv)
            if "tuned_probe" in extra:
                for arm, Z in fold_reps.items():
                    best = None
                    for C in args.tc_grid:  # ties keep the smaller C (stronger regularisation)
                        clf = fit_probe(Z[fit], y[fit], C=C, seed=args.seed)
                        sv = probe_scores(clf, Z[val])
                        ap = average_precision_score(y[val], sv)
                        if best is None or ap > best[0]:
                            best = (ap, C, clf, sv)
                    _, C, clf, sv = best
                    tuned[arm] = clf
                    scores[arm + "_tc"] = (probe_scores(clf, Z[te]), best_f1_threshold(y[val], sv), sv)
                    info[arm + "_tc"] = {"C": C}
            if "supervised_es" in extra:
                torch.manual_seed(fs)
                es = train_supervised(ResNet1D(), Xn[fit], y[fit], Xn[val], y[val], epochs=args.es_max_epochs,
                                      seed=fs, device=args.device, log=None, patience=args.patience)
                sv = predict(es, Xn[val], device=args.device)
                scores["supervised_es"] = (predict(es, Xn[te], device=args.device), best_f1_threshold(y[val], sv), sv)
                info["supervised_es"] = {"best_epoch": es.best_epoch, "epochs_run": es.epochs_run}
            if "harnet5_ft" in extra:
                torch.manual_seed(fs)
                enc, head = P.harnet_for_finetuning("harnet5")
                ft = train_supervised(enc, X[fit], y[fit], X[val], y[val], epochs=args.ft_max_epochs, lr=args.ft_lr,
                                      seed=fs, device=args.device, log=None, patience=args.patience,
                                      optimizer="adam_amsgrad", head=head)
                sv = predict(ft, X[val], device=args.device)
                scores["harnet5_ft"] = (predict(ft, X[te], device=args.device), best_f1_threshold(y[val], sv), sv)
                info["harnet5_ft"] = {"best_epoch": ft.best_epoch, "epochs_run": ft.epochs_run}
                del ft, enc, head
            if "harnet5_lpft" in extra:
                Z = fold_reps["harnet5"]
                clf = tuned.get("harnet5")
                if clf is None:  # the tuned harnet5 probe, as in tuned_probe
                    best = None
                    for C in args.tc_grid:
                        c = fit_probe(Z[fit], y[fit], C=C, seed=args.seed)
                        ap = average_precision_score(y[val], probe_scores(c, Z[val]))
                        if best is None or ap > best[0]:
                            best = (ap, c)
                    clf = best[1]
                torch.manual_seed(fs)
                enc, _ = P.harnet_for_finetuning("harnet5")
                head = P.linear_head_from_probe(clf, enc.emb_dim)
                lp = train_supervised(enc, X[fit], y[fit], X[val], y[val], epochs=args.ft_max_epochs, lr=args.lpft_lr,
                                      seed=fs, device=args.device, log=None, patience=args.patience,
                                      optimizer="adam_amsgrad", head=head, eval_before=True)
                sv = predict(lp, X[val], device=args.device)
                scores["harnet5_lpft"] = (predict(lp, X[te], device=args.device), best_f1_threshold(y[val], sv), sv)
                info["harnet5_lpft"] = {"best_epoch": lp.best_epoch, "epochs_run": lp.epochs_run,
                                        "C": float(clf.steps[-1][1].C)}
                del lp, enc, head
            budget = "all" if n >= ALL else str(n)
            for arm, (s, thr, sv) in scores.items():
                rep = binary_report(y[te], s, thr)
                rho, mae = pct_time_frozen(y[te], (s >= thr).astype(float), g[te])
                rows.append({"cohort": cohort, "fold": fold, "budget": budget, "n_train_patients": n_eff,
                             "arm": arm, **rep, "ptf_spearman": rho, "ptf_mae_pp": mae, "threshold": thr,
                             **info.get(arm, {})})
                oof.append(pd.DataFrame({"cohort": cohort, "fold": fold, "budget": budget, "arm": arm,
                                         "subject": g[te], "y": y[te], "score": s, "pred": (s >= thr).astype(int),
                                         "window": wi[te]}))
                vals.append(pd.DataFrame({"cohort": cohort, "fold": fold, "budget": budget, "arm": arm,
                                          "subject": g[val], "y": y[val], "score": sv, "window": wi[val],
                                          "threshold": thr}))
            best = max(scores, key=lambda a: roc_auc_score(y[te], scores[a][0]))
            log(f"    budget {budget:>3} (n={n_eff}): " + " ".join(
                f"{a}={roc_auc_score(y[te], scores[a][0]):.3f}" for a in scores) + f"  best={best}"
                + "".join(f"\n      {a}: {v}" for a, v in info.items() if "_tc" not in a)
                + (f"\n      C: " + " ".join(f"{a}={v['C']:g}" for a, v in info.items() if "_tc" in a)
                   if "tuned_probe" in extra else ""))
        log(f"  fold {fold} done in {(time.time() - t0) / 60:.1f} min")
    return rows, oof, vals


def summarize(oof: pd.DataFrame, per_fold: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    out = []
    for (cohort, budget, arm), d in oof.groupby(["cohort", "budget", "arm"]):
        auc, lo, hi = subject_bootstrap_ci(d.y.to_numpy(), d.score.to_numpy(), d.subject.to_numpy(), n_boot=n_boot)
        f = per_fold[(per_fold.cohort == cohort) & (per_fold.budget == budget) & (per_fold.arm == arm)]
        out.append({"cohort": cohort, "budget": budget, "arm": arm, "n_train_patients_mean": f.n_train_patients.mean(),
                    "auroc": auc, "auroc_lo": lo, "auroc_hi": hi,
                    "auprc_mean": f.auprc.mean(), "auprc_sd": f.auprc.std(),
                    "f1_pos_mean": f.f1_pos.mean(), "f1_macro_mean": f.f1_macro.mean(),
                    "sens_mean": f.sensitivity.mean(), "spec_mean": f.specificity.mean(),
                    "ptf_spearman_mean": f.ptf_spearman.mean(), "ptf_mae_pp_mean": f.ptf_mae_pp.mean(),
                    "prevalence": d.y.mean(), "folds": len(f)})
    order = {"5": 0, "10": 1, "20": 2, "all": 3}
    s = pd.DataFrame(out)
    return s.sort_values(["cohort", "budget", "auroc"], key=lambda c: c.map(order) if c.name == "budget" else c,
                         ascending=[True, True, False])


def main():
    run_experiment(base_parser().parse_args(), "rq12_fog_transfer")


def base_parser(cohorts=("tdcsfog", "defog")) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--budgets", type=int, nargs="+", default=[5, 10, 20, ALL])
    ap.add_argument("--ssl-epochs", type=int, default=10)
    ap.add_argument("--sup-epochs", type=int, default=6)
    ap.add_argument("--cohorts", nargs="+", default=list(cohorts))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--device", choices=["cpu", "auto", "mps", "cuda"], default="cpu",
                    help="cpu (default) reproduces the CPU results exactly; auto picks cuda, then mps, then cpu")
    ap.add_argument("--out", default=None, help="results/<out> (default <experiment name>, or _quick)")
    ap.add_argument("--extra", nargs="*", default=[],
                    choices=["tuned_probe", "supervised_es", "harnet5_ft", "harnet5_lpft"],
                    help="optional arms (see the docstring); default none")
    ap.add_argument("--tc-grid", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0])
    ap.add_argument("--patience", type=int, default=5, help="early stopping for supervised_es and harnet5_ft")
    ap.add_argument("--es-max-epochs", type=int, default=50)
    ap.add_argument("--ft-max-epochs", type=int, default=50)
    ap.add_argument("--ft-lr", type=float, default=1e-4)
    ap.add_argument("--lpft-lr", type=float, default=1e-5)
    return ap


def run_experiment(args, name: str, make_hook=None):
    """Runs the RQ1/RQ2 protocol and writes results/<out>. make_hook(args, log) may return a fold_hook that adds
    arms per fold (RQ4); without it the run is the plain RQ1/RQ2 experiment."""
    if args.quick:
        args.folds, args.ssl_epochs, args.sup_epochs, args.budgets, args.n_boot = 2, 1, 1, [5, ALL], 100
        args.es_max_epochs, args.ft_max_epochs, args.patience = 3, 3, 2
    args.device = pick_device(args.device)  # the device actually used goes into the log and config.json
    torch.set_num_threads(args.threads)
    out = ROOT / "results" / (args.out or (name + ("_quick" if args.quick else "")))
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "src", "scripts"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    log(f"=== {time.strftime('%Y-%m-%d %H:%M')} commit {commit}{' (uncommitted changes)' if dirty else ''} args {vars(args)}")
    W = F.load_windows(FS, WIN, HOP)
    reps = frozen_representations(W["X"], log, args.device, args.seed)
    fold_hook = make_hook(args, log) if make_hook else None
    rows, oof, vals = [], [], []
    for c in args.cohorts:
        r, o, v = run_cohort(c, W, reps, args, log, fold_hook=fold_hook)
        rows += r
        oof += o
        vals += v
        pd.DataFrame(rows).to_csv(out / "per_fold.csv", index=False)  # checkpoint after each cohort
    per_fold = pd.DataFrame(rows)
    oof = pd.concat(oof, ignore_index=True)
    oof.to_parquet(out / "out_of_fold_scores.parquet")
    pd.concat(vals, ignore_index=True).to_parquet(out / "validation_scores.parquet")
    summ = summarize(oof, per_fold, args.n_boot)
    summ.to_csv(out / "summary.csv", index=False)
    (out / "config.json").write_text(json.dumps({**vars(args), "commit": commit, "dirty": dirty,
                                                 "fs": FS, "win_s": WIN, "hop_s": HOP}, indent=2))
    pd.set_option("display.width", 220)
    log("\n" + summ[["cohort", "budget", "arm", "n_train_patients_mean", "auroc", "auroc_lo", "auroc_hi",
                     "auprc_mean", "f1_pos_mean", "f1_macro_mean", "ptf_spearman_mean", "ptf_mae_pp_mean"]]
        .round(3).to_string(index=False))
    log("DONE")


if __name__ == "__main__":
    main()
