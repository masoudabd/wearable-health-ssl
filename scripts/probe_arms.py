"""Add frozen representations to RQ1/RQ2 by replaying its probe protocol (no network training; CPU).

For each seed, cohort, fold and label budget, the patient folds (subject_kfold), the budget draw (subsample_subjects)
and the validation patients (carve_validation) are drawn exactly as scripts/rq12_fog_transfer.py draws them, and each
representation gets the default probe (C = 1) and the tuned probe (C chosen on validation AUPRC). The out-of-fold rows
come in the run's order, so each seed can be merged with its RQ1/RQ2 run for paired tests ("RUN_A+RUN_B" in
scripts/paired_tests.py). The replay is checked by re-scoring harnet5 (arm harnet5_replay), which must equal the
fixed run's harnet5 exactly.

Representations:
  forge_mc   the released FORGE medium-context encoder (whssl.models.forge), frozen, on 5 s windows at 100 Hz cut at
             the same times as the analysis windows; token mean (512). FORGE was pretrained on the daily-living
             recordings, which include 45 of the 57 home-cohort patients: clean for the lab cohort only (home results
             are reported with that caveat). Invariant to the axis-sign differences between datasets.

  harnet5_galign  harnet5 on the analysis windows rotated, per recording, so that the recording's mean acceleration
             (gravity and the sensor's mounting tilt) points along the network's -x axis, the gravity direction of
             its wrist pretraining data and of the default input's vertical channel; the minimal rotation (Rodrigues),
             so the heading about gravity is left as recorded. Probed like harnet5.

Usage: python scripts/probe_arms.py --seed 0 [--reps forge_mc harnet5_galign] [--device cpu]
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
from whssl.models import pretrained as P  # noqa: E402
from whssl.signal import resample  # noqa: E402

TC_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]


def windows_100hz(log=print) -> np.ndarray:
    """(N, 500, 3) windows at 100 Hz in g, cut at the start times of the 5 s analysis windows (cache order)."""
    path = F.PROC / "windows_100hz_5s_aligned.npz"
    W = F.load_windows(rq12.FS, rq12.WIN, rq12.HOP)
    if path.exists():
        X = np.load(path)["X"]
    else:
        fs_out, T, hop, T100 = rq12.FS, int(rq12.WIN * rq12.FS), int(rq12.HOP * rq12.FS), int(rq12.WIN * 100)
        parts, t0s = [], []
        recs = F.recordings()
        for i, rec in recs.iterrows():
            r = F.load_recording(rec)
            acc30 = resample(r["acc"], r["fs"], fs_out, axis=0)
            m = len(acc30)
            src = np.minimum((np.arange(m) * r["fs"] / fs_out).round().astype(int), len(r["fog"]) - 1)
            valid = r["valid"][src]
            starts = np.arange(0, m - T + 1, hop)
            if len(starts) == 0:
                continue
            keep = valid[starts[:, None] + np.arange(T)[None, :]].mean(axis=1) >= 1.0
            acc100 = resample(r["acc"], r["fs"], 100.0, axis=0).astype(np.float32)
            s100 = np.round(starts[keep] / fs_out * 100).astype(int)
            acc100 = np.pad(acc100, ((0, max(0, int(s100.max(initial=0)) + T100 - len(acc100))), (0, 0)), mode="edge")
            parts.append(acc100[s100[:, None] + np.arange(T100)[None, :]])
            t0s.append(starts[keep] / fs_out)
            if log and (i + 1) % 200 == 0:
                log(f"  100 Hz windows: {i + 1}/{len(recs)} recordings")
        X = np.concatenate(parts)
        if not np.allclose(np.concatenate(t0s), W["t0"]):
            raise SystemExit("100 Hz windows do not line up with the analysis windows")
        np.savez(path, X=X)
    if len(X) != len(W["X"]):
        raise SystemExit("100 Hz window count differs from the analysis windows")
    return X


def rotation_to(u: np.ndarray, t: np.ndarray) -> np.ndarray:
    """The minimal rotation matrix taking the direction of u to the direction of t (Rodrigues)."""
    u, t = u / np.linalg.norm(u), t / np.linalg.norm(t)
    k, c = np.cross(u, t), float(np.dot(u, t))
    s = np.linalg.norm(k)
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        raise SystemExit("antiparallel gravity: rotation undefined")
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]]) / s
    th = np.arctan2(s, c)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def gravity_aligned(log=print) -> np.ndarray:
    """Analysis windows rotated per recording so that the recording's mean acceleration points along -x."""
    path = F.PROC / "windows_30hz_5s_galign.npz"
    W = F.load_windows(rq12.FS, rq12.WIN, rq12.HOP)
    if path.exists():
        return np.load(path)["X"]
    X = W["X"].astype(np.float64)
    out = np.empty_like(W["X"])
    angles = {}
    for rec in np.unique(W["rec"]):
        m = W["rec"] == rec
        g = X[m].reshape(-1, 3).mean(axis=0)
        R = rotation_to(g, np.array([-1.0, 0.0, 0.0]))
        out[m] = (X[m] @ R.T).astype(np.float32)
        angles[rec] = float(np.degrees(np.arccos(np.clip(-g[0] / np.linalg.norm(g), -1, 1))))
    for cohort in np.unique(W["cohort"]):
        a = np.array([angles[r] for r in np.unique(W["rec"][W["cohort"] == cohort])])
        log(f"  gravity alignment, {cohort}: rotation per recording median {np.median(a):.1f} deg "
            f"(IQR {np.percentile(a, 25):.1f} to {np.percentile(a, 75):.1f}, max {a.max():.1f})")
    np.savez(path, X=out)
    return out


def representation(name: str, device: str, log) -> np.ndarray:
    if name == "harnet5_galign":
        return rq12.cached(F.PROC / "emb_harnet5_galign.npz", lambda: P.harnet_embed(
            P.load_harnet("harnet5", True), gravity_aligned(log), device=device))
    if name == "forge_mc":
        from whssl.models.forge import forge_embed, load_forge

        return rq12.cached(F.PROC / "emb_forge_mc.npz",
                           lambda: forge_embed(load_forge("mc"), windows_100hz(log), device=device))
    raise SystemExit(f"unknown representation {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reps", nargs="+", default=["forge_mc"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--budgets", type=int, nargs="+", default=[5, 10, 20, rq12.ALL])
    ap.add_argument("--cohorts", nargs="+", default=["tdcsfog", "defog"])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = ROOT / "results" / (args.out or f"probe_arms_seed{args.seed}")
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
    base = rq12.frozen_representations(W["X"], log, "cpu", args.seed)
    reps = {"harnet5_replay": base["harnet5"], **{r: representation(r, args.device, log) for r in args.reps}}
    rows, oof, vals = [], [], []
    for cohort in args.cohorts:
        m = W["cohort"] == cohort
        y, g, wi = W["y"][m].astype(int), W["subject"][m], np.flatnonzero(m)
        R = {k: v[m] for k, v in reps.items()}
        for fold, (tr, te) in enumerate(subject_kfold(y, g, args.folds, args.seed)):
            fs = args.seed * 1000 + fold  # as rq12_fog_transfer.run_cohort
            for n in args.budgets:
                sub = subsample_subjects(y, g, tr, n, seed=fs)
                fit, val = carve_validation(y, g, sub, frac=0.2, seed=fs)
                budget = "all" if n >= rq12.ALL else str(n)
                for name, Z in R.items():
                    variants = [(name, fit_probe(Z[fit], y[fit], seed=args.seed), 1.0)]
                    if name != "harnet5_replay":
                        best = None
                        for C in TC_GRID:
                            c = fit_probe(Z[fit], y[fit], C=C, seed=args.seed)
                            ap_ = average_precision_score(y[val], probe_scores(c, Z[val]))
                            if best is None or ap_ > best[0]:
                                best = (ap_, c, C)
                        variants.append((name + "_tc", best[1], best[2]))
                    for arm, clf, C in variants:
                        s, sv = probe_scores(clf, Z[te]), probe_scores(clf, Z[val])
                        thr = best_f1_threshold(y[val], sv)
                        rep = binary_report(y[te], s, thr)
                        rho, mae = rq12.pct_time_frozen(y[te], (s >= thr).astype(float), g[te])
                        rows.append({"cohort": cohort, "fold": fold, "budget": budget,
                                     "n_train_patients": len(np.unique(g[sub])), "arm": arm, **rep,
                                     "ptf_spearman": rho, "ptf_mae_pp": mae, "threshold": thr, "C": C})
                        oof.append(pd.DataFrame({"cohort": cohort, "fold": fold, "budget": budget, "arm": arm,
                                                 "subject": g[te], "y": y[te], "score": s,
                                                 "pred": (s >= thr).astype(int), "window": wi[te]}))
                        vals.append(pd.DataFrame({"cohort": cohort, "fold": fold, "budget": budget, "arm": arm,
                                                  "subject": g[val], "y": y[val], "score": sv, "window": wi[val],
                                                  "threshold": thr}))
            log(f"  {cohort} fold {fold}: " + " ".join(
                f"{r['arm']}={r['auroc']:.3f}" for r in rows[-len(R) * 2 + 1:] if r["budget"] == "all"))
    per_fold, oof = pd.DataFrame(rows), pd.concat(oof, ignore_index=True)
    per_fold.to_csv(out / "per_fold.csv", index=False)
    oof.to_parquet(out / "out_of_fold_scores.parquet")
    pd.concat(vals, ignore_index=True).to_parquet(out / "validation_scores.parquet")
    rq12.summarize(oof, per_fold, args.n_boot).to_csv(out / "summary.csv", index=False)
    (out / "config.json").write_text(json.dumps({**vars(args), "commit": commit, "dirty": dirty, "fs": rq12.FS,
                                                 "win_s": rq12.WIN, "hop_s": rq12.HOP, "device": args.device}, indent=2))
    log("DONE")


if __name__ == "__main__":
    main()
