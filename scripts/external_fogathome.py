"""External test on the FogAtHome recordings of the FORGE release (decision 21; whssl.data.fogathome).

Detectors are trained on one contest cohort (home by default, the natural source for home and free-living
recordings; the lab cohort as a second source) and tested without any retraining or threshold change on
  structured   FogAtHome FOG-provoking home recordings (12 participants)
  daily        FogAtHome naturalistic free living (11 participants, about 294 h of valid windows); scored over all
               valid windows and inside the walking-and-standing domain (activity codes 1 and 4), where the FORGE card
               reports its own detector (AUROC 0.803 for its nine-head ensemble)
Per repetition (seed), the source's patients are split into fitting and validation patients (one fifth); validation
patients choose the probe's C (tuned arms) and the thresholds:
  f1           highest window F1 on the source validation patients
  burden       share of source validation windows predicted FOG equals the share annotated
  burden_time  the source validation patients' predicted FOG time equals their annotated FOG time, both as elapsed time
               over analyzed time (analysis h's definition; the rule of scripts/paper_val_operating.py, whose bisection
               is used here, with the contest annotation tables of results/paper_derived/annotations)
Arms: feat60_fi and harnet5, each with the default and the tuned probe. FogAtHome axes are mapped to the contest
convention before any representation is computed: for the home cohort as source, --axes Vflip (vertical flipped, the
default) or VMLflip (also mediolateral, the sensitivity option); for the lab cohort, whose anteroposterior sign is the
opposite of the home cohort's, VAPflip or VAPMLflip.

Reported per test set: pooled AUROC and AUPRC; at each threshold, sensitivity and specificity of windows, false alarms
per hour of analyzed recording (a false alarm is a run of consecutive predicted-FOG windows of one session that touches
no annotated FOG), the share of predicted episodes that touch FOG, and per-patient percent time frozen against
annotated, as the mean signed error: in elapsed time (columns *_ptf_bias_pp_time; scripts/fogathome_time.py, analysis
h's definition, the endpoint the paper reports) and as the share of windows (*_ptf_bias_pp, kept for comparison).
Patient-bootstrap intervals (2,000 resamples of the test patients by default, shared by arms and seeds; per resample
the statistic is averaged over seeds).

Usage: python scripts/external_fogathome.py [--source defog] [--axes Vflip] [--device mps]
  (embeds the FogAtHome windows with harnet5 once per axis mode: minutes on the GPU, about 90 on the CPU)
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
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rq12 = _load("rq12_fog_transfer")

import paper_common as PC  # noqa: E402
from fogathome_time import ElapsedTime  # noqa: E402
from paper_ptf_time import patient_time, window_lookup  # noqa: E402
from paper_val_operating import burden_time_threshold  # noqa: E402

from whssl.data import fog_contest as F  # noqa: E402
from whssl.data import fogathome as FH  # noqa: E402
from whssl.eval.metrics import best_f1_threshold  # noqa: E402
from whssl.eval.paired import boot_auroc, patient_counts, weighted_auroc  # noqa: E402
from whssl.eval.probe import fit_probe, probe_scores  # noqa: E402
from whssl.eval.splits import carve_validation  # noqa: E402
from whssl.features import feat60, freeze_index_features  # noqa: E402
from whssl.models import pretrained as P  # noqa: E402

AXES = {"Vflip": (-1.0, 1.0, 1.0), "VMLflip": (-1.0, -1.0, 1.0),  # to the home-cohort convention
        "VAPflip": (-1.0, 1.0, -1.0), "VAPMLflip": (-1.0, -1.0, -1.0)}  # to the lab-cohort convention (AP sign differs)
TC_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
N_BOOT = 2000


def target_reps(split: str, axes: str, device: str, log) -> tuple[dict, dict]:
    W = FH.load_windows(split, AXES[axes])
    proc = FH.PROC
    f60 = rq12.cached(proc / f"{split}_{axes}_feat60.npz", lambda: feat60(W["X"], rq12.FS)[0])
    ffi = rq12.cached(proc / f"{split}_{axes}_freeze_index.npz", lambda: freeze_index_features(W["X"], rq12.FS)[0])
    t = time.time()
    h = rq12.cached(proc / f"{split}_{axes}_emb_harnet5.npz",
                    lambda: P.harnet_embed(P.load_harnet("harnet5", True), W["X"], device=device))
    log(f"  {split} ({axes}): {len(W['y']):,} windows, representations ready ({time.time() - t:.0f}s)")
    return W, {"feat60_fi": np.concatenate([f60, ffi], 1), "harnet5": h}


def episodes(W: dict, pred: np.ndarray, hop: float = 2.5, win: float = 5.0) -> dict:
    """False alarms, predicted episodes and analyzed hours per patient (runs within a session, windows hop apart)."""
    rec, t0 = W["rec"], W["t0"]
    cont = np.r_[False, (rec[1:] == rec[:-1]) & (np.abs(np.diff(t0) - hop) < 1e-6)]
    pts, col = np.unique(W["subject"], return_inverse=True)
    hours = np.bincount(col, (np.full(len(t0), hop) + (~cont) * (win - hop)) / 3600, minlength=len(pts))
    start = pred & ~(np.r_[False, pred[:-1]] & cont)
    run = np.cumsum(start) - 1
    ip = np.flatnonzero(pred)
    touch = np.zeros(int(start.sum()), bool)
    np.logical_or.at(touch, run[ip], W["frac"][ip] > 0)
    rcol = col[np.flatnonzero(start)]
    return {"fa": np.bincount(rcol[~touch], minlength=len(pts)), "eps": np.bincount(rcol, minlength=len(pts)),
            "hours": hours, "pts": pts, "col": col}


def main():
    global N_BOOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["defog", "tdcsfog"], default="defog")
    ap.add_argument("--axes", choices=list(AXES), default="Vflip")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--splits", nargs="+", default=["fogathome", "fogathome_dailyliving"], choices=list(FH.SPLITS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()
    N_BOOT = args.n_boot
    out = ROOT / "results" / (args.out or f"external_fogathome_{args.source}_{args.axes}")
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
    Wc = F.load_windows(rq12.FS, rq12.WIN, rq12.HOP)
    reps_c = rq12.frozen_representations(Wc["X"], log, "cpu", 0)
    reps_c = {"feat60_fi": reps_c["feat60_fi"], "harnet5": reps_c["harnet5"]}
    m = np.flatnonzero(Wc["cohort"] == args.source)
    ys, gs = Wc["y"][m].astype(int), Wc["subject"][m]
    # elapsed FOG time of the source patients, for the burden_time threshold (analysis h's tables and functions)
    A = PC.ROOT / "results" / "paper_derived" / "annotations"
    ann, recs_c = pd.read_parquet(A / "windows.parquet"), pd.read_csv(A / "recordings.csv")
    if not np.array_equal(ann.sort_values("win").y.to_numpy(), Wc["y"].astype(int)):
        raise SystemExit("the window cache and the annotation tables disagree; rerun scripts/paper_annotations.py")
    pt_c = patient_time(ann, pd.read_csv(A / "episodes.csv"), recs_c)
    pt_c = pt_c[pt_c.cohort == args.source].set_index("subject")
    lookup_c = window_lookup(ann, recs_c)
    tests, elapsed = {}, {}
    for split, name in [(s_, n_) for s_, n_ in [("fogathome", "structured"), ("fogathome_dailyliving", "daily")]
                        if s_ in args.splits]:
        W, R = target_reps(split, args.axes, args.device, log)
        tests[name] = (W, R, np.ones(len(W["y"]), bool))
        elapsed[name] = ElapsedTime(W, split)
        if name == "daily":
            tests["daily_walkstand"] = (W, R, np.isin(W["activity"], FH.WALK_STAND_CODES))
    scores, rows = {}, []
    for seed in args.seeds:
        fit, val = carve_validation(ys, gs, np.arange(len(m)), frac=0.2, seed=seed * 1000 + 700)
        pv = pt_c.loc[np.unique(gs[val])]
        if np.isin(gs, pv.index).sum() != len(val):
            raise SystemExit("validation rows are not all the windows of the validation patients")
        target = pv.annotated_fog_s.sum() / pv.analyzed_s.sum()
        for arm in ["feat60_fi", "harnet5"]:
            Z = reps_c[arm][m]
            clf = fit_probe(Z[fit], ys[fit], seed=seed)
            best = None
            for C in TC_GRID:
                c = fit_probe(Z[fit], ys[fit], C=C, seed=seed)
                ap_ = average_precision_score(ys[val], probe_scores(c, Z[val]))
                if best is None or ap_ > best[0]:
                    best = (ap_, c, C)
            for name_arm, c in [(arm, clf), (arm + "_tc", best[1])]:
                sv = probe_scores(c, Z[val])
                thr = {"f1": best_f1_threshold(ys[val], sv), "burden": float(np.quantile(sv, 1 - ys[val].mean())),
                       "burden_time": burden_time_threshold(m[val], sv, target, pv.analyzed_s.sum(), lookup_c)[0]}
                for test, (W, R, keep) in tests.items():
                    s = probe_scores(c, R[arm])
                    scores[seed, name_arm, test] = (s, thr)
        log(f"  seed {seed} done")
    # summaries with patient-bootstrap intervals, statistics averaged over seeds within each resample
    arms = ["feat60_fi", "feat60_fi_tc", "harnet5", "harnet5_tc"]
    per_patient = []   # one row per participant, arm and threshold rule: the free-living spread the pooled rates hide
    for test, (W, R, keep) in tests.items():
        y = W["y"][keep].astype(int)
        pts, col = np.unique(W["subject"][keep], return_inverse=True)
        counts = patient_counts(len(pts), N_BOOT, seed=0)
        boot = {}
        for arm in arms:
            au = [weighted_auroc(scores[s, arm, test][0][keep], y, np.ones(len(y))) for s in args.seeds]
            bt = np.mean([boot_auroc(scores[s, arm, test][0][keep], y, col, counts) for s in args.seeds], axis=0)
            boot[arm] = bt
            ap_ = np.mean([average_precision_score(y, scores[s, arm, test][0][keep]) for s in args.seeds])
            row = {"source": args.source, "axes": args.axes, "test": test, "arm": arm, "n_patients": len(pts),
                   "windows": int(keep.sum()), "prevalence": float(y.mean()), "auroc": float(np.mean(au)),
                   "auroc_lo": float(np.nanpercentile(bt, 2.5)), "auroc_hi": float(np.nanpercentile(bt, 97.5)),
                   "auprc": float(ap_)}
            if test != "daily_walkstand":
                E = elapsed[test]
                for rule in ("f1", "burden", "burden_time"):
                    fa, eps, hrs, bias, bias_t, sens, spec = [], [], [], [], [], [], []
                    e_pts = None
                    for s in args.seeds:
                        sc, thr = scores[s, arm, test]
                        pred = sc >= thr[rule]
                        e = episodes(W, pred)
                        e_pts = e["pts"]
                        if not np.array_equal(e_pts, E.pts):
                            raise SystemExit("participant order differs between episode counts and elapsed time")
                        bias_t.append(E.ptf_pred(pred) - E.ptf_true)
                        fa.append(e["fa"]); eps.append(e["eps"]); hrs.append(e["hours"])
                        ptf_p = np.bincount(e["col"], pred, minlength=len(e["pts"])) / np.bincount(e["col"], minlength=len(e["pts"]))
                        ptf_t = np.bincount(e["col"], W["y"], minlength=len(e["pts"])) / np.bincount(e["col"], minlength=len(e["pts"]))
                        bias.append(100 * (ptf_p - ptf_t))
                        sens.append(pred[W["y"] == 1].mean()); spec.append(1 - pred[W["y"] == 0].mean())
                    fah = [f.sum() / h.sum() for f, h in zip(fa, hrs)]
                    fa_b = np.mean([(counts @ f) / (counts @ h) for f, h in zip(fa, hrs)], axis=0)
                    pre = [1 - f.sum() / max(ep.sum(), 1) for f, ep in zip(fa, eps)]
                    b_b = np.mean([(counts @ b) / counts.sum(1) for b in bias], axis=0)
                    bt_b = np.mean([(counts @ b) / counts.sum(1) for b in bias_t], axis=0)
                    row.update({f"{rule}_sens": float(np.mean(sens)), f"{rule}_spec": float(np.mean(spec)),
                                f"{rule}_fa_per_hour": float(np.mean(fah)),
                                f"{rule}_fa_per_hour_lo": float(np.percentile(fa_b, 2.5)),
                                f"{rule}_fa_per_hour_hi": float(np.percentile(fa_b, 97.5)),
                                f"{rule}_episode_precision": float(np.mean(pre)),
                                f"{rule}_ptf_bias_pp": float(np.mean([b.mean() for b in bias])),
                                f"{rule}_ptf_bias_lo": float(np.percentile(b_b, 2.5)),
                                f"{rule}_ptf_bias_hi": float(np.percentile(b_b, 97.5)),
                                f"{rule}_ptf_bias_pp_time": float(np.mean([b.mean() for b in bias_t])),
                                f"{rule}_ptf_bias_time_lo": float(np.percentile(bt_b, 2.5)),
                                f"{rule}_ptf_bias_time_hi": float(np.percentile(bt_b, 97.5))})
                    # the same quantities per participant, averaged over seeds, so the spread can be shown
                    fa_h = np.mean([f / np.maximum(h, 1e-9) for f, h in zip(fa, hrs)], axis=0)
                    for i, p in enumerate(e_pts):
                        per_patient.append({"source": args.source, "axes": args.axes, "test": test, "arm": arm,
                                            "rule": rule, "subject": p,
                                            "hours": float(np.mean([h[i] for h in hrs])),
                                            "fa_per_hour": float(fa_h[i]),
                                            "ptf_bias_pp": float(np.mean([b[i] for b in bias])),
                                            "ptf_bias_pp_time": float(np.mean([b[i] for b in bias_t]))})
            rows.append(row)
        for a, b in [("harnet5", "feat60_fi"), ("harnet5_tc", "feat60_fi_tc")]:
            d = boot[a] - boot[b]
            pa = [r for r in rows if r["test"] == test and r["arm"] == a][0]
            pb = [r for r in rows if r["test"] == test and r["arm"] == b][0]
            rows.append({"source": args.source, "axes": args.axes, "test": test, "arm": f"{a} - {b}",
                         "n_patients": len(pts), "auroc": pa["auroc"] - pb["auroc"],
                         "auroc_lo": float(np.nanpercentile(d, 2.5)), "auroc_hi": float(np.nanpercentile(d, 97.5)),
                         "p": float(min(1.0, max(2 * min(np.mean(d <= 0), np.mean(d >= 0)), 1 / len(d))))})
    res = pd.DataFrame(rows)
    res.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(per_patient).to_csv(out / "per_participant.csv", index=False)
    (out / "config.json").write_text(json.dumps({**vars(args), "commit": commit, "dirty": dirty}, indent=2))
    pd.set_option("display.width", 250)
    log(res[["test", "arm", "n_patients", "prevalence", "auroc", "auroc_lo", "auroc_hi", "auprc", "f1_fa_per_hour",
             "burden_fa_per_hour", "burden_time_fa_per_hour", "f1_ptf_bias_pp_time", "burden_ptf_bias_pp_time",
             "burden_time_ptf_bias_pp_time"]].round(3).to_string(index=False))
    log("DONE")


if __name__ == "__main__":
    main()
