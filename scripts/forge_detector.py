"""Score the released FORGE detector on the home cohort: a published, FOG-specific comparator for Paper 1.

The FORGE release (huggingface.co/Liornis/forge-fog) ships 33 trained DeFOG classifiers, of which the model card
names nine as the released detector: one frozen medium-context encoder with a BiGRU head, for three
participant-level folds and seeds 42, 43 and 44. This script runs them on our own DeFOG windows.

Fold alignment (the condition for using any of this). Each released head names its split in its safetensors
metadata and in the public code's release/manifest.yaml, and the split file lists participant codes:
configs/data/splits/kaggle_labeled/kfold_defog_fogcount_valid_mc{0,1,2}.yaml. The three folds are a 19 / 19 / 19
participant partition of exactly the 57 DeFOG patients of this project, with train, val and test disjoint inside
each fold (checked here and in tests/test_forge_detector.py). Every patient is therefore scored by the heads of the
one fold that has it in `test`, so no head sees a patient it was trained or validated on, and the three folds
together give one out-of-fold score per window of all 57 patients. This is the authors' own DeFOG protocol.

The comparison with our arms is unpaired by construction: FORGE's three folds are not our five, its heads are
trained on 19 patients each where ours are trained on about 45, and its windows and labels differ from ours. The
numbers from our runs are printed beside FORGE's for context, on the same patients and the same windows, and no
paired test is computed.

Pretraining overlap. FORGE's encoder was pretrained by masked autoencoding on unlabeled at-home recordings, and the
public split (configs/data/splits/kaggle_daily/ssl_pretraining.yaml) lists participant codes identical, as a set, to
the 65 participants of the contest's daily-living release (data/raw/fog_contest/.../daily_metadata.csv). 45 of our 57
home patients are in it, so the encoder saw the unlabeled recordings of most of the patients it is tested on here.
Reported per fold in folds.csv and overlap.json.

Scoring variants, all written to summary.csv, with the authors' own choices marked `primary`:
  reading    softmax(logits)[1], as scripts/eval/eval_comprehensive.py reads it, or sigmoid(logit 1), as the
             training loss (FogRatioLoss) implies
  ensemble   mean over the three seeds of the patient's fold, or seed 42 alone (the card's DeFOG reference)
  labels     our window labels (FOG for at least half the window) or FORGE's (any FOG sample in the window)
  threshold  the released fixed operating point 0.35, or the threshold that maximizes positive-class F1 on the
             fold's 19 validation patients (this project's rule; those patients are not in any test set)

Input units. The detector's RevIN rescales every window, so its output does not depend on whether the input is in g
or in m/s^2 beyond RevIN's epsilon of 1e-5 on the variance. Checked on fold 0's validation patients (never in any
test set): AUROC 0.8717 in g against 0.8712 in m/s^2, AP 0.4447 against 0.4435. Our windows are in g, as for every
other arm.

Usage: .venv/bin/python scripts/forge_detector.py [--device mps] [--n-boot 20000] [--out forge_detector]
                                                  [--reuse-scores]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata
from sklearn.metrics import confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import paper_common as C  # noqa: E402  (shared bootstrap and formatting helpers; writes nothing itself)

from whssl.data import fog_contest as F  # noqa: E402
from whssl.eval.metrics import best_f1_threshold  # noqa: E402
from whssl.eval.paired import patient_counts  # noqa: E402

FORGE = ROOT / "data" / "raw" / "forge"
SPLITS = FORGE / "code" / "forge-public" / "configs" / "data" / "splits"
HEADS = FORGE / "model" / "classification"
MANIFEST = FORGE / "code" / "forge-public" / "release" / "manifest.yaml"
SEEDS = (42, 43, 44)
FOLDS = (0, 1, 2)
RELEASED_THRESHOLD = 0.35
OUR_RUNS = [f"results/rq12_fog_transfer_mps_fixed_seed{s}" for s in range(5)]
TUNED_RUNS = [f"results/rq12_tuned_ft_seed{s}" for s in range(5)]
OUR_ARMS = ["harnet5", "feat60_fi", "ssl_indomain", "supervised"]
TUNED_ARMS = ["harnet5_tc", "feat60_fi_tc", "harnet5_ft"]


def head_name(fold: int, seed: int) -> str:
    return f"mc_probe_fold{fold}" if seed == 42 else f"mc_probe_s{seed}_fold{fold}"


def fold_splits() -> dict[int, dict[str, list[str]]]:
    """The three released MC folds, read from the public code's split files."""
    out = {}
    for fold in FOLDS:
        s = yaml.safe_load((SPLITS / "kaggle_labeled" / f"kfold_defog_fogcount_valid_mc{fold}.yaml").read_text())
        out[fold] = {k: [str(x) for x in s[k]] for k in ("train", "val", "test")}
    return out


def pretraining_participants() -> set[str]:
    s = yaml.safe_load((SPLITS / "kaggle_daily" / "ssl_pretraining.yaml").read_text())
    return {str(x) for v in s.values() for x in v}


def check_alignment(splits: dict, subjects: np.ndarray) -> None:
    """Refuse to go on unless the released folds are a clean partition of our home patients."""
    ours = set(subjects)
    tests = [set(splits[f]["test"]) for f in FOLDS]
    problems = []
    if set().union(*tests) != ours:
        problems.append(f"the three test sets cover {len(set().union(*tests))} patients, not our {len(ours)}")
    for i, j in ((0, 1), (0, 2), (1, 2)):
        if tests[i] & tests[j]:
            problems.append(f"folds {i} and {j} share test patients: {sorted(tests[i] & tests[j])}")
    for f in FOLDS:
        s = splits[f]
        if set(s["train"]) & set(s["val"]) or set(s["train"]) & set(s["test"]) or set(s["val"]) & set(s["test"]):
            problems.append(f"fold {f}: train, val and test are not disjoint")
        if set(s["train"]) | set(s["val"]) | set(s["test"]) != ours:
            problems.append(f"fold {f}: train, val and test do not cover our patients")
    if problems:
        raise SystemExit("released fold membership does not line up with our home cohort, so no fair comparison is "
                         "possible:\n  " + "\n  ".join(problems))


def manifest_heads() -> pd.DataFrame:
    """The released nine-head ensemble as release/manifest.yaml declares it."""
    m = yaml.safe_load(MANIFEST.read_text())
    rows = [{"name": e["name"], "context": e["context"], "phase": e["phase"], "fold": e["fold"], "seed": e["seed"],
             "splits": e["splits"], "headline": bool(e.get("headline", False))} for e in m["classification"]]
    return pd.DataFrame(rows)


def score_windows(W: dict, X100: np.ndarray, splits: dict, device: str, batch_size: int, log) -> pd.DataFrame:
    """One row per (head, window) for the test and validation patients of that head's fold."""
    from whssl.models.forge import forge_detector_probs, load_forge_detector

    home = W["cohort"] == "defog"
    subject, y, frac = W["subject"][home], W["y"][home].astype(int), W["frac"][home]
    window = np.flatnonzero(home)
    parts = []
    for fold in FOLDS:
        role = np.where(np.isin(subject, splits[fold]["test"]), "test",
                        np.where(np.isin(subject, splits[fold]["val"]), "val", "train"))
        use = np.flatnonzero(role != "train")
        for seed in SEEDS:
            t0 = time.time()
            det = load_forge_detector(head_name(fold, seed))
            if det.metadata["splits"] != f"kaggle_labeled/kfold_defog_fogcount_valid_mc{fold}":
                raise SystemExit(f"{head_name(fold, seed)}: metadata names split {det.metadata['splits']}")
            if int(det.metadata["seed"]) != seed or int(det.metadata["fold"]) != fold:
                raise SystemExit(f"{head_name(fold, seed)}: metadata fold/seed disagree with the file name")
            soft, sig = forge_detector_probs(det, X100[window[use]], batch_size=batch_size, device=device)
            parts.append(pd.DataFrame({"fold": fold, "seed": seed, "role": role[use], "subject": subject[use],
                                       "window": window[use], "y": y[use], "frac": frac[use],
                                       "softmax": soft, "sigmoid": sig}))
            log(f"  fold {fold} seed {seed}: {len(use):,} windows in {time.time() - t0:.0f} s "
                f"(test {int((role[use] == 'test').sum()):,}, val {int((role[use] == 'val').sum()):,})")
    return pd.concat(parts, ignore_index=True)


def agreement(x: np.ndarray, z: np.ndarray) -> dict:
    """Spearman, ICC(A,1), mean absolute error and signed bias of percent time frozen, as
    scripts/paper_ptf_agreement.py computes them; x annotated, z predicted, both (B, n)."""
    n = x.shape[1]
    rx, rz = rankdata(x, axis=1), rankdata(z, axis=1)
    rx, rz = rx - rx.mean(1, keepdims=True), rz - rz.mean(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = (rx * rz).sum(1) / np.sqrt((rx ** 2).sum(1) * (rz ** 2).sum(1))
        g = (x.mean(1) + z.mean(1)) / 2
        m = (x + z) / 2
        msr = 2 * ((m - g[:, None]) ** 2).sum(1) / (n - 1)
        msc = n * ((x.mean(1) - g) ** 2 + (z.mean(1) - g) ** 2)
        sst = ((x - g[:, None]) ** 2).sum(1) + ((z - g[:, None]) ** 2).sum(1)
        mse = (sst - (n - 1) * msr - msc) / (n - 1)
        icc = (msr - mse) / (msr + mse + 2 / n * (msc - mse))
    return {"ptf_rho": rho, "ptf_icc": icc, "ptf_mae_pp": np.abs(z - x).mean(1), "ptf_bias_pp": (z - x).mean(1)}


def discrimination(d: pd.DataFrame, score: str, label: str, counts: np.ndarray, col: np.ndarray) -> dict:
    s, y = d[score].to_numpy(), d[label].to_numpy()
    rows = {}
    for key, fn in (("auroc", C.weighted_auroc), ("auprc", C.weighted_auprc)):
        boot = C.boot_metric(fn, s, y, col, counts)
        lo, hi = C.interval(boot)
        rows[key], rows[key + "_lo"], rows[key + "_hi"] = fn(s, y, np.ones(len(y))), lo, hi
    return rows


def threshold_metrics(d: pd.DataFrame, score: str, label: str, thr: np.ndarray, counts: np.ndarray,
                      col: np.ndarray, pts: np.ndarray) -> dict:
    """Window metrics at a per-window threshold (one value per fold) and percent-time-frozen agreement."""
    s, y = d[score].to_numpy(), d[label].to_numpy()
    pred = (s >= thr).astype(int)
    tn, fp, fn_, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {"f1_pos": float(f1_score(y, pred, zero_division=0)),
           "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
           "sensitivity": float(tp / max(tp + fn_, 1)), "specificity": float(tn / max(tn + fp, 1)),
           "predicted_positive_rate": float(pred.mean())}
    per = pd.DataFrame({"subject": d.subject.to_numpy(), "y": y, "pred": pred}).groupby("subject").mean() * 100
    per = per.reindex(pts)
    idx = np.stack([np.repeat(np.arange(len(pts)), c.astype(int)) for c in counts])
    point = agreement(per.y.to_numpy()[None, :], per.pred.to_numpy()[None, :])
    boot = agreement(per.y.to_numpy()[idx], per.pred.to_numpy()[idx])
    for k in point:
        lo, hi = C.interval(boot[k])
        out[k], out[k + "_lo"], out[k + "_hi"] = float(point[k][0]), lo, hi
    return out


def our_arms(runs: list[str], arms: list[str], patients: set[str], counts: np.ndarray, log) -> list[dict]:
    """Pooled out-of-fold AUROC and AUPRC of our own home arms on the same patients, averaged over seeds."""
    rows = []
    for arm in arms:
        per_seed, boots = [], []
        for run in runs:
            p = ROOT / run / "out_of_fold_scores.parquet"
            if not p.exists():
                log(f"  {run} missing, skipped")
                continue
            d = pd.read_parquet(p, columns=["cohort", "budget", "arm", "subject", "y", "score"])
            d = d[(d.cohort == "defog") & (d.budget == "all") & (d.arm == arm)]
            if d.empty:
                continue
            if set(d.subject) != patients:
                raise SystemExit(f"{run}: arm {arm} covers different patients")
            d = d.sort_values("subject", kind="mergesort")
            pts, col = C.patient_index(d.subject.to_numpy())
            s, y = d.score.to_numpy(), d.y.to_numpy()
            per_seed.append({"auroc": C.auroc(s, y), "auprc": C.auprc(s, y)})
            boots.append({"auroc": C.boot_metric(C.weighted_auroc, s, y, col, counts),
                          "auprc": C.boot_metric(C.weighted_auprc, s, y, col, counts)})
        if not per_seed:
            continue
        row = {"source": "ours", "arm": arm, "n_seeds": len(per_seed), "n_patients": len(patients)}
        for k in ("auroc", "auprc"):
            mean_boot = np.mean([b[k] for b in boots], axis=0)
            lo, hi = C.interval(mean_boot)
            row[k] = float(np.mean([p[k] for p in per_seed]))
            row[k + "_lo"], row[k + "_hi"] = lo, hi
        rows.append(row)
        log(f"  {arm}: AUROC {row['auroc']:.3f} [{row['auroc_lo']:.3f}, {row['auroc_hi']:.3f}], "
            f"AUPRC {row['auprc']:.3f} over {row['n_seeds']} repetitions")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--n-boot", type=int, default=20_000)
    ap.add_argument("--out", default="forge_detector")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--reuse-scores", action="store_true",
                    help="read per_window_scores.parquet from a previous run instead of scoring again")
    a = ap.parse_args()
    t_start = time.time()
    out = ROOT / "results" / a.out
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    commit = C.git_commit()
    log(f"=== {time.strftime('%Y-%m-%d %H:%M')} commit {commit} args {vars(a)}")

    W = F.load_windows()
    home = W["cohort"] == "defog"
    subjects = np.array(sorted(set(W["subject"][home])))
    splits = fold_splits()
    check_alignment(splits, W["subject"][home])
    log(f"fold alignment verified: three released MC folds partition the {len(subjects)} home patients "
        f"({[len(splits[f]['test']) for f in FOLDS]} test patients per fold)")

    pre = pretraining_participants()
    daily = set(pd.read_csv(F.RAW / "daily_metadata.csv").Subject.astype(str))
    overlap = {"forge_pretraining_participants": len(pre),
               "contest_daily_living_participants": len(daily),
               "identical_sets": pre == daily,
               "home_patients_in_pretraining": int(len(set(subjects) & pre)),
               "test_patients_in_pretraining_per_fold": {str(f): int(len(set(splits[f]["test"]) & pre))
                                                         for f in FOLDS}}
    log(f"pretraining overlap: FORGE's {len(pre)} pretraining participants are "
        f"{'the same set as' if pre == daily else 'NOT the same set as'} the contest's {len(daily)} daily-living "
        f"participants; {overlap['home_patients_in_pretraining']} of {len(subjects)} home patients are in it "
        f"({overlap['test_patients_in_pretraining_per_fold']} of the 19 test patients per fold)")

    folds_tbl = pd.DataFrame([{"subject": s, **{f"fold{f}_role": ("test" if s in splits[f]["test"] else
                                                                 "val" if s in splits[f]["val"] else "train")
                                                for f in FOLDS},
                               "in_forge_pretraining": s in pre} for s in subjects])

    cached = out / "per_window_scores.parquet"
    if a.reuse_scores and cached.exists():
        scores = pd.read_parquet(cached)
        log(f"reusing {len(scores):,} scored windows from {cached}")
    else:
        X100 = np.load(F.PROC / "windows_100hz_5s_aligned.npz")["X"]
        if len(X100) != len(W["X"]):
            raise SystemExit("the 100 Hz window cache does not match the analysis windows")
        log(f"scoring nine released heads on {int(home.sum()):,} home windows at 100 Hz (device {a.device})")
        scores = score_windows(W, X100, splits, a.device, a.batch_size, log)
        scores.to_parquet(cached, index=False)

    scores["any_fog"] = (scores.frac > 0).astype(int)
    test = scores[scores.role == "test"]
    val = scores[scores.role == "val"]

    # One row per window: seed 42 alone, and the mean over the patient's three seeds.
    keys = ["fold", "subject", "window", "y", "any_fog"]
    pooled = {"mean3": test.groupby(keys, as_index=False)[["softmax", "sigmoid"]].mean(),
              "s42": test[test.seed == 42][keys + ["softmax", "sigmoid"]].copy()}
    val_pooled = {"mean3": val.groupby(keys, as_index=False)[["softmax", "sigmoid"]].mean(),
                  "s42": val[val.seed == 42][keys + ["softmax", "sigmoid"]].copy()}
    for k, d in pooled.items():
        if len(d) != int(home.sum()):
            raise SystemExit(f"{k}: {len(d):,} out-of-fold windows, expected {int(home.sum()):,}")

    counts = patient_counts(len(subjects), a.n_boot, seed=0)
    rows, thr_rows, pp_rows = [], [], []
    for ens, d in pooled.items():
        d = d.sort_values("subject", kind="mergesort").reset_index(drop=True)
        pts, col = C.patient_index(d.subject.to_numpy())
        if not np.array_equal(pts, subjects):
            raise SystemExit("out-of-fold patients differ from the home cohort")
        for reading in ("softmax", "sigmoid"):
            for label in ("y", "any_fog"):
                rows.append({"source": "forge", "arm": "forge_mc_probe", "ensemble": ens, "reading": reading,
                             "labels": "ours_majority" if label == "y" else "forge_any_fog",
                             "n_patients": len(pts), "n_windows": len(d), "prevalence": float(d[label].mean()),
                             "primary": ens == "mean3" and reading == "softmax" and label == "y",
                             **discrimination(d, reading, label, counts, col)})
        # Threshold metrics for the authors' reading and our labels only.
        vd = val_pooled[ens]
        val_thr = {f: best_f1_threshold(vd[vd.fold == f].y.to_numpy(), vd[vd.fold == f].softmax.to_numpy())
                   for f in FOLDS}
        for tname, thr in (("released_0.35", np.full(len(d), RELEASED_THRESHOLD)),
                           ("validation_f1", d.fold.map(val_thr).to_numpy())):
            m = threshold_metrics(d, "softmax", "y", thr, counts, col, pts)
            thr_rows.append({"source": "forge", "arm": "forge_mc_probe", "ensemble": ens, "reading": "softmax",
                             "labels": "ours_majority", "threshold": tname,
                             "threshold_value": "0.35" if tname == "released_0.35"
                             else "; ".join(f"fold{f}={val_thr[f]:.3f}" for f in FOLDS),
                             "primary": ens == "mean3" and tname == "released_0.35", **m})
            per = pd.DataFrame({"subject": d.subject, "y": d.y, "pred": (d.softmax.to_numpy() >= thr).astype(int)}
                               ).groupby("subject").mean() * 100
            pp_rows.append(per.reset_index().assign(ensemble=ens, threshold=tname)
                           .rename(columns={"y": "ptf_annotated", "pred": "ptf_predicted"}))
        log(f"  {ens}: out-of-fold AUROC {rows[-4]['auroc']:.3f}, released threshold F1 "
            f"{thr_rows[-2]['f1_pos']:.3f}, ICC {thr_rows[-2]['ptf_icc']:.3f}")

    log("our own home arms on the same patients (context only, not a paired comparison)")
    ours = our_arms(OUR_RUNS, OUR_ARMS, set(subjects), counts, log)
    ours += [{**r, "arm": r["arm"] + " (equal effort)"} for r in our_arms(TUNED_RUNS, TUNED_ARMS, set(subjects),
                                                                         counts, log)]
    summary = pd.concat([pd.DataFrame(rows), pd.DataFrame(ours)], ignore_index=True)
    summary.to_csv(out / "summary.csv", index=False, float_format="%.6g")
    pd.DataFrame(thr_rows).to_csv(out / "threshold_metrics.csv", index=False, float_format="%.6g")
    pd.concat(pp_rows, ignore_index=True).to_csv(out / "per_patient.csv", index=False, float_format="%.6g")
    folds_tbl.to_csv(out / "folds.csv", index=False)
    manifest_heads().to_csv(out / "released_heads.csv", index=False)
    (out / "overlap.json").write_text(json.dumps(overlap, indent=1) + "\n")

    per_head = (test.groupby(["fold", "seed"])
                .apply(lambda g: pd.Series({"n_patients": g.subject.nunique(), "n_windows": len(g),
                                            "prevalence": g.y.mean(),
                                            "auroc": C.auroc(g.softmax.to_numpy(), g.y.to_numpy()),
                                            "auprc": C.auprc(g.softmax.to_numpy(), g.y.to_numpy())}),
                       include_groups=False).reset_index())
    per_head.to_csv(out / "per_fold.csv", index=False, float_format="%.6g")

    prim = summary[summary.primary.fillna(False).astype(bool)]
    th = pd.DataFrame(thr_rows)
    t42 = test[test.seed == 42]
    prev_major = t42.groupby("fold").y.mean().to_dict()
    prev_any = t42.groupby("fold").any_fog.mean().to_dict()
    report = f"""# The released FORGE detector on our home cohort (DeFOG, {len(subjects)} patients)

Written by `scripts/forge_detector.py` on {time.strftime('%Y-%m-%d')} at commit {commit};
{a.n_boot:,} patient resamples; device {a.device}; {time.time() - t_start:.0f} s.

## Fold alignment

The nine released heads (`mc_probe`, seeds 42, 43, 44 x folds 0, 1, 2) each name their split in their safetensors
metadata and in `release/manifest.yaml`; the split files list participant codes. The three folds are a
{' / '.join(str(len(splits[f]['test'])) for f in FOLDS)} participant partition of exactly our {len(subjects)} home
patients, train, val and test disjoint within each fold. Each patient is scored only by the heads of the fold that
holds it in `test`, giving one out-of-fold score per window for all {len(subjects)} patients, which is the authors'
own DeFOG protocol. Patient by patient in `folds.csv`.

## Pretraining overlap

FORGE's self-supervised pretraining split (`configs/data/splits/kaggle_daily/ssl_pretraining.yaml`) lists
{overlap['forge_pretraining_participants']} participant codes; as a set they are
{'identical to' if overlap['identical_sets'] else 'not identical to'} the
{overlap['contest_daily_living_participants']} participants of the contest's daily-living release. So the encoder
was pretrained on the unlabeled at-home recordings of {overlap['home_patients_in_pretraining']} of our
{len(subjects)} home patients, including {sum(overlap['test_patients_in_pretraining_per_fold'].values())} of the
{len(subjects)} out-of-fold test patients
({', '.join(f'fold {f}: {overlap["test_patients_in_pretraining_per_fold"][str(f)]} of 19' for f in FOLDS)}).
The labels of the test patients were never used, but their signals were, which this project's own rules forbid
inside a fold. The comparison below therefore favours FORGE.

## Out-of-fold discrimination, authors' reading and our window labels

{C.md_table(summary[["source", "arm", "ensemble", "reading", "labels", "auroc", "auroc_lo", "auroc_hi", "auprc", "auprc_lo", "auprc_hi"]].fillna(""))}

The arms marked `ours` come from this project's registered home runs (five repetitions, all training patients,
out of fold) and are printed for context only. They are **not** a paired comparison: FORGE's folds are its own,
its heads are trained on 19 patients each against about 45 for ours, and its encoder saw most of these patients
during pretraining. No paired test is reported.

These numbers are not the model card's in-distribution reference of AP 0.730 on the DeFOG validation folds, and
are not meant to be. The windows differ: FORGE cuts 5 s patches every 2 s and every 1 s inside FOG regions and
calls a patch positive if any sample is FOG, which gives the 24.8% FOG share its own fold 0 split file records,
while our windows run every 2.5 s with a majority label and are kept only when every sample is valid
({', '.join(f'fold {f}: {prev_major[f]:.1%} majority, {prev_any[f]:.1%} any FOG' for f in FOLDS)}). Average
precision rises with the positive share, which is most of the difference; the card's figure is also on validation
patients rather than the out-of-fold test patients used here.

## At an operating point

{C.md_table(th[["ensemble", "threshold", "threshold_value", "f1_pos", "f1_macro", "sensitivity", "specificity", "ptf_icc", "ptf_mae_pp", "ptf_bias_pp"]])}

Percent time frozen is the share of a patient's analysed windows classified FOG, against the annotated share;
ICC is ICC(A,1), MAE and bias in percentage points. Intervals: `threshold_metrics.csv`.

## What this gives the paper

A published, FOG-specific detector with released weights, trained by its own authors on one of our own cohorts and
run here under its own fold structure, lands in the same band as our arms on window discrimination while
overestimating each patient's freezing burden by {th[(th.ensemble == 'mean3') & (th.threshold == 'released_0.35')].ptf_bias_pp.iloc[0]:.0f}
percentage points at its released operating point. The state-of-the-art gap an external review raised is therefore
answerable, and the answer points the same way as the paper's own finding: window-level discrimination near 0.84
AUROC does not deliver reliable measurement of freezing at home.

## Files

`summary.csv` (every scoring variant), `threshold_metrics.csv`, `per_fold.csv` (one row per head),
`per_patient.csv`, `per_window_scores.parquet` (every head's score for its fold's test and validation patients),
`folds.csv`, `released_heads.csv`, `overlap.json`.
"""
    (out / "report.md").write_text(report)
    (out / "summary.json").write_text(json.dumps(
        {"commit": commit, "date": time.strftime("%Y-%m-%d"), "device": a.device, "n_boot": a.n_boot,
         "seconds": round(time.time() - t_start, 1), "n_patients": len(subjects),
         "n_windows": int(home.sum()), "released_threshold": RELEASED_THRESHOLD,
         "heads": [head_name(f, s) for f in FOLDS for s in SEEDS], "overlap": overlap,
         "primary": prim.to_dict("records")}, indent=1, default=str) + "\n")
    log(report)
    log(f"DONE in {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
