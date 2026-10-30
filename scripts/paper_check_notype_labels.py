"""Data-integrity check (post hoc, read-only): the FOG labels of the 46 DeFOG "notype" recordings.

Found on 2026-10-03 while reading the dataset documentation for the paper. In the files under train/notype, the
Event column is written at the wrong time scale: a FOG event that events.csv places at t seconds is flagged at row
index t (that is t / 100 seconds at 100 Hz), not at row 100 t. whssl.data.fog_contest.load_recording reads the Event
column as the FOG label of these recordings, so the cached windows of the notype recordings carry almost no FOG
label although events.csv lists their FOG episodes. The typed training files (train/defog, train/tdcsfog) agree
with events.csv.

This script does not change any data, cache or result. It
  1. measures the misalignment in the raw files (Event column against events.csv read at 100 Hz and as row indices);
  2. counts the cached home-cohort windows that the problem touches and what their labels would be with the
     events.csv intervals (same rule as the loader: FOG when it covers at least half of the 5 s window);
  3. re-scores the stored out-of-fold scores of one run against the corrected test labels. The models were trained
     on the uncorrected labels, so step 3 only indicates the direction and size of the effect; the experiments must
     be rerun after the loader is fixed.

Usage: .venv/bin/python scripts/paper_check_notype_labels.py [--run results/rq12_fog_transfer_mps]
Writes results/paper_derived/notype_label_check/ (summary.json, raw_files.csv, per_patient.csv, rescored.csv,
report.md). About one minute on the CPU.
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
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from whssl.data import fog_contest as F  # noqa: E402
from whssl.eval.splits import subject_kfold  # noqa: E402

OUT = ROOT / "results" / "paper_derived" / "notype_label_check"
FS, WIN = 100.0, 5.0


def flag(col: pd.Series) -> np.ndarray:
    return col.astype(str).str.strip().str.lower().eq("true").to_numpy()


def raw_files(ev: pd.DataFrame) -> pd.DataFrame:
    """Per notype file: FOG samples in the Event column and in events.csv, and where the flagged samples fall."""
    rows = []
    for f in sorted((F.RAW / "train" / "notype").glob("*.csv")):
        d = pd.read_csv(f)
        n = len(d)
        col = d["Event"].to_numpy().astype(bool)
        valid = flag(d["Valid"]) & flag(d["Task"])
        at_100hz, as_index = np.zeros(n, bool), np.zeros(n, bool)
        for r in ev[ev.Id == f.stem].dropna(subset=["Init", "Completion"]).itertuples():
            at_100hz[max(int(round(r.Init * FS)), 0):min(int(round(r.Completion * FS)), n)] = True
            as_index[max(int(np.floor(r.Init)), 0):min(int(np.ceil(r.Completion)) + 1, n)] = True
        idx = np.flatnonzero(col)
        rows.append({"file": f.stem, "samples": n, "valid_samples": int(valid.sum()),
                     "event_column_fog": int(col.sum()), "event_column_fog_valid": int((col & valid).sum()),
                     "events_csv_fog": int(at_100hz.sum()), "events_csv_fog_valid": int((at_100hz & valid).sum()),
                     "flagged_inside_events_at_100hz": int((col & at_100hz).sum()),
                     "flagged_inside_events_as_row_index": int((col & as_index).sum()),
                     "last_flagged_row": int(idx.max()) if len(idx) else -1})
    return pd.DataFrame(rows)


def typed_agreement(ev: pd.DataFrame, n_files: int = 30) -> float:
    """Share of samples where the typed label columns of train/defog agree with events.csv at 100 Hz."""
    agree = total = 0
    for f in sorted((F.RAW / "train" / "defog").glob("*.csv"))[:n_files]:
        d = pd.read_csv(f)
        n = len(d)
        col = d[F.FOG_TYPES].to_numpy().max(axis=1).astype(bool)
        lab = np.zeros(n, bool)
        for r in ev[ev.Id == f.stem].dropna(subset=["Init", "Completion"]).itertuples():
            lab[max(int(round(r.Init * FS)), 0):min(int(round(r.Completion * FS)), n)] = True
        agree += int((col == lab).sum())
        total += n
    return agree / total


def corrected_window_labels(W: dict, home: np.ndarray, ev: pd.DataFrame) -> np.ndarray:
    """Window labels of the home cohort with the notype recordings labeled from events.csv."""
    y = W["y"][home].astype(int).copy()
    by_rec = {k: g[["Init", "Completion"]].dropna().to_numpy() for k, g in ev.groupby("Id")}
    rec, t0, split = W["rec"][home], W["t0"][home], W["split"][home]
    for i in np.flatnonzero(split == "notype"):
        iv = by_rec.get(rec[i])
        overlap = 0.0 if iv is None else float(
            np.clip(np.minimum(iv[:, 1], t0[i] + WIN) - np.maximum(iv[:, 0], t0[i]), 0, None).sum())
        y[i] = int(overlap / WIN >= 0.5)
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/rq12_fog_transfer_mps", help="run whose stored scores are re-scored")
    a = ap.parse_args()
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    ev = pd.read_csv(F.RAW / "events.csv")

    raw = raw_files(ev)
    raw.to_csv(OUT / "raw_files.csv", index=False)

    W = F.load_windows(30.0, 5.0, 2.5)
    home = np.flatnonzero(W["cohort"] == "defog")
    y_now = W["y"][home].astype(int)
    y_fix = corrected_window_labels(W, home, ev)
    subj, split = W["subject"][home], W["split"][home]
    nt = split == "notype"
    pp = pd.DataFrame({"subject": subj, "y": y_now, "y_fix": y_fix, "notype": nt}).groupby("subject").agg(
        windows=("y", "size"), notype_windows=("notype", "sum"), fog_windows_cached=("y", "sum"),
        fog_windows_corrected=("y_fix", "sum")).reset_index()
    pp["ptf_cached"] = 100 * pp.fog_windows_cached / pp.windows
    pp["ptf_corrected"] = 100 * pp.fog_windows_corrected / pp.windows
    pp.to_csv(OUT / "per_patient.csv", index=False)
    aff = pp[pp.notype_windows > 0]

    run = ROOT / a.run
    cfg = json.loads((run / "config.json").read_text())
    oof = pd.read_parquet(run / "out_of_fold_scores.parquet")
    oc = oof[oof.cohort == "defog"]
    order = np.concatenate([te for _, te in subject_kfold(y_now, subj, cfg["folds"], cfg["seed"])])
    rows = []
    for (budget, arm), d in oc.groupby(["budget", "arm"]):
        assert (d.subject.to_numpy() == subj[order]).all() and (d.y.to_numpy() == y_now[order]).all(), \
            "stored rows do not line up with the rebuilt window order"
        s = d.score.to_numpy()
        yo, yf, isnt = y_now[order], y_fix[order], nt[order]
        rows.append({"budget": budget, "arm": arm,
                     "auroc_cached_labels": roc_auc_score(yo, s), "auroc_corrected_labels": roc_auc_score(yf, s),
                     "auroc_typed_recordings_only": roc_auc_score(yo[~isnt], s[~isnt]),
                     "auprc_cached_labels": average_precision_score(yo, s),
                     "auprc_corrected_labels": average_precision_score(yf, s),
                     "mean_score_notype_fog": float(s[isnt & (yf == 1)].mean()),
                     "mean_score_notype_not_fog": float(s[isnt & (yf == 0)].mean())})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "rescored.csv", index=False)

    summary = {
        "date": time.strftime("%Y-%m-%d %H:%M"), "commit": commit, "rescored_run": a.run,
        "rescored_run_commit": cfg.get("commit"), "rescored_run_seed": cfg.get("seed"),
        "notype_files": int(len(raw)),
        "event_column_fog_min": float(raw.event_column_fog.sum() / FS / 60),
        "event_column_fog_valid_min": float(raw.event_column_fog_valid.sum() / FS / 60),
        "events_csv_fog_min": float(raw.events_csv_fog.sum() / FS / 60),
        "events_csv_fog_valid_min": float(raw.events_csv_fog_valid.sum() / FS / 60),
        "flagged_samples": int(raw.event_column_fog.sum()),
        "flagged_inside_events_at_100hz": int(raw.flagged_inside_events_at_100hz.sum()),
        "flagged_inside_events_as_row_index": int(raw.flagged_inside_events_as_row_index.sum()),
        "last_flagged_row_max": int(raw.last_flagged_row.max()),
        "typed_files_agreement_with_events_csv": float(typed_agreement(ev)),
        "home_windows": int(len(home)), "home_patients": int(pp.subject.nunique()),
        "notype_windows": int(nt.sum()), "notype_windows_pct_of_home": float(100 * nt.mean()),
        "notype_patients": int(len(aff)),
        "notype_fog_windows_cached": int(y_now[nt].sum()), "notype_fog_windows_corrected": int(y_fix[nt].sum()),
        "home_fog_windows_cached": int(y_now.sum()), "home_fog_windows_corrected": int(y_fix.sum()),
        "home_prevalence_cached_pct": float(100 * y_now.mean()),
        "home_prevalence_corrected_pct": float(100 * y_fix.mean()),
        "windows_relabeled": int((y_now != y_fix).sum()),
        "patients_no_fog_cached": int((pp.fog_windows_cached == 0).sum()),
        "patients_no_fog_corrected": int((pp.fog_windows_corrected == 0).sum()),
        "affected_patients_ptf_cached_mean_pct": float(aff.ptf_cached.mean()),
        "affected_patients_ptf_corrected_mean_pct": float(aff.ptf_corrected.mean()),
        "seconds": round(time.time() - t_start, 1),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")

    s = summary
    show = res[res.budget.isin(["all", "10"]) & res.arm.isin(["harnet5", "feat60_fi", "ssl_indomain", "supervised"])]
    table = "\n".join(
        f"| {r.budget} | {r.arm} | {r.auroc_cached_labels:.3f} | {r.auroc_corrected_labels:.3f} | "
        f"{r.auprc_cached_labels:.3f} | {r.auprc_corrected_labels:.3f} | {r.mean_score_notype_fog:.2f} | "
        f"{r.mean_score_notype_not_fog:.2f} |" for r in show.sort_values(["budget", "arm"]).itertuples())
    report = f"""# Check of the notype labels (home cohort)

Written by `scripts/paper_check_notype_labels.py` on {s['date']} at commit {s['commit']}. Read-only check; no data,
cache or result was changed.

## Raw files

- {s['notype_files']} files under `train/notype`. Their Event column flags {s['flagged_samples']:,} samples as FOG
  ({s['event_column_fog_min']:.2f} min; {s['event_column_fog_valid_min']:.2f} min where Valid and Task are true).
- `events.csv` lists {s['events_csv_fog_min']:.1f} min of FOG for the same files
  ({s['events_csv_fog_valid_min']:.1f} min where Valid and Task are true).
- Of the flagged samples, {s['flagged_inside_events_at_100hz']:,} fall inside the `events.csv` intervals read at
  100 Hz, and {s['flagged_inside_events_as_row_index']:,} fall inside the same intervals read as row indices. The
  last flagged row in any file is row {s['last_flagged_row_max']:,} ({s['last_flagged_row_max'] / FS:.0f} s).
  So the Event column holds the event times in seconds written as row numbers.
- In the typed files (`train/defog`, first 30), the label columns agree with `events.csv` at 100 Hz in
  {100 * s['typed_files_agreement_with_events_csv']:.2f}% of samples.

## Cached windows of the home cohort (5 s at 30 Hz, hop 2.5 s)

- {s['notype_windows']:,} of {s['home_windows']:,} home windows ({s['notype_windows_pct_of_home']:.1f}%) come from
  notype recordings, from {s['notype_patients']} of {s['home_patients']} patients.
- FOG windows among them: {s['notype_fog_windows_cached']:,} in the cache; {s['notype_fog_windows_corrected']:,} when
  labeled from `events.csv` with the loader's rule (FOG covers at least half of the window).
- Home cohort: {s['home_fog_windows_cached']:,} FOG windows ({s['home_prevalence_cached_pct']:.1f}%) in the cache;
  {s['home_fog_windows_corrected']:,} ({s['home_prevalence_corrected_pct']:.1f}%) after correction;
  {s['windows_relabeled']:,} windows change label.
- Patients without any FOG window: {s['patients_no_fog_cached']} in the cache, {s['patients_no_fog_corrected']}
  after correction. Mean annotated percent time frozen of the {s['notype_patients']} affected patients:
  {s['affected_patients_ptf_cached_mean_pct']:.1f}% in the cache, {s['affected_patients_ptf_corrected_mean_pct']:.1f}%
  after correction.

## Stored scores of `{s['rescored_run']}` (seed {s['rescored_run_seed']}, commit {s['rescored_run_commit']}) against corrected test labels

Indicative only: the models were trained on the uncorrected labels.

| Budget | Arm | AUROC, cached labels | AUROC, corrected labels | AUPRC, cached | AUPRC, corrected | Mean score, notype windows that are FOG | Mean score, notype windows that are not FOG |
|---|---|---|---|---|---|---|---|
{table}

## What it means

Every home-cohort result computed from the current window cache is provisional: the label budget experiments (RQ1
and RQ2), the severity analysis (RQ3) and the daily-living pretraining experiment (RQ4). The lab cohort is not
affected. Fix: label the notype recordings from `events.csv` (Init and Completion are in seconds), rebuild the
window cache, and rerun the home cohort. The cached features and embeddings do not depend on the labels.
"""
    (OUT / "report.md").write_text(report)
    print(report)
    print(f"wrote {OUT} in {summary['seconds']} s")


if __name__ == "__main__":
    main()
