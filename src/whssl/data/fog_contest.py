"""MJFF / Tel Aviv freezing-of-gait contest data, Zenodo copy (Salomon et al., Nat Commun 2024).

Cite: doi 10.5281/zenodo.12565292 (newest version; the Competition Dataset.zip in it is byte-identical,
md5 35512a51d7169706109bd490e4fab9af, to the one in record 10959560). Zenodo lists CC BY 4.0;
the bundled LICENSE.txt says Apache-2.0. Both permit research use with attribution.

Unlike the Kaggle release, this copy includes the public and private TEST recordings and the official
per-sample solution, so all 128 patients are labelled:
  tdcsfog  71 patients, lab protocol, 128 Hz, acceleration in m/s^2
  defog    57 patients, home protocol, 100 Hz, acceleration in g (includes the 'notype' recordings,
           which are labelled FOG / no FOG without a FOG type)
All recordings: one lower-back sensor, axes vertical (V), mediolateral (ML), anteroposterior (AP).
The two cohorts share no patients. Grouping by Subject removes the train/test overlap documented
in the archive's README.

Validity: defog and notype samples count only where Valid and Task are both true (the contest's
scoring rule); test samples marked 'Ignored' in the official solution are invalid.

Labels of the 46 'notype' recordings come from events.csv (Init and Completion in seconds), not from their Event
column: that column holds the event times in seconds written as row numbers, so read as a label it marks almost no
FOG (0.8 min against 80.9 min in events.csv). Found 2026-10-03 by scripts/paper_check_notype_labels.py; until the
fix (commit after a7c3226) the window cache labelled 1,172 home FOG windows non-FOG (home prevalence 10.0% instead of
13.7%). The rule matches the typed files, whose label columns agree with events.csv at 100 Hz in 99.99% of samples.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..signal import resample  # torch-free, so analysis scripts never need torch

ROOT = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("WHSSL_DATA", ROOT / "data"))
RAW = DATA / "raw" / "fog_contest" / "Competition Dataset" / "Competition Dataset"
PROC = DATA / "processed" / "fog_contest"

FS = {"tdcsfog": 128.0, "defog": 100.0}
G = 9.80665
FOG_TYPES = ["StartHesitation", "Turn", "Walking"]


def recordings() -> pd.DataFrame:
    """One row per recording: Id, Subject, Visit, Medication, Test, cohort, split, path, fs."""
    td = pd.read_csv(RAW / "tdcsfog_metadata.csv").assign(cohort="tdcsfog")
    de = pd.read_csv(RAW / "defog_metadata.csv").assign(cohort="defog", Test=np.nan)
    df = pd.concat([td, de], ignore_index=True)
    split, path = [], []
    for i, c in zip(df.Id, df.cohort):
        candidates = [("train", RAW / "train" / c / f"{i}.csv"), ("public", RAW / "public" / c / f"{i}.csv"),
                      ("private", RAW / "private" / c / f"{i}.csv")]
        if c == "defog":
            candidates.insert(1, ("notype", RAW / "train" / "notype" / f"{i}.csv"))
        hit = next(((s, p) for s, p in candidates if p.exists()), ("missing", None))
        split.append(hit[0])
        path.append(str(hit[1]) if hit[1] else None)
    df["split"], df["path"] = split, path
    df["fs"] = df.cohort.map(FS)
    return df[df.split != "missing"].reset_index(drop=True)


def subjects() -> pd.DataFrame:
    """Clinical table: Age, Sex, YearsSinceDx, UPDRSIII_On, UPDRSIII_Off, NFOGQ (no Hoehn and Yahr stage).
    Some subjects have two visits; Visit is NaN when only one row exists."""
    return pd.read_csv(RAW / "subjects.csv")


def _solution() -> pd.DataFrame:
    """Official test labels, cached as parquet (the CSV has 9.5 M rows)."""
    cache = PROC / "official_solution.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    s = pd.read_csv(RAW / "official solution.csv")
    parts = s.Id.str.rsplit("_", n=1)
    s["file"] = parts.str[0]
    s["Time"] = parts.str[1].astype(np.int64)
    s = s.drop(columns="Id")
    PROC.mkdir(parents=True, exist_ok=True)
    s.to_parquet(cache)
    return s


_SOLUTION_BY_FILE: Optional[Dict[str, pd.DataFrame]] = None
_EVENTS: Optional[pd.DataFrame] = None


def events_fog(rec_id: str, n: int, fs: float) -> np.ndarray:
    """Per-sample FOG (0/1) of one recording from events.csv: samples round(Init * fs) to round(Completion * fs)."""
    global _EVENTS
    if _EVENTS is None:
        _EVENTS = pd.read_csv(RAW / "events.csv")
    fog = np.zeros(n, np.int8)
    for r in _EVENTS[_EVENTS.Id == rec_id].dropna(subset=["Init", "Completion"]).itertuples():
        fog[max(int(round(r.Init * fs)), 0):min(int(round(r.Completion * fs)), n)] = 1
    return fog


def _flag(col: pd.Series) -> np.ndarray:
    """Parse true/false columns explicitly (astype(bool) would turn the text 'false' into True)."""
    return col.astype(str).str.strip().str.lower().eq("true").to_numpy()


def load_recording(rec: pd.Series) -> Dict[str, np.ndarray]:
    """Returns acc (N, 3) in g ordered V, ML, AP; fog (N,) 0/1; fog_type (N,) 0 none, 1 hesitation,
    2 turn, 3 walking, -1 unknown type; valid (N,) bool; fs."""
    global _SOLUTION_BY_FILE
    d = pd.read_csv(rec.path)
    acc = d[["AccV", "AccML", "AccAP"]].to_numpy(np.float32)
    if rec.cohort == "tdcsfog":
        acc = acc / G
    n = len(d)
    if rec.split in ("public", "private"):
        if _SOLUTION_BY_FILE is None:
            _SOLUTION_BY_FILE = {k: g for k, g in _solution().groupby("file")}
        s = _SOLUTION_BY_FILE[rec.Id].set_index("Time").reindex(d.Time)
        lab = s[FOG_TYPES].fillna(0).to_numpy(np.int8)
        valid = (s.Usage.fillna("Ignored") != "Ignored").to_numpy()
    elif rec.split == "notype":
        lab = None
        fog = events_fog(rec.Id, n, rec.fs)  # not the Event column: it holds seconds as row numbers (see top)
        valid = _flag(d["Valid"]) & _flag(d["Task"])
    else:
        lab = d[FOG_TYPES].to_numpy(np.int8)
        valid = _flag(d["Valid"]) & _flag(d["Task"]) if rec.cohort == "defog" else np.ones(n, bool)
    if lab is not None:
        fog = lab.max(axis=1)
        fog_type = np.where(fog > 0, lab.argmax(axis=1) + 1, 0).astype(np.int8)
    else:
        fog_type = np.where(fog > 0, -1, 0).astype(np.int8)
    return {"acc": acc, "fog": fog.astype(np.int8), "fog_type": fog_type, "valid": valid, "fs": float(rec.fs)}


def window_recording(r: Dict[str, np.ndarray], fs_out: float, win_s: float, hop_s: float,
                     min_valid: float = 1.0) -> Dict[str, np.ndarray]:
    """Resample to fs_out, cut fixed windows, keep windows whose valid fraction >= min_valid.
    Label = 1 if at least half of the window is FOG. Also returns the FOG fraction and window start (s)."""
    acc = resample(r["acc"], r["fs"], fs_out, axis=0).astype(np.float32)
    m = len(acc)
    src = np.minimum((np.arange(m) * r["fs"] / fs_out).round().astype(int), len(r["fog"]) - 1)
    fog, valid = r["fog"][src], r["valid"][src]
    T, hop = int(round(win_s * fs_out)), int(round(hop_s * fs_out))
    starts = np.arange(0, m - T + 1, hop)
    if len(starts) == 0:
        return {"X": np.zeros((0, T, 3), np.float32), "y": np.zeros(0, np.int8), "frac": np.zeros(0), "t0": np.zeros(0)}
    idx = starts[:, None] + np.arange(T)[None, :]
    vfrac = valid[idx].mean(axis=1)
    keep = vfrac >= min_valid
    frac = fog[idx].mean(axis=1)
    return {"X": acc[idx[keep]], "y": (frac[keep] >= 0.5).astype(np.int8), "frac": frac[keep],
            "t0": starts[keep] / fs_out}


def build_windows(fs_out: float = 30.0, win_s: float = 5.0, hop_s: float = 2.5, log=print) -> Path:
    """Window every recording and cache the result as one .npz with per-window metadata."""
    out = PROC / f"windows_{int(fs_out)}hz_{win_s:g}s_hop{hop_s:g}.npz"
    if out.exists():
        return out
    recs = recordings()
    parts = {k: [] for k in ["X", "y", "frac", "t0", "rec", "subject", "cohort", "split", "medication", "visit"]}
    for i, rec in recs.iterrows():
        w = window_recording(load_recording(rec), fs_out, win_s, hop_s)
        n = len(w["y"])
        for k in ["X", "y", "frac", "t0"]:
            parts[k].append(w[k])
        for k, v in [("rec", rec.Id), ("subject", rec.Subject), ("cohort", rec.cohort), ("split", rec.split),
                     ("medication", rec.Medication), ("visit", rec.Visit)]:
            parts[k].append(np.repeat(np.asarray(v, dtype=object), n))
        if log and (i + 1) % 100 == 0:
            log(f"  windowed {i + 1}/{len(recs)} recordings")
    arrays = {k: np.concatenate(v) for k, v in parts.items()}
    for k in ["rec", "subject", "cohort", "split", "medication"]:
        arrays[k] = arrays[k].astype(str)
    arrays["visit"] = arrays["visit"].astype(float)
    arrays["config"] = np.array([fs_out, win_s, hop_s])
    PROC.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    return out


def load_windows(fs_out: float = 30.0, win_s: float = 5.0, hop_s: float = 2.5) -> Dict[str, np.ndarray]:
    path = build_windows(fs_out, win_s, hop_s)
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}
