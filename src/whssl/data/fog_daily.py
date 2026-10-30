"""FOG daily-living recordings (Zenodo 10.5281/zenodo.12565292, Daily Living Data Part 1 + Part 2; CC BY 4.0).

65 Parkinson's patients, one recording each (about 8 days, 11,724 h in all), one lower-back sensor at 100 Hz,
AccV / AccML / AccAP in g (the DeFOG device and axes), unlabelled. daily_metadata.csv maps each file (Id) to a
Subject; 45 of the 65 subjects are DeFOG patients, so pretraining must exclude, inside each fold, the daily-living
data of that fold's DeFOG test patients.

For self-supervised pretraining: resampled to 30 Hz exactly as the DeFOG windows (polyphase), cut into
non-overlapping 5 s windows, kept when the acceleration magnitude varies (SD >= min_sd g, which removes still
periods such as sleep: about 55% of windows; 93% of DeFOG FOG windows pass 0.01 g, so freezing-like motion stays),
then at most max_windows per subject drawn at random (reproducible).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..signal import gait_like, resample

ROOT = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("WHSSL_DATA", ROOT / "data"))
RAW = DATA / "raw" / "fog_daily_living"
META = DATA / "raw" / "fog_contest" / "Competition Dataset" / "Competition Dataset" / "daily_metadata.csv"
PROC = DATA / "processed" / "fog_daily"
FS_IN = 100.0
# Walking-like windows (gait=True): magnitude SD >= 0.02 g and rhythmicity >= 0.33, validated on HAR70+ lower-back data
# (scripts/validate_gait_filter.py: held-out sensitivity 0.898, specificity 0.977 against static postures; about a
# third of the active daily-living windows pass).
GAIT_MIN_SD, GAIT_MIN_CONC = 0.02, 0.33


def recordings() -> pd.DataFrame:
    """One row per recording: Id, Subject, path (sorted by Subject)."""
    meta = pd.read_csv(META)
    paths = {p.stem: p for p in RAW.glob("*/*/*.parquet")}
    missing = sorted(set(meta.Id) - set(paths))
    if missing:
        raise FileNotFoundError(f"{len(missing)} daily-living files missing, e.g. {missing[0]} (run download_datasets.py --wave 2)")
    return meta.assign(path=meta.Id.map(paths)).sort_values("Subject").reset_index(drop=True)[["Id", "Subject", "path"]]


def window_recording(path: Path, fs_out: float = 30.0, win_s: float = 5.0, min_sd: float = 0.01,
                     max_windows: int = 10_000, seed: int = 0, gait: bool = False) -> np.ndarray:
    """(n, T, 3) float32 windows in g, axes V, ML, AP. gait=True keeps only walking-like active windows."""
    tab = pq.read_table(path, columns=["AccV", "AccML", "AccAP"])
    acc = np.stack([tab.column(c).to_numpy().astype(np.float32) for c in ["AccV", "AccML", "AccAP"]], axis=1)
    acc = resample(acc, FS_IN, fs_out, axis=0).astype(np.float32)
    T = int(round(win_s * fs_out))
    n = len(acc) // T
    W = acc[: n * T].reshape(n, T, 3)
    active = np.flatnonzero(np.linalg.norm(W, axis=2).std(axis=1) >= min_sd)
    if gait:
        active = active[gait_like(W[active], fs_out, GAIT_MIN_SD, GAIT_MIN_CONC)]
    if len(active) > max_windows:
        active = np.sort(np.random.default_rng(seed).choice(active, max_windows, replace=False))
    return W[active]


def build(fs_out: float = 30.0, win_s: float = 5.0, min_sd: float = 0.01, max_windows: int = 10_000,
          seed: int = 0, log=print, gait: bool = False) -> Path:
    out = PROC / f"windows_{int(fs_out)}hz_{win_s:g}s_sd{min_sd:g}_max{max_windows}_seed{seed}{'_gait' if gait else ''}.npz"
    if out.exists():
        return out
    X, subj, n_active = [], [], []
    for i, r in recordings().iterrows():
        w = window_recording(r.path, fs_out, win_s, min_sd, max_windows, seed=seed * 1000 + i, gait=gait)
        X.append(w)
        subj.append(np.repeat(r.Subject, len(w)))
        if log and (i + 1) % 10 == 0:
            log(f"  daily living: {i + 1} recordings windowed")
    PROC.mkdir(parents=True, exist_ok=True)
    np.savez(out, X=np.concatenate(X), subject=np.concatenate(subj).astype(str),
             config=np.array([fs_out, win_s, min_sd, max_windows, seed]))
    return out


def load(fs_out: float = 30.0, win_s: float = 5.0, min_sd: float = 0.01, max_windows: int = 10_000,
         seed: int = 0, gait: bool = False) -> Dict[str, np.ndarray]:
    z = np.load(build(fs_out, win_s, min_sd, max_windows, seed, gait=gait), allow_pickle=False)
    return {k: z[k] for k in z.files}
