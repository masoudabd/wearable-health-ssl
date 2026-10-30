"""FogAtHome recordings from the FORGE data release (Hugging Face Liornis/fog-dataset; downloaded to data/raw/forge
with Masoud's approval of 2026-10-04, decision 21). External test sets for detectors trained on the contest cohorts.

  fogathome              structured FOG-provoking home recordings: 12 participants, 97 sessions, about 36% FOG
  fogathome_dailyliving  naturalistic free living: 11 participants, 3,428 sessions, 301.8 h, about 1.1% FOG

Lower-back Axivity, 100 Hz, columns AccV, AccML, AccAP (g), StartHesitation, Turn, Walking (FOG classes), Valid. The
dataset's owners describe both as independent of the contest cohorts (cross-study external evaluation); participant
codes are pseudonymous and must not be used to re-identify anyone. Data terms: non-commercial research use under the
source studies' terms (paper/notes/forge_check_2026-10-04.md).

Axes: the vertical axis has the opposite sign of the contest cohorts (mean +0.92 g in every structured session, about
-0.95 g in the contest data); the anteroposterior axis has the home cohort's sign. axis_sign (default V flipped) maps
the recordings into the contest convention; the mediolateral sign cannot be told from gravity (sensitivity option).

Daily living: activity.parquet segments each session into activity codes. Codes 1 and 4 together cover 58.17 h, the
walking-and-standing domain (58.18 h) in which the FORGE card scores daily living, so they are taken as that domain
(inferred from the durations; the card does not name the codes). Each window carries its majority code.

Windows follow whssl.data.fog_contest exactly (resampling, 5 s windows every 2.5 s, FOG if at least half the window,
only fully valid windows).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .fog_contest import DATA, FOG_TYPES, _flag, window_recording

RAW = DATA / "raw" / "forge" / "dataset"
PROC = DATA / "processed" / "fogathome"
SPLITS = ("fogathome", "fogathome_dailyliving")
FS = 100.0
WALK_STAND_CODES = (1, 4)  # inferred: their durations sum to the FORGE walking-and-standing domain


def sessions(split: str) -> pd.DataFrame:
    meta = pd.read_csv(RAW / split / split / "metadata.csv")
    meta["path"] = [str(RAW / split / split / "sessions" / f"{i}.csv") for i in meta.Id]
    return meta


def _activity_frames(split: str) -> dict:
    if split != "fogathome_dailyliving":
        return {}
    a = pd.read_parquet(RAW / split / "activity.parquet")
    return {sid: g[["start_frame", "end_frame", "Activity"]].to_numpy() for sid, g in a.groupby("session_id")}


def build_windows(split: str, axis_sign=(-1.0, 1.0, 1.0), fs_out: float = 30.0, win_s: float = 5.0,
                  hop_s: float = 2.5, log=print) -> Path:
    """Window every session of a split (axes V, ML, AP multiplied by axis_sign) and cache one .npz."""
    tag = "".join("p" if s > 0 else "m" for s in axis_sign)
    out = PROC / f"{split}_{int(fs_out)}hz_{win_s:g}s_hop{hop_s:g}_axes{tag}.npz"
    if out.exists():
        return out
    meta = sessions(split)
    acts = _activity_frames(split)
    sign = np.asarray(axis_sign, np.float32)
    parts = {k: [] for k in ["X", "y", "frac", "t0", "rec", "subject", "activity"]}
    for i, r in meta.iterrows():
        d = pd.read_csv(r.path)
        acc = d[["AccV", "AccML", "AccAP"]].to_numpy(np.float32) * sign
        lab = d[FOG_TYPES].to_numpy(np.int8)
        rec = {"acc": acc, "fog": lab.max(axis=1), "valid": _flag(d["Valid"]), "fs": FS}
        w = window_recording(rec, fs_out, win_s, hop_s)
        n = len(w["y"])
        act = np.full(n, -1, np.int16)
        if r.Id in acts and n:
            frame_act = np.full(len(d), -1, np.int16)
            for s0, s1, code in acts[r.Id]:
                frame_act[int(s0):int(s1)] = code
            starts = np.round(w["t0"] * FS).astype(int)
            T = int(round(win_s * FS))
            for j, s0 in enumerate(starts):  # majority activity code over the window's frames
                v = frame_act[s0:s0 + T]
                v = v[v >= 0]
                if len(v):
                    act[j] = np.bincount(v).argmax()
        for k in ["X", "y", "frac", "t0"]:
            parts[k].append(w[k])
        parts["rec"].append(np.repeat(r.Id, n))
        parts["subject"].append(np.repeat(r.Subject, n))
        parts["activity"].append(act)
        if log and (i + 1) % 250 == 0:
            log(f"  {split}: windowed {i + 1}/{len(meta)} sessions")
    arrays = {k: np.concatenate(v) for k, v in parts.items()}
    for k in ["rec", "subject"]:
        arrays[k] = arrays[k].astype(str)
    arrays["axis_sign"] = sign
    PROC.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    return out


def fog_intervals(split: str, log=print) -> tuple[pd.DataFrame, pd.Series]:
    """Annotated FOG of every session at the native rate, for percent time frozen in elapsed time.

    Returns the intervals (rec, start_s, end_s: runs of consecutive samples with any FOG class, half-open, in seconds
    from the session start, as scripts/paper_annotations.py times the contest episodes) and each session's duration in
    seconds (index rec). Validity is not applied here: windows are only cut from fully valid stretches, so FOG time is
    later intersected with the windows. Both tables are cached in PROC.
    """
    f_iv, f_dur = PROC / f"{split}_fog_intervals.parquet", PROC / f"{split}_durations.parquet"
    if f_iv.exists() and f_dur.exists():
        return pd.read_parquet(f_iv), pd.read_parquet(f_dur).set_index("rec").duration_s
    meta = sessions(split)
    iv, dur = [], []
    for i, r in meta.iterrows():
        fog = pd.read_csv(r.path, usecols=FOG_TYPES, engine="pyarrow").to_numpy().max(axis=1) > 0
        edges = np.diff(np.r_[0, fog.astype(np.int8), 0])
        a, b = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        iv.append(pd.DataFrame({"rec": str(r.Id), "start_s": a / FS, "end_s": b / FS}))
        dur.append({"rec": str(r.Id), "duration_s": len(fog) / FS})
        if log and (i + 1) % 500 == 0:
            log(f"  {split}: FOG intervals of {i + 1}/{len(meta)} sessions")
    intervals, durations = pd.concat(iv, ignore_index=True), pd.DataFrame(dur)
    PROC.mkdir(parents=True, exist_ok=True)
    intervals.to_parquet(f_iv, index=False)
    durations.to_parquet(f_dur, index=False)
    return intervals, durations.set_index("rec").duration_s


def load_windows(split: str, axis_sign=(-1.0, 1.0, 1.0), **kw) -> dict:
    z = np.load(build_windows(split, axis_sign, **kw), allow_pickle=False)
    return {k: z[k] for k in z.files}
