"""Post hoc data preparation for the paper (read-only): recordings, annotated FOG episodes and window annotations.

One pass over the raw FOG contest recordings with the project loader (whssl.data.fog_contest.load_recording, which
labels the notype recordings from events.csv since commit 493fb42). For every recording it repeats the windowing of
window_recording (30 Hz, 5 s windows every 2.5 s, kept when every sample is valid, FOG when FOG covers at least half
of the window) and checks the kept windows one by one against the window cache (start time, label, FOG fraction).
Nothing is written outside results/paper_derived/annotations/.

Outputs, used by the Table 1, episode and FOG-manifestation analyses (scripts/paper_table1.py, paper_episodes.py,
paper_fog_types.py):
  recordings.csv       one row per recording: duration, time with valid labels, FOG time, candidate and kept windows
  episodes.csv         annotated FOG episodes at the native sampling rate: a run of consecutive FOG samples of one
                       class (start hesitation, turn, walking, or no class for the notype recordings); duration,
                       share inside valid time, kinetic flag from events.csv where the recording is listed there
  episode_windows.csv  every cached window that overlaps an episode, with the overlap in seconds
  windows.parquet      every cached window (global index into the cache): label, FOG fraction, share of its FOG
                       samples by class and by kinetic flag, dominant class
  summary.json, report.md

Usage: .venv/bin/python scripts/paper_annotations.py   (about two minutes on the CPU)
"""
from __future__ import annotations

import time
from fractions import Fraction

import numpy as np
import pandas as pd

import paper_common as C
from paper_common import F

NAME = "annotations"
TYPES = {1: "start_hesitation", 2: "turn", 3: "walking", -1: "no_class"}
T, H = int(round(C.WIN * C.FS)), int(round(C.HOP * C.FS))


def runs(code: np.ndarray):
    """(start, end) of maximal runs of one nonzero value."""
    change = np.flatnonzero(np.diff(np.r_[0, code, 0]) != 0)
    for a, b in zip(change[:-1], change[1:]):
        if code[a] != 0:
            yield a, b


def main():
    t_start = time.time()
    recs = F.recordings()
    ev = pd.read_csv(F.RAW / "events.csv")
    ev_by_rec = {k: g for k, g in ev.groupby("Id")}
    W = C.window_meta("fixed")
    cache_idx = pd.Series(np.arange(len(W["rec"]))).groupby(W["rec"]).apply(np.asarray).to_dict()

    rec_rows, ep_rows, epw_rows, win_parts = [], [], [], []
    for k, rec in recs.iterrows():
        r = F.load_recording(rec)
        fog, ftype, valid, fs = r["fog"].astype(int), r["fog_type"].astype(int), r["valid"], r["fs"]
        n = len(fog)
        kin = np.full(n, -1, np.int8)  # kinetic flag per sample from events.csv: 1 with movement, 0 akinetic
        e = ev_by_rec.get(rec.Id)
        if e is not None:
            for x in e.dropna(subset=["Init", "Completion"]).itertuples():
                if not pd.isna(x.Kinetic):
                    kin[max(int(round(x.Init * fs)), 0):min(int(round(x.Completion * fs)), n)] = int(x.Kinetic)
        # windowing, as whssl.data.fog_contest.window_recording
        q = Fraction(C.FS / fs).limit_denominator(1000)
        m = -(-n * q.numerator // q.denominator)  # resample_poly output length, ceil(n * up / down)
        src = np.minimum((np.arange(m) * fs / C.FS).round().astype(int), n - 1)
        starts = np.arange(0, m - T + 1, H)
        idx = starts[:, None] + np.arange(T)[None, :]
        keep = valid[src][idx].mean(axis=1) >= 1.0
        f30, t30, k30 = fog[src][idx], ftype[src][idx], kin[src][idx]
        frac = f30.mean(axis=1)
        gi = cache_idx.get(rec.Id, np.zeros(0, int))
        if len(gi) != keep.sum():
            raise SystemExit(f"{rec.Id}: {keep.sum()} kept windows, {len(gi)} in the cache")
        if len(gi):
            ok = (np.allclose(W["t0"][gi], starts[keep] / C.FS) and np.array_equal(W["y"][gi], (frac[keep] >= 0.5))
                  and np.allclose(W["frac"][gi], frac[keep]))
            if not ok:
                raise SystemExit(f"{rec.Id}: kept windows differ from the cache")
            nf = np.maximum(f30[keep].sum(axis=1), 1)
            shares = {f"share_{v}": ((t30[keep] == c) & (f30[keep] == 1)).sum(axis=1) / nf for c, v in TYPES.items()}
            kinetic = ((k30[keep] == 1) & (f30[keep] == 1)).sum(axis=1)
            akinetic = ((k30[keep] == 0) & (f30[keep] == 1)).sum(axis=1)
            sh = np.column_stack([shares[f"share_{v}"] for v in TYPES.values()])
            dom = np.where(frac[keep] > 0, np.array(list(TYPES.values()))[sh.argmax(axis=1)], "none")
            known = kinetic + akinetic
            win_parts.append(pd.DataFrame({
                "win": gi, "rec": rec.Id, "subject": rec.Subject, "cohort": rec.cohort, "t0": starts[keep] / C.FS,
                "y": (frac[keep] >= 0.5).astype(int), "frac": frac[keep], **shares,
                "kinetic_share": np.where(known > 0, kinetic / np.maximum(known, 1), np.nan),
                "kinetic_known_share": known / nf, "dominant_class": dom}))
        # annotated episodes at the native rate
        t0k = starts[keep] / C.FS
        n_ep = 0
        for a, b in runs(ftype * (fog > 0)):
            s0, s1 = a / fs, b / fs
            ks = kin[a:b][kin[a:b] >= 0]
            eid = f"{rec.Id}_{n_ep:03d}"
            ep_rows.append({"episode": eid, "rec": rec.Id, "subject": rec.Subject, "cohort": rec.cohort,
                            "split": rec.split, "medication": rec.Medication, "start_s": s0, "end_s": s1,
                            "duration_s": (b - a) / fs, "fog_class": TYPES[int(ftype[a])],
                            "valid_share": float(valid[a:b].mean()),
                            "kinetic": float(np.round(ks.mean())) if len(ks) else np.nan})
            lo, hi = np.searchsorted(t0k, s0 - C.WIN, side="right"), np.searchsorted(t0k, s1, side="left")
            for j in range(lo, hi):
                ov = min(t0k[j] + C.WIN, s1) - max(t0k[j], s0)
                if ov > 0:
                    epw_rows.append({"episode": eid, "win": int(gi[j]), "overlap_s": ov})
            n_ep += 1
        rec_rows.append({"rec": rec.Id, "subject": rec.Subject, "cohort": rec.cohort, "split": rec.split,
                         "medication": rec.Medication, "visit": rec.Visit, "fs": fs, "n_samples": n,
                         "duration_s": n / fs, "valid_s": valid.sum() / fs, "fog_s": fog.sum() / fs,
                         "fog_valid_s": (fog.astype(bool) & valid).sum() / fs, "candidate_windows": len(starts),
                         "kept_windows": int(keep.sum()), "fog_windows": int((frac[keep] >= 0.5).sum()),
                         "episodes": n_ep, "events_csv_rows": 0 if e is None else len(e)})
        if (k + 1) % 200 == 0:
            print(f"  {k + 1}/{len(recs)} recordings ({time.time() - t_start:.0f} s)", flush=True)

    rr, ep = pd.DataFrame(rec_rows), pd.DataFrame(ep_rows)
    epw, wn = pd.DataFrame(epw_rows), pd.concat(win_parts, ignore_index=True).sort_values("win")
    if not np.array_equal(wn.win.to_numpy(), np.arange(len(W["rec"]))):
        raise SystemExit("window table does not cover the cache exactly once")
    by = rr.groupby("cohort")
    summary = {
        "commit": C.git_commit(), "date": time.strftime("%Y-%m-%d"), "cache": str(C.CACHES["fixed"].name),
        "recordings": by.size().to_dict(), "recordings_without_window": by.kept_windows.apply(lambda s: int((s == 0).sum())).to_dict(),
        "windows_candidate": by.candidate_windows.sum().to_dict(), "windows_kept": by.kept_windows.sum().to_dict(),
        "windows_excluded": (by.candidate_windows.sum() - by.kept_windows.sum()).to_dict(),
        "hours_recorded": (by.duration_s.sum() / 3600).to_dict(), "hours_valid": (by.valid_s.sum() / 3600).to_dict(),
        "fog_min_valid": (by.fog_valid_s.sum() / 60).to_dict(), "episodes": ep.groupby("cohort").size().to_dict(),
        "episodes_fully_valid": ep[ep.valid_share == 1].groupby("cohort").size().to_dict(),
        "episodes_with_window": ep[ep.episode.isin(epw.episode)].groupby("cohort").size().to_dict(),
        "windows_checked_against_cache": int(len(wn)), "seconds": round(time.time() - t_start, 1),
    }
    lines = [f"| {c} | {summary['recordings'][c]:,} | {summary['hours_recorded'][c]:.1f} | {summary['hours_valid'][c]:.1f} | "
             f"{summary['windows_candidate'][c]:,} | {summary['windows_kept'][c]:,} | {summary['windows_excluded'][c]:,} | "
             f"{summary['episodes'][c]:,} | {summary['episodes_fully_valid'].get(c, 0):,} |" for c in sorted(summary["recordings"])]
    report = f"""# Recordings, annotated episodes and window annotations

Written by `scripts/paper_annotations.py` on {summary['date']} at commit {summary['commit']} (post hoc, read-only).
Every kept window was checked against `{summary['cache']}` (start time, label and FOG fraction):
{summary['windows_checked_against_cache']:,} windows, all identical.

| Cohort | Recordings | Hours recorded | Hours with valid labels | Candidate windows | Kept windows | Excluded windows | Annotated episodes | Episodes entirely in valid time |
|---|---|---|---|---|---|---|---|---|
{chr(10).join(lines)}

A window is excluded when any of its samples is invalid (home cohort: outside a protocol task or marked ambiguous
by the dataset's Valid flag; lab test recordings: samples the official solution marks Ignored). An episode is a run
of consecutive FOG samples of one class at the native sampling rate; the notype recordings have no class.
"""
    out = C.write_outputs(NAME, summary, {"recordings.csv": rr, "episodes.csv": ep, "episode_windows.csv": epw,
                                          "windows.parquet": wn}, report)
    print(report)
    print(f"wrote {out} in {summary['seconds']} s")


if __name__ == "__main__":
    main()
