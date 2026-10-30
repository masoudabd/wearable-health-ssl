"""Post hoc analysis c (writing brief 8.5): Table 1 descriptives and the counts of the participant flow (Fig. 7).

Reads the clinical table (subjects.csv), the window cache and results/paper_derived/annotations/ (run
scripts/paper_annotations.py first). Read-only; writes results/paper_derived/table1/.

Clinical scores are per patient: the mean over the patient's rows in subjects.csv (one row per visit for the home
cohort), as in the severity analysis; sex is the value in most rows. Values are reported as they are in the file;
the three inconsistencies the dataset notes list (a lab patient with an NFOG-Q of 0, a home patient aged 28, one
patient whose sex differs between visits) are counted in summary.json and left in.

Usage: .venv/bin/python scripts/paper_table1.py   (seconds)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import paper_common as C
from paper_ptf_time import intersect_length, union_length, window_intervals

NAME = "table1"
SCORES = {"age": "Age, years", "years_dx": "Years since diagnosis", "updrs3_on": "MDS-UPDRS Part III, ON",
          "updrs3_off": "MDS-UPDRS Part III, OFF", "nfogq": "NFOG-Q"}


def main():
    t_start = time.time()
    A = C.DERIVED / "annotations"
    rec = pd.read_csv(A / "recordings.csv")
    ep = pd.read_csv(A / "episodes.csv")
    ann = pd.read_parquet(A / "windows.parquet")
    dur = rec.set_index("rec").duration_s.to_dict()
    ep_iv = {r: (d.start_s.to_numpy(), d.end_s.to_numpy()) for r, d in ep.groupby("rec")}
    # elapsed analyzed and annotated FOG time per recording, as analysis h measures percent time frozen
    rt = []
    for r_id, d in ann.groupby("rec"):
        ws, we = window_intervals(d.t0.to_numpy(), dur[r_id])
        fog = intersect_length(ws, we, *ep_iv[r_id]) if r_id in ep_iv else 0.0
        rt.append({"rec": r_id, "analyzed_s": union_length(ws, we), "annotated_fog_s": fog})
    rt = pd.DataFrame(rt).merge(rec[["rec", "subject", "cohort", "medication"]], on="rec")
    W = C.window_meta("fixed")
    pt = C.patient_table()
    rows, summary = [], {"commit": C.git_commit(), "date": time.strftime("%Y-%m-%d"), "cohorts": {}}
    for cohort in C.COHORTS:
        m = W["cohort"] == cohort
        subj = np.unique(W["subject"][m])
        p = pt.loc[subj]
        per = pd.Series(W["y"][m]).groupby(W["subject"][m]).sum()
        r = rec[rec.cohort == cohort]
        e = ep[ep.cohort == cohort]
        ev = e[e.valid_share == 1]
        meds = pd.Series(W["medication"][m]).groupby(W["subject"][m]).agg(lambda s: set(s))
        s = {"n_patients": len(subj), "n_windows": int(m.sum()), "n_fog_windows": int(W["y"][m].sum()),
             "prevalence_pct": float(100 * W["y"][m].mean()), "n_patients_no_fog": int((per == 0).sum()),
             "n_recordings": len(r), "n_recordings_no_window": int((r.kept_windows == 0).sum()),
             "hours_recorded": float(r.duration_s.sum() / 3600), "hours_valid": float(r.valid_s.sum() / 3600),
             "fog_min_valid": float(r.fog_valid_s.sum() / 60),
             "n_windows_candidate": int(r.candidate_windows.sum()),
             "n_windows_excluded": int(r.candidate_windows.sum() - r.kept_windows.sum()),
             "n_episodes": len(e), "n_episodes_valid": len(ev),
             "episode_s_median": float(ev.duration_s.median()), "episode_s_q1": float(ev.duration_s.quantile(0.25)),
             "episode_s_q3": float(ev.duration_s.quantile(0.75)),
             "episodes_under_2.5s_pct": float(100 * (ev.duration_s < 2.5).mean()),
             "n_patients_both_medication": int(meds.apply(lambda x: {"on", "off"} <= x).sum()),
             "n_recordings_off": int((r.medication == "off").sum()),
             "female": int((p.sex == "F").sum()), "female_pct": float(100 * (p.sex == "F").mean())}
        # added 2026-10-29 (internal review v0.1, reviewer 2): what the window-based descriptors leave out
        s["episodes_under_2.5s_time_pct"] = float(100 * ev.duration_s[ev.duration_s < 2.5].sum() / ev.duration_s.sum())
        # no FOG window (majority labels) is not the same as no freezing: count those with none in the analyzed time
        s["n_patients_no_fog_analyzed"] = int((pd.Series(W["frac"][m]).groupby(W["subject"][m]).max() == 0).sum())
        s["n_patients_no_off"] = int(meds.apply(lambda x: "off" not in x).sum())
        t = rt[rt.cohort == cohort]
        for state in ("on", "off"):
            ts = t[t.medication == state]
            s[f"ptf_annotated_time_{state}_pct"] = float(100 * ts.annotated_fog_s.sum() / ts.analyzed_s.sum())
        s["analyzed_hours_elapsed"] = float(t.analyzed_s.sum() / 3600)
        s["recorded_time_excluded_pct"] = float(100 * (1 - t.analyzed_s.sum() / r.duration_s.sum()))
        s["annotated_fog_excluded_pct"] = float(100 * (1 - t.annotated_fog_s.sum() / e.duration_s.sum()))
        s["fog_min_all"] = float(e.duration_s.sum() / 60)           # every annotated episode, valid time or not
        s["fog_min_analyzed"] = float(t.annotated_fog_s.sum() / 60)   # inside the analyzed time
        per_min = t.groupby("subject").analyzed_s.sum() / 60
        s["analyzed_min_per_patient"] = {"median": float(per_min.median()), "q1": float(per_min.quantile(0.25)),
                                         "q3": float(per_min.quantile(0.75))}
        for k, label in SCORES.items():
            v = p[k].dropna()
            s[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "n": int(len(v)), "min": float(v.min()),
                    "max": float(v.max())}
            rows.append({"cohort": cohort, "characteristic": label, "key": k, "mean": v.mean(), "sd": v.std(ddof=1),
                         "n": len(v), "n_patients": len(subj), "min": v.min(), "max": v.max()})
        s["oddities"] = {"nfogq_zero": int((p.nfogq == 0).sum()), "age_under_40": int((p.age < 40).sum()),
                         "sex_differs_between_visits": int(p.sex_inconsistent.sum()),
                         "updrs3_off_missing": int(p.updrs3_off.isna().sum()),
                         "updrs3_on_missing": int(p.updrs3_on.isna().sum())}
        summary["cohorts"][cohort] = s
        for k in ["n_patients", "female", "n_recordings", "hours_valid", "n_windows", "n_fog_windows",
                  "n_patients_no_fog", "n_episodes_valid", "n_patients_both_medication"]:
            rows.append({"cohort": cohort, "characteristic": k, "key": k, "mean": s[k], "sd": np.nan, "n": np.nan,
                         "n_patients": len(subj), "min": np.nan, "max": np.nan})
    summary["seconds"] = round(time.time() - t_start, 1)

    def ms(c, k):
        x = summary["cohorts"][c][k]
        return f"{x['mean']:.1f} ({x['sd']:.1f}); n = {x['n']}"

    lab, home = "tdcsfog", "defog"
    L, H = summary["cohorts"][lab], summary["cohorts"][home]
    body = "\n".join([
        f"| Patients | {L['n_patients']} | {H['n_patients']} |",
        f"| Female, n (%) | {L['female']} ({L['female_pct']:.1f}) | {H['female']} ({H['female_pct']:.1f}) |",
        *[f"| {SCORES[k]}, mean (SD) | {ms(lab, k)} | {ms(home, k)} |" for k in SCORES],
        f"| Recordings | {L['n_recordings']:,} | {H['n_recordings']:,} |",
        f"| Hours recorded / with valid labels | {L['hours_recorded']:.1f} / {L['hours_valid']:.1f} | "
        f"{H['hours_recorded']:.1f} / {H['hours_valid']:.1f} |",
        f"| Candidate / excluded / analyzed windows | {L['n_windows_candidate']:,} / {L['n_windows_excluded']:,} / "
        f"{L['n_windows']:,} | {H['n_windows_candidate']:,} / {H['n_windows_excluded']:,} / {H['n_windows']:,} |",
        f"| FOG windows (%) | {L['n_fog_windows']:,} ({L['prevalence_pct']:.1f}) | {H['n_fog_windows']:,} "
        f"({H['prevalence_pct']:.1f}) |",
        f"| Patients without a FOG window | {L['n_patients_no_fog']} | {H['n_patients_no_fog']} |",
        f"| Episodes in valid time; median (IQR) duration, s | {L['n_episodes_valid']:,}; {L['episode_s_median']:.1f} "
        f"({L['episode_s_q1']:.1f} to {L['episode_s_q3']:.1f}) | {H['n_episodes_valid']:,}; {H['episode_s_median']:.1f} "
        f"({H['episode_s_q1']:.1f} to {H['episode_s_q3']:.1f}) |",
        f"| Patients recorded both ON and OFF | {L['n_patients_both_medication']} | {H['n_patients_both_medication']} |"])
    report = f"""# Table 1 descriptives and flow counts (analysis c, post hoc)

Written by `scripts/paper_table1.py` on {summary['date']} at commit {summary['commit']} from the fixed window cache
and `results/paper_derived/annotations/`.

| | Lab cohort (tDCS FOG) | Home cohort (DeFOG) |
|---|---|---|
{body}

Inconsistencies left in (per cohort, lab / home): NFOG-Q of 0: {L['oddities']['nfogq_zero']} / {H['oddities']['nfogq_zero']};
age under 40: {L['oddities']['age_under_40']} / {H['oddities']['age_under_40']}; sex differs between visits:
{L['oddities']['sex_differs_between_visits']} / {H['oddities']['sex_differs_between_visits']}. MDS-UPDRS Part III OFF missing:
{L['oddities']['updrs3_off_missing']} / {H['oddities']['updrs3_off_missing']}; ON missing: {L['oddities']['updrs3_on_missing']} /
{H['oddities']['updrs3_on_missing']}.
"""
    out = C.write_outputs(NAME, summary, {"table1.csv": pd.DataFrame(rows)}, report)
    print(report)
    print(f"wrote {out} in {summary['seconds']} s")


if __name__ == "__main__":
    main()
