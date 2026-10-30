"""Post hoc analysis h: percent time frozen as elapsed time, not as a share of windows.

Analysis b (scripts/paper_ptf_agreement.py) measures a patient's percent time frozen as the share of the patient's
analyzed windows that are FOG. With 5 s windows at a 2.5 s hop and majority labels, that is a biased proxy for the
share of elapsed time spent freezing: overlapping windows double-count the time they share, a window counts whole or
not at all, and an episode shorter than 2.5 s cannot make a window FOG by itself. This analysis measures the endpoint
in time instead, so the clinical quantity the paper reports is the one a clinician would mean by it.

For each patient:

    analyzed time   the union of [t0, t0 + 5) over the patient's windows, clipped to each recording's duration
    annotated time  the union of the patient's annotated FOG episodes, intersected with the analyzed time
    predicted time  the union of [t0, t0 + 5) over the windows the detector called FOG, which is a subset of the
                    analyzed time, so no further intersection is needed

and percent time frozen is annotated or predicted time over analyzed time. Unions are computed exactly by merging
intervals, never on a sampling grid. The analyzed time does not depend on the approach or the label budget, because
every window of a test patient is scored in every arm, so the annotated value is computed once per patient.

The statistics, repetitions, bootstrap and families are exactly those of analysis b, and `stats` is imported from it
rather than copied, so the two differ only in how percent time frozen is defined. Scores come from the equal-effort
runs, which store each window's index; their six base arms are bit-identical to the registered RQ1/RQ2 runs (checked
by `verify_base_arms`), so these are the registered arms' own predictions.

  agreement_time.csv   cohort, budget, approach, statistic: mean over repetitions, 95% interval, between-repetition SD
  paired_time.csv      differences between approaches for each statistic, two-sided bootstrap P, Holm per statistic
  per_patient_time.csv annotated and predicted percent time frozen per patient, time-based and window-based
  window_vs_time.csv   the two definitions side by side per cohort, budget and arm
  summary.json, report.md

Usage: .venv/bin/python scripts/paper_ptf_time.py [--runs RUN ...] [--n-boot N]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import paper_common as C
from paper_ptf_agreement import STATS, stats

NAME = "ptf_time"
WIN_S = 5.0  # window length; the hop is 2.5 s, so consecutive windows overlap by half
# the equal-effort runs store the window index and reproduce the registered RQ1/RQ2 arms exactly
DEFAULT_RUNS = [f"results/rq12_tuned_ft_seed{s}" for s in range(5)]
BASE_ARMS = ["harnet5", "feat60_fi", "feat60", "ssl_indomain", "supervised", "harnet5_rand"]


def union_length(starts: np.ndarray, ends: np.ndarray) -> float:
    """Total length covered by the intervals, counting overlap once."""
    if len(starts) == 0:
        return 0.0
    o = np.argsort(starts, kind="mergesort")
    s, e = starts[o], ends[o]
    keep_s, keep_e, cur_s, cur_e = [], [], s[0], e[0]
    for a, b in zip(s[1:], e[1:]):
        if a > cur_e:                      # a gap: close the run and start a new one
            keep_s.append(cur_s); keep_e.append(cur_e); cur_s, cur_e = a, b
        else:
            cur_e = max(cur_e, b)
    keep_s.append(cur_s); keep_e.append(cur_e)
    return float(np.sum(np.asarray(keep_e) - np.asarray(keep_s)))


def intersect_length(a_s: np.ndarray, a_e: np.ndarray, b_s: np.ndarray, b_e: np.ndarray) -> float:
    """Length of the intersection of two interval sets, each of which may itself overlap."""
    if len(a_s) == 0 or len(b_s) == 0:
        return 0.0
    # merge each set first, then sweep the two merged sets together
    def merged(s, e):
        o = np.argsort(s, kind="mergesort")
        s, e = s[o], e[o]
        out_s, out_e, cs, ce = [], [], s[0], e[0]
        for x, y in zip(s[1:], e[1:]):
            if x > ce:
                out_s.append(cs); out_e.append(ce); cs, ce = x, y
            else:
                ce = max(ce, y)
        out_s.append(cs); out_e.append(ce)
        return np.asarray(out_s), np.asarray(out_e)

    as_, ae_ = merged(a_s, a_e)
    bs_, be_ = merged(b_s, b_e)
    i = j = 0
    total = 0.0
    while i < len(as_) and j < len(bs_):
        lo, hi = max(as_[i], bs_[j]), min(ae_[i], be_[j])
        if hi > lo:
            total += hi - lo
        if ae_[i] < be_[j]:
            i += 1
        else:
            j += 1
    return float(total)


def window_intervals(t0: np.ndarray, duration: float) -> tuple[np.ndarray, np.ndarray]:
    """[t0, t0 + 5) per window, with the final window clipped to the recording's duration."""
    return t0, np.minimum(t0 + WIN_S, duration)


def patient_time(ann_windows: pd.DataFrame, episodes: pd.DataFrame,
                 recordings: pd.DataFrame) -> pd.DataFrame:
    """Analyzed and annotated FOG time per patient, from the annotation tables alone."""
    dur = recordings.set_index("rec").duration_s.to_dict()
    ep_by_rec = {r: (d.start_s.to_numpy(), d.end_s.to_numpy()) for r, d in episodes.groupby("rec")}
    rows = []
    for (cohort, subject), d in ann_windows.groupby(["cohort", "subject"], sort=True):
        analyzed = fog = 0.0
        for rec, dr in d.groupby("rec"):
            ws, we = window_intervals(dr.t0.to_numpy(), dur[rec])
            analyzed += union_length(ws, we)
            if rec in ep_by_rec:
                es, ee = ep_by_rec[rec]
                fog += intersect_length(ws, we, es, ee)
        rows.append({"cohort": cohort, "subject": subject, "analyzed_s": analyzed, "annotated_fog_s": fog,
                     "ptf_true_time": 100 * fog / analyzed if analyzed else np.nan})
    return pd.DataFrame(rows)


def window_lookup(ann: pd.DataFrame, recordings: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Start time, recording code and recording duration of every window, indexed by window-cache index."""
    t0_of = np.full(len(ann), np.nan)
    t0_of[ann.win.to_numpy()] = ann.t0.to_numpy()
    rec_codes, _ = pd.factorize(ann.rec)
    rec_of = np.full(len(ann), -1, dtype=np.int64)
    rec_of[ann.win.to_numpy()] = rec_codes
    dur_of = np.full(len(ann), np.nan)
    dur_of[ann.win.to_numpy()] = recordings.set_index("rec").duration_s.reindex(ann.rec).to_numpy()
    return t0_of, rec_of, dur_of


def predicted_time(d: pd.DataFrame, t0_of: np.ndarray, rec_of: np.ndarray, dur_of: np.ndarray) -> float:
    """Elapsed time covered by the windows this arm called FOG, for one patient."""
    w = d.window.to_numpy()
    sel = w[d.pred.to_numpy().astype(bool)]
    if len(sel) == 0:
        return 0.0
    total = 0.0
    for rec in np.unique(rec_of[sel]):
        m = sel[rec_of[sel] == rec]
        ws, we = window_intervals(t0_of[m], dur_of[m][0])
        total += union_length(ws, we)
    return total


def verify_base_arms(runs: list[str], log) -> None:
    """The equal-effort runs must reproduce the registered RQ1/RQ2 arms exactly, or these are not those arms."""
    for r in runs:
        seed = r.rstrip("/").split("seed")[-1]
        reg = C.ROOT / f"results/rq12_fog_transfer_mps_fixed_seed{seed}/out_of_fold_scores.parquet"
        if not reg.exists():
            log(f"  seed {seed}: no registered run to check against ({reg.name} missing)")
            continue
        cols = ["cohort", "fold", "budget", "arm", "subject", "y", "score"]
        a = pd.read_parquet(reg, columns=cols)
        b = pd.read_parquet(C.ROOT / r / "out_of_fold_scores.parquet", columns=cols)
        for arm in sorted(set(a.arm) & set(b.arm)):
            k = ["cohort", "fold", "budget", "arm", "subject", "y", "score"]
            x = a[a.arm == arm].sort_values(k).reset_index(drop=True)
            y = b[b.arm == arm].sort_values(k).reset_index(drop=True)
            if len(x) != len(y) or not np.array_equal(x.y.to_numpy(), y.y.to_numpy()) \
                    or not np.array_equal(x.score.to_numpy(), y.score.to_numpy()):
                raise SystemExit(f"seed {seed}, arm {arm}: the equal-effort run does not reproduce the registered "
                                 f"RQ1/RQ2 scores, so its windows cannot stand in for them")
        log(f"  seed {seed}: base arms identical to the registered run")


def main():
    ap = C.common_args(argparse.ArgumentParser())
    ap.add_argument("--n-boot", type=int, default=C.N_BOOT)
    a = ap.parse_args()
    if a.runs == C.FIXED_RUNS:   # the registered RQ1/RQ2 runs store no window index; the equal-effort runs do
        a.runs = DEFAULT_RUNS
    t_start = time.time()

    def log(msg):
        print(msg, flush=True)

    ann = pd.read_parquet(C.DERIVED / "annotations" / "windows.parquet")
    episodes = pd.read_csv(C.DERIVED / "annotations" / "episodes.csv")
    recordings = pd.read_csv(C.DERIVED / "annotations" / "recordings.csv")

    log("checking that the scored runs reproduce the registered arms")
    verify_base_arms(a.runs, log)

    pt = patient_time(ann, episodes, recordings)
    # Cross-check: the FOG seconds derived here from the episode intervals should not exceed the dataset's own
    # per-recording FOG seconds, since the analyzed span is a subset of each recording.
    own = recordings.groupby("subject").fog_valid_s.sum().reindex(pt.subject).to_numpy()
    excess = pt.annotated_fog_s.to_numpy() - own
    log(f"annotated FOG seconds against the recordings table: largest excess {np.nanmax(excess):.2f} s, "
        f"largest shortfall {-np.nanmin(excess):.2f} s over {len(pt)} patients "
        f"(a shortfall is expected where windows do not cover a whole recording)")

    t0_of, rec_of, dur_of = window_lookup(ann, recordings)

    by_seed, labels, cfgs = C.load_runs(a.runs, a.allow_superseded,
                                        columns=["cohort", "fold", "budget", "arm", "subject", "y", "pred", "window"])
    seeds = sorted(by_seed)
    point, boot, pp_rows, npat = {}, {}, [], {}
    for cohort in C.COHORTS:
        pts, idx, truth = None, None, None
        for s in seeds:
            dc = by_seed[s][(by_seed[s].cohort == cohort) & (by_seed[s].arm.isin(BASE_ARMS))]
            p_s = np.array(sorted(dc.subject.unique()))
            if pts is None:
                pts = p_s
                counts = C.resamples(len(pts), a.n_boot, seed=0)
                idx = np.stack([np.repeat(np.arange(len(pts)), c.astype(int)) for c in counts])
                npat[cohort] = len(pts)
                tt = pt[pt.cohort == cohort].set_index("subject").reindex(pts)
                truth = tt.ptf_true_time.to_numpy()
                analyzed = tt.analyzed_s.to_numpy()
            elif not np.array_equal(pts, p_s):
                raise SystemExit(f"seed {s}: patients of {cohort} differ")
            for (budget, arm), d in dc.groupby(["budget", "arm"]):
                per_sub = {p: predicted_time(ds, t0_of, rec_of, dur_of) for p, ds in d.groupby("subject")}
                pred = np.array([100 * per_sub.get(p, 0.0) / an for p, an in zip(pts, analyzed)])
                win_based = d.groupby("subject")[["y", "pred"]].mean().reindex(pts) * 100
                st, bt = stats(truth[None, :], pred[None, :]), stats(truth[idx], pred[idx])
                for k in STATS:
                    point[k, s, cohort, budget, arm] = float(st[k][0])
                    boot[k, s, cohort, budget, arm] = bt[k]
                pp_rows.append(pd.DataFrame({"cohort": cohort, "budget": budget, "arm": arm, "seed": s,
                                             "subject": pts, "ptf_true_time": truth, "ptf_pred_time": pred,
                                             "analyzed_s": analyzed,
                                             "ptf_true_window": win_based.y.to_numpy(),
                                             "ptf_pred_window": win_based.pred.to_numpy()}))
            log(f"{cohort} seed {s} done ({time.time() - t_start:.0f} s)")

    def mp(k, c, b, arm):
        return float(np.mean([point[k, s, c, b, arm] for s in seeds]))

    def mb(k, c, b, arm):
        return np.mean([boot[k, s, c, b, arm] for s in seeds], axis=0)

    cells = sorted({key[2:] for key in point})
    rows = []
    for c, b, arm in cells:
        for k in STATS:
            lo, hi = C.interval(mb(k, c, b, arm))
            vals = [point[k, s, c, b, arm] for s in seeds]
            rows.append({"cohort": c, "budget": b, "arm": arm, "statistic": k, "value": mp(k, c, b, arm), "lo": lo,
                         "hi": hi, "seed_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
                         "n_patients": npat[c], "n_seeds": len(seeds)})
    agreement = pd.DataFrame(rows)

    pp = pd.concat(pp_rows, ignore_index=True)
    per_patient = (pp.groupby(["cohort", "budget", "arm", "subject"])
                   .agg(ptf_true_time=("ptf_true_time", "mean"), ptf_pred_time=("ptf_pred_time", "mean"),
                        ptf_pred_time_sd=("ptf_pred_time", "std"), analyzed_s=("analyzed_s", "mean"),
                        ptf_true_window=("ptf_true_window", "mean"), ptf_pred_window=("ptf_pred_window", "mean"),
                        n_seeds=("seed", "nunique")).reset_index())

    # the two definitions side by side, averaged over patients and repetitions
    wv = (pp.groupby(["cohort", "budget", "arm"])
          [["ptf_true_time", "ptf_pred_time", "ptf_true_window", "ptf_pred_window"]].mean().reset_index())
    wv["bias_time"] = wv.ptf_pred_time - wv.ptf_true_time
    wv["bias_window"] = wv.ptf_pred_window - wv.ptf_true_window

    prs = []
    for k in STATS:
        part = []
        for c in C.COHORTS:
            for A, B in C.COMPARISONS:
                for b in C.BUDGETS:
                    if (c, b, A) not in set(cells) or (c, b, B) not in set(cells):
                        continue
                    d_boot = mb(k, c, b, A) - mb(k, c, b, B)
                    part.append({"cohort": c, "budget": b, "comparison": f"{A} - {B}", "statistic": k,
                                 "d_mean": mp(k, c, b, A) - mp(k, c, b, B),
                                 "ci_lo": C.interval(d_boot)[0], "ci_hi": C.interval(d_boot)[1],
                                 "p": C.two_sided_p(d_boot), "n_patients": npat[c], "n_seeds": len(seeds)})
        pf = pd.DataFrame(part)
        if not pf.empty:
            pf["p_holm"] = C.holm(pf.p.to_numpy())
        prs.append(pf)
    paired = pd.concat(prs, ignore_index=True)

    summary = {"commit": C.git_commit(), "date": time.strftime("%Y-%m-%d"), "labels": labels,
               "runs": {str(s): cfgs[s]["run"] for s in seeds},
               "run_commits": {str(s): cfgs[s]["commit"] for s in seeds},
               "n_boot": a.n_boot, "seeds": seeds, "arms": BASE_ARMS, "n_patients": npat,
               "n_tests_per_statistic": int(len(paired) / max(len(STATS), 1)),
               "definition": "elapsed annotated or predicted FOG time over analyzed time, intervals merged exactly",
               "seconds": round(time.time() - t_start, 1)}
    show = agreement[agreement.budget == "all"].pivot_table(index=["cohort", "arm"], columns="statistic",
                                                           values="value").round(3)
    report = f"""# Percent time frozen in elapsed time (analysis h, post hoc)

Written by `scripts/paper_ptf_time.py` on {summary['date']} at commit {summary['commit']}; labels: {labels};
{len(seeds)} repetitions; {a.n_boot:,} patient resamples per cohort. Percent time frozen is elapsed FOG time over
analyzed time, with overlapping windows and episodes merged exactly, in place of the share of windows used by
analysis b. ICC is ICC(A,1); MAE and bias in percentage points (bias = predicted minus annotated).

## All training patients, mean over repetitions

{C.md_table(show)}

## The two definitions side by side, all training patients

{C.md_table(wv[wv.budget == "all"].drop(columns="budget"), floatfmt=".2f")}

Intervals, other budgets and paired differences: `agreement_time.csv`, `paired_time.csv`.
"""
    out = C.write_outputs(NAME, summary, {"agreement_time.csv": agreement, "paired_time.csv": paired,
                                          "per_patient_time.csv": per_patient, "window_vs_time.csv": wv},
                          report, base=a.out_base)
    print(report)
    log(f"wrote {out} in {summary['seconds']} s")


if __name__ == "__main__":
    main()
