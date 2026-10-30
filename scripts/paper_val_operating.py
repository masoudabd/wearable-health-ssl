"""Post hoc (writing brief 8.5): operating points chosen on validation patients.

Analysis e (scripts/paper_episodes.py) could only use the stored threshold, chosen on validation patients for window
F1, because the first RQ1/RQ2 runs did not keep validation scores. The equal-effort runs (results/rq12_tuned_ft_seed*,
whose base approaches reproduce the fixed runs exactly) store them (validation_scores.parquet, with window indices).
Here every threshold is chosen on the validation patients of a fold and budget, from their stored scores only, and
then applied once to that fold's test patients:

  f1        the stored threshold (highest positive-class F1 on validation windows; analysis e's operating point)
  fa4, fa8, fa16   the lowest threshold whose false alarms per hour on the validation patients are at most 4, 8 or 16
                   (searched over 200 quantiles of the validation scores, 0.5 to 0.9995; if none qualifies, the
                   largest validation score, which predicts almost nothing)
  burden    the threshold at which the share of validation windows predicted FOG equals the share annotated
            (a burden-matched operating point for percent time frozen; the rule the external test also uses)
  burden_time  the threshold at which the validation patients' predicted FOG time equals their annotated FOG time,
            both as elapsed time over analyzed time pooled over the validation patients (analysis h's definition):
            the burden match made in the unit the paper reports. The predicted share falls as the threshold rises,
            so it is found by bisection over the validation scores, keeping the score whose share is closest

Episodes, false alarms and analyzed time follow analysis e exactly (its Cohort class). On the test patients, pooled
over folds: false alarms per analyzed hour, share of assessable episodes detected, share of predicted episodes that
touch FOG, and percent time frozen agreement per patient (mean signed error and mean absolute error in percentage
points, ICC(A,1)), and each rule's ICC minus the ICC at the stored F1 threshold (paired over the same resamples).

Percent time frozen is measured in elapsed time, as analysis h (scripts/paper_ptf_time.py) defines it, with its own
`patient_time` and `predicted_time`, so the F1 rule reproduces analysis h's values; this is checked whenever analysis
h's output is present in the same derived tree (points always, intervals when the resample counts agree). The share of
windows of analysis b is kept beside it (columns ptf_bias, ptf_icc, ptf_icc_minus_f1) for comparison with earlier
builds; the elapsed-time columns are ptf_time_*. Means over repetitions, with patient resamples per cohort shared by
repetitions, approaches and rules (percentile 95% intervals). Validation patients are few (2 with a budget of 10;
about a fifth of the training patients with all), which is part of what this shows.

Outputs (results/paper_derived/val_operating/): operating_points.csv (cohort, budget, approach, rule), thresholds.csv
(every fold's chosen threshold, its validation false-alarm rate and the validation burden in elapsed time),
per_patient.csv (annotated and predicted percent time frozen in elapsed time for every patient, approach and rule, mean
over repetitions; added 2026-10-29 for Fig. 3), summary.json, report.md.

Usage: .venv/bin/python scripts/paper_val_operating.py   (a few minutes on the CPU)
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

import paper_common as C
from paper_episodes import Cohort, ratio_boot
from paper_ptf_agreement import stats as ptf_stats
from paper_ptf_time import patient_time, predicted_time, window_lookup

NAME = "val_operating"
RUNS = [f"results/rq12_tuned_ft_seed{s}" for s in range(5)]
ARMS = ["harnet5", "feat60_fi", "ssl_indomain", "harnet5_tc", "harnet5_ft"]
BUDGETS = ["10", "all"]
FA_TARGETS = [4, 8, 16]
RULES = ["f1"] + [f"fa{t}" for t in FA_TARGETS] + ["burden", "burden_time"]
GRID = np.linspace(0.5, 0.9995, 200)


def fog_seconds(win: np.ndarray, pos: np.ndarray, lookup: tuple) -> float:
    """Elapsed time covered by the windows called FOG (any number of patients), by analysis h's predicted_time."""
    return predicted_time(pd.DataFrame({"window": win, "pred": pos}), *lookup)


def burden_time_threshold(win: np.ndarray, sc: np.ndarray, target: float, analyzed_s: float,
                          lookup: tuple) -> tuple[float, float]:
    """The validation score whose predicted FOG time share is closest to target; returns it and that share."""
    cand = np.unique(sc)  # ascending, so the predicted share is non-increasing in the index

    def share(k):
        return fog_seconds(win, sc >= cand[k], lookup) / analyzed_s

    lo, hi = 0, len(cand) - 1
    s_lo, s_hi = share(lo), share(hi)
    if s_hi > target:  # even the largest score predicts too much
        return float(cand[hi]), s_hi
    while hi - lo > 1:  # invariant: share(lo) > target >= share(hi)
        mid = (lo + hi) // 2
        s_mid = share(mid)
        if s_mid > target:
            lo, s_lo = mid, s_mid
        else:
            hi, s_hi = mid, s_mid
    return (float(cand[lo]), s_lo) if s_lo - target < target - s_hi else (float(cand[hi]), s_hi)


def choose(S: Cohort, v: pd.DataFrame, pt: pd.DataFrame, lookup: tuple) -> tuple[dict, dict]:
    """Thresholds of every rule from one fold's validation rows; also their validation false-alarm rates and the
    validation burden in elapsed time (annotated, and predicted at the burden_time threshold), in percent."""
    idx = np.searchsorted(S.win, v.win.to_numpy())
    sc = v.score.to_numpy()
    hours = S.hours[np.unique(S.col[idx])].sum()
    cand = np.unique(np.r_[np.quantile(sc, GRID), sc.max()])
    fa = np.empty(len(cand))
    for i, thr in enumerate(cand):
        pos = np.zeros(len(S.win), bool)
        pos[idx] = sc >= thr
        fa[i] = S.evaluate(pos)["fa"].sum() / hours
    thr = {"f1": float(v.threshold.iloc[0]), "burden": float(np.quantile(sc, 1 - v.y.mean()))}
    info = {}
    for t in FA_TARGETS:
        ok = np.flatnonzero(fa <= t)
        k = ok[0] if len(ok) else len(cand) - 1
        thr[f"fa{t}"], info[f"val_fa_per_hour_fa{t}"] = float(cand[k]), float(fa[k])
    p = pt.loc[v.subject.unique()]
    analyzed, target = p.analyzed_s.sum(), p.annotated_fog_s.sum() / p.analyzed_s.sum()
    thr["burden_time"], got = burden_time_threshold(v.win.to_numpy(), sc, target, analyzed, lookup)
    info.update({"val_ptf_true_time": 100 * target, "val_ptf_pred_time_burden_time": 100 * got})
    return thr, info


def patient_fog_seconds(S: Cohort, pos: np.ndarray, lookup: tuple) -> np.ndarray:
    """Predicted FOG time of every patient of the cohort (S.pts order), as analysis h computes it."""
    d = pd.DataFrame({"window": S.win, "pred": pos})
    out = np.array([predicted_time(g, *lookup) for _, g in d.groupby(S.col, sort=True)])
    if len(out) != len(S.pts):
        raise SystemExit("a patient of the cohort has no window")
    return out


def check_against_h(op: pd.DataFrame, n_boot: int, log) -> str:
    """The F1 rule is analysis h's operating point on the same runs, so its elapsed-time values must reproduce h's."""
    f, s = C.DERIVED / "ptf_time" / "agreement_time.csv", C.DERIVED / "ptf_time" / "summary.json"
    if not f.exists():
        msg = "analysis h output not found in this derived tree; the F1 rule was not cross-checked against it"
        log(msg)
        return msg
    h = pd.read_csv(f, dtype={"budget": str})
    same_boot = json.loads(s.read_text())["n_boot"] == n_boot
    n, worst = 0, 0.0
    for r in op[op.rule == "f1"].itertuples(index=False):
        for k in ("bias", "mae", "icc"):
            m = h[(h.cohort == r.cohort) & (h.budget == r.budget) & (h.arm == r.arm) & (h.statistic == k)]
            if m.empty:
                continue
            pairs = [("value", "")] + ([("lo", "_lo"), ("hi", "_hi")] if same_boot else [])
            for hc, suffix in pairs:
                a, b = float(m[hc].iloc[0]), float(getattr(r, f"ptf_time_{k}{suffix}"))
                if not np.isclose(a, b, rtol=1e-5, atol=1e-9):  # h's csv keeps six significant digits
                    raise SystemExit(f"F1 rule does not reproduce analysis h: {r.cohort} {r.budget} {r.arm} {k} "
                                     f"{hc}: {b} here, {a} in {f}")
                worst, n = max(worst, abs(a - b)), n + 1
    msg = (f"F1 rule reproduces analysis h in {n} values ({'points and intervals' if same_boot else 'points only'}; "
           f"largest difference {worst:.2g}, within the six significant digits h stores)")
    log(msg)
    return msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=RUNS)
    ap.add_argument("--n-boot", type=int, default=C.N_BOOT)
    a = ap.parse_args()
    t_start = time.time()

    def log(msg):
        print(msg, flush=True)

    A = C.DERIVED / "annotations"
    ep, epw = pd.read_csv(A / "episodes.csv"), pd.read_csv(A / "episode_windows.csv")
    ann, recordings = pd.read_parquet(A / "windows.parquet"), pd.read_csv(A / "recordings.csv")
    pt_all = patient_time(ann, ep, recordings)  # analyzed and annotated FOG seconds per patient, as analysis h
    lookup = window_lookup(ann, recordings)
    cols = ["cohort", "fold", "budget", "arm", "subject", "y", "score", "pred", "window"]
    by_seed, labels, cfgs = C.load_runs(a.runs, columns=cols)
    for s, o in by_seed.items():  # the stored window index must equal the rebuilt one
        if not np.array_equal(o.window.to_numpy(), o.win.to_numpy()):
            raise SystemExit(f"seed {s}: stored window indices differ from the rebuilt ones")
    vals = {cfgs[s]["seed"]: pd.read_parquet(C.ROOT / cfgs[s]["run"] / "validation_scores.parquet")
            .rename(columns={"window": "win"}) for s in by_seed}
    W = C.window_meta(labels)
    seeds = sorted(by_seed)
    rows, thr_rows, pp_rows, summ = [], [], [], {}
    for cohort in C.COHORTS:
        S = Cohort(cohort, W, ep, epw)
        pt = pt_all[pt_all.cohort == cohort].set_index("subject")
        analyzed = pt.analyzed_s.reindex(S.pts).to_numpy()
        ptf_true_time = pt.ptf_true_time.reindex(S.pts).to_numpy()
        counts = C.resamples(len(S.pts), a.n_boot, seed=0)
        boot_idx = np.stack([np.repeat(np.arange(len(S.pts)), c.astype(np.int64)) for c in counts])  # (resamples, patients)
        res = {}
        for s in seeds:
            o, v0 = by_seed[s], vals[s]
            for budget in BUDGETS:
                for arm in ARMS:
                    d = o[(o.cohort == cohort) & (o.budget == budget) & (o.arm == arm)]
                    vv = v0[(v0.cohort == cohort) & (v0.budget == budget) & (v0.arm == arm)]
                    pos = {r: np.zeros(len(S.win), bool) for r in RULES}
                    for fold, t in d.groupby("fold"):
                        thr, info = choose(S, vv[vv.fold == fold], pt, lookup)
                        tidx = np.searchsorted(S.win, t.win.to_numpy())
                        if not np.array_equal((t.score.to_numpy() >= thr["f1"]).astype(int), t.pred.to_numpy()):
                            raise SystemExit(f"{cohort} seed {s} {budget} {arm} fold {fold}: stored threshold mismatch")
                        for r in RULES:
                            pos[r][tidx] = t.score.to_numpy() >= thr[r]
                        thr_rows.append({"cohort": cohort, "seed": s, "budget": budget, "arm": arm, "fold": fold,
                                         **{f"thr_{r}": thr[r] for r in RULES}, **info,
                                         "val_patients": vv[vv.fold == fold].subject.nunique()})
                    y = np.zeros(len(S.win))
                    y[np.searchsorted(S.win, d.win.to_numpy())] = d.y.to_numpy()
                    n_w = np.bincount(S.col, minlength=len(S.pts))
                    ptf_true = 100 * np.bincount(S.col, y, minlength=len(S.pts)) / n_w
                    for r in RULES:
                        e = S.evaluate(pos[r])
                        e["ptf_pred"] = 100 * np.bincount(S.col, pos[r], minlength=len(S.pts)) / n_w
                        e["ptf_true"] = ptf_true
                        e["ptf_pred_time"] = 100 * patient_fog_seconds(S, pos[r], lookup) / analyzed
                        res[s, budget, arm, r] = e
        for budget in BUDGETS:  # per patient, mean over repetitions (Fig. 3: each patient before and after)
            for arm in ARMS:
                for r in RULES:
                    pred = np.mean([res[s, budget, arm, r]["ptf_pred_time"] for s in seeds], axis=0)
                    pp_rows.append(pd.DataFrame({"cohort": cohort, "budget": budget, "arm": arm, "rule": r,
                                                 "subject": S.pts, "ptf_true_time": ptf_true_time,
                                                 "ptf_pred_time": pred, "n_seeds": len(seeds)}))
        for budget in BUDGETS:
            for arm in ARMS:
                icc_boot, icc_point = {}, {}
                for r in RULES:
                    rs = [res[s, budget, arm, r] for s in seeds]
                    row = {"cohort": cohort, "budget": budget, "arm": arm, "rule": r, "n_patients": len(S.pts),
                           "n_seeds": len(seeds)}
                    ratios = {"false_alarms_per_hour": ([q["fa"] for q in rs], [S.hours] * len(rs)),
                              "episodes_detected": ([q["det_p"] for q in rs], [q["n_ep_p"] for q in rs]),
                              "episode_precision": ([q["pred_eps"] - q["fa"] for q in rs], [q["pred_eps"] for q in rs])}
                    for k, (nums, dens) in ratios.items():
                        with np.errstate(invalid="ignore", divide="ignore"):
                            pt_ = [n.sum() / dd.sum() for n, dd in zip(nums, dens)]
                            bt = np.mean([ratio_boot(n, dd, counts) for n, dd in zip(nums, dens)], axis=0)
                        lo, hi = C.interval(bt)
                        row.update({k: float(np.nanmean(pt_)), f"{k}_lo": lo, f"{k}_hi": hi})
                    # percent time frozen: share of windows (analysis b, kept for comparison) and elapsed time (h)
                    for tag, keys, truth, pred in (("ptf", ("bias", "icc"), lambda q: q["ptf_true"], "ptf_pred"),
                                                   ("ptf_time", ("bias", "mae", "icc"), lambda q: ptf_true_time,
                                                    "ptf_pred_time")):
                        point = [ptf_stats(truth(q)[None], q[pred][None]) for q in rs]
                        boot = [ptf_stats(truth(q)[boot_idx], q[pred][boot_idx]) for q in rs]
                        for k in keys:
                            lo, hi = C.interval(np.mean([b[k] for b in boot], axis=0))
                            row.update({f"{tag}_{k}": float(np.mean([p[k][0] for p in point])), f"{tag}_{k}_lo": lo,
                                        f"{tag}_{k}_hi": hi})
                        icc_boot[tag, r] = np.mean([b["icc"] for b in boot], axis=0)
                        icc_point[tag, r] = row[f"{tag}_icc"]
                        # paired against the stored F1 threshold (same windows, patients and resamples)
                        dlt = icc_boot[tag, r] - icc_boot[tag, "f1"]
                        lo, hi = C.interval(dlt)
                        row.update({f"{tag}_icc_minus_f1": icc_point[tag, r] - icc_point[tag, "f1"],
                                    f"{tag}_icc_minus_f1_lo": lo, f"{tag}_icc_minus_f1_hi": hi,
                                    f"{tag}_icc_minus_f1_p": C.two_sided_p(dlt) if r != "f1" else np.nan})
                    row.update({"ptf_time_true_mean": float(np.mean(ptf_true_time)),
                                "ptf_time_pred_mean": float(np.mean([q["ptf_pred_time"].mean() for q in rs]))})
                    rows.append(row)
        summ[cohort] = {"analyzed_hours": float(S.hours.sum()), "assessable_episodes": len(S.episodes),
                        "analyzed_hours_elapsed": float(analyzed.sum() / 3600)}
        log(f"{cohort} done ({time.time() - t_start:.0f} s)")

    op = pd.DataFrame(rows)
    th = pd.DataFrame(thr_rows)
    check = check_against_h(op, a.n_boot, log)
    summary = {"commit": C.git_commit(), "date": time.strftime("%Y-%m-%d"), "labels": labels,
               "runs": {str(s): cfgs[s]["run"] for s in seeds}, "run_commits": {str(s): cfgs[s]["commit"] for s in seeds},
               "n_boot": a.n_boot, "seeds": seeds, "cohorts": summ, "rules": RULES, "arms": ARMS, "budgets": BUDGETS,
               "ptf_definition": "ptf_time_*: elapsed annotated or predicted FOG time over analyzed time (analysis h); "
                                 "ptf_*: share of windows (analysis b), kept for comparison",
               "check_against_analysis_h": check, "seconds": round(time.time() - t_start, 1)}
    show = op[op.budget == "all"][["cohort", "arm", "rule", "false_alarms_per_hour", "episodes_detected",
                                   "episode_precision", "ptf_time_bias", "ptf_time_mae", "ptf_time_icc", "ptf_bias",
                                   "ptf_icc"]]
    report = f"""# Operating points chosen on validation patients (post hoc)

Written by `scripts/paper_val_operating.py` on {summary['date']} at commit {summary['commit']}; labels: {labels};
{len(seeds)} repetitions of the equal-effort runs; {a.n_boot:,} patient resamples per cohort. Rules: f1 (stored
threshold, analysis e), fa4/fa8/fa16 (validation false alarms per hour at most 4, 8, 16), burden (validation share of
windows predicted equals share annotated), burden_time (validation predicted FOG time equals annotated FOG time,
elapsed time over analyzed time). Percent time frozen in elapsed time (ptf_time_*, analysis h's definition, the
reported endpoint) and as a share of windows (ptf_*, analysis b, for comparison); bias and MAE in percentage points.
{check}.

## All training patients, mean over repetitions

{C.md_table(show.set_index(["cohort", "arm", "rule"]))}

Intervals, the paired ICC differences against the F1 rule and the budget of 10 patients: `operating_points.csv`;
per-fold thresholds, their validation false-alarm rates and the validation burden in elapsed time: `thresholds.csv`.
"""
    out = C.write_outputs(NAME, summary, {"operating_points.csv": op, "thresholds.csv": th,
                                          "per_patient.csv": pd.concat(pp_rows, ignore_index=True)}, report)
    print(report)
    print(f"wrote {out} in {summary['seconds']} s")


if __name__ == "__main__":
    main()
