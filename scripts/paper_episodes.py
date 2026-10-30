"""Post hoc analysis e (writing brief 8.5): FOG episodes detected and false alarms per hour.

From the stored window predictions of RQ1/RQ2 (threshold chosen on validation patients for window F1; validation
scores were not stored, so no other operating point can be chosen on validation patients) and the annotations of
scripts/paper_annotations.py (run it first). Definitions:

  predicted episode   a run of consecutive predicted-FOG windows of one recording (windows 2.5 s apart; a gap or
                      an excluded window ends the run)
  false alarm         a predicted episode none of whose windows contains any annotated FOG
  analyzed time       the union of a recording's analyzed windows (a run of n windows covers (n - 1) x 2.5 s + 5 s)
  assessable episode  an annotated episode entirely inside valid time that overlaps at least one analyzed window
  detected            at least one window overlapping the episode is predicted FOG

Pooled rates (false alarms per analyzed hour; share of assessable episodes detected; share of predicted episodes
that touch FOG) are averaged over repetitions, with 2,000 patient resamples per cohort shared by repetitions and
approaches. A threshold sweep on the test scores (pooled over folds, like pooled AUROC) traces episodes detected
against false alarms per hour; it describes the trade-off and selects nothing. Outputs:

  episode_metrics.csv      cohort, budget, approach: rates with intervals; sensitivity by episode duration
  false_alarms.csv         per patient (budgets 10 and all): analyzed hours, false alarms per hour, episodes detected,
                           mean over repetitions
  sweep.csv                budget all: false alarms per hour and episodes detected at 40 score thresholds
  operating_points.csv     budget all: episodes detected at 4, 8 and 16 false alarms per hour (interpolated on each
                           repetition's curve, then averaged; descriptive, chosen on test scores)
  episode_detection.parquet  every assessable episode x repetition x budget x approach: detected (input of
                           scripts/paper_fog_types.py)
  summary.json, report.md

Usage: .venv/bin/python scripts/paper_episodes.py [--runs RUN ...]   (about two minutes on the CPU)
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import paper_common as C

NAME = "episodes"
BUDGETS = ["10", "all"]
DUR_BINS = [0, 1, 2.5, 5, 10, 30, np.inf]
DUR_LABELS = ["under 1 s", "1 to 2.5 s", "2.5 to 5 s", "5 to 10 s", "10 to 30 s", "30 s or longer"]
N_SWEEP = 40
FA_TARGETS = [4, 8, 16]


class Cohort:
    """Window and episode structure of one cohort, in window-cache order."""

    def __init__(self, cohort: str, W: dict, ep: pd.DataFrame, epw: pd.DataFrame):
        self.win = np.flatnonzero(W["cohort"] == cohort)
        self.rec, self.t0 = W["rec"][self.win], W["t0"][self.win]
        self.subject, self.touch = W["subject"][self.win], W["frac"][self.win] > 0
        self.cont = np.r_[False, (self.rec[1:] == self.rec[:-1]) & (np.abs(np.diff(self.t0) - C.HOP) < 1e-6)]
        self.pts, self.col = C.patient_index(self.subject)
        run_start = ~self.cont
        hours = (np.full(len(self.win), C.HOP) + run_start * (C.WIN - C.HOP)) / 3600
        self.hours = np.bincount(self.col, hours, minlength=len(self.pts))
        pos_of = np.full(int(self.win.max()) + 1, -1)
        pos_of[self.win] = np.arange(len(self.win))
        e = ep[(ep.cohort == cohort) & (ep.valid_share == 1)]
        link = epw[epw.episode.isin(e.episode)]
        self.episodes = e[e.episode.isin(link.episode)].reset_index(drop=True)
        eid = pd.Series(np.arange(len(self.episodes)), index=self.episodes.episode)
        self.link_ep, self.link_pos = eid[link.episode].to_numpy(), pos_of[link.win.to_numpy()]
        self.ep_col = np.searchsorted(self.pts, self.episodes.subject.to_numpy())
        self.ep_bin = pd.cut(self.episodes.duration_s, DUR_BINS, right=False, labels=False).to_numpy()

    def evaluate(self, pos: np.ndarray) -> dict:
        """pos: predicted FOG per window (cohort window order). Per-patient counts."""
        start = pos & ~(np.r_[False, pos[:-1]] & self.cont)
        run_id = np.cumsum(start) - 1
        ip = np.flatnonzero(pos)
        n_runs = int(start.sum())
        touches = np.zeros(n_runs, bool)
        np.logical_or.at(touches, run_id[ip], self.touch[ip])
        run_col = self.col[np.flatnonzero(start)]
        fa = np.bincount(run_col[~touches], minlength=len(self.pts))
        pred_eps = np.bincount(run_col, minlength=len(self.pts))
        det = np.zeros(len(self.episodes), bool)
        np.logical_or.at(det, self.link_ep, pos[self.link_pos])
        return {"fa": fa, "pred_eps": pred_eps, "det": det,
                "det_p": np.bincount(self.ep_col, det, minlength=len(self.pts)),
                "n_ep_p": np.bincount(self.ep_col, minlength=len(self.pts))}


def ratio_boot(num: np.ndarray, den: np.ndarray, counts: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return (counts @ num) / (counts @ den)


def main():
    ap = C.common_args(argparse.ArgumentParser())
    ap.add_argument("--n-boot", type=int, default=C.N_BOOT)
    a = ap.parse_args()
    t_start = time.time()
    A = C.DERIVED / "annotations"
    ep, epw = pd.read_csv(A / "episodes.csv"), pd.read_csv(A / "episode_windows.csv")
    by_seed, labels, cfgs = C.load_runs(a.runs, a.allow_superseded,
                                        columns=["cohort", "fold", "budget", "arm", "subject", "y", "score", "pred"])
    W = C.window_meta(labels)
    seeds = sorted(by_seed)
    rows, pat_rows, sweep_rows, det_rows, summ_cohort = [], [], [], [], {}
    for cohort in C.COHORTS:
        S = Cohort(cohort, W, ep, epw)
        counts = C.resamples(len(S.pts), a.n_boot, seed=0)
        summ_cohort[cohort] = {"assessable_episodes": len(S.episodes), "analyzed_hours": float(S.hours.sum())}
        res = {}
        for s in seeds:
            dc = by_seed[s][(by_seed[s].cohort == cohort) & by_seed[s].budget.isin(BUDGETS)]
            for (budget, arm), d in dc.groupby(["budget", "arm"]):
                order = np.searchsorted(S.win, d.win.to_numpy())
                if not np.array_equal(np.sort(order), np.arange(len(S.win))):
                    raise SystemExit(f"seed {s} {cohort} {budget} {arm}: windows do not cover the cohort once")
                pos = np.zeros(len(S.win), bool)
                pos[order] = d.pred.to_numpy().astype(bool)
                r = S.evaluate(pos)
                res[s, budget, arm] = r
                det_rows.append(pd.DataFrame({"episode": S.episodes.episode, "seed": s, "budget": budget, "arm": arm,
                                              "detected": r["det"]}))
                if budget == "all":
                    score = np.zeros(len(S.win))
                    score[order] = d.score.to_numpy()
                    for q in np.linspace(0.5, 0.995, N_SWEEP):
                        thr = np.quantile(score, q)
                        rq = S.evaluate(score >= thr)
                        sweep_rows.append({"cohort": cohort, "budget": budget, "arm": arm, "seed": s, "quantile": q,
                                           "false_alarms_per_hour": rq["fa"].sum() / S.hours.sum(),
                                           "episodes_detected": rq["det"].mean()})
        for budget, arm in sorted({k[1:] for k in res}):
            rs = [res[s, budget, arm] for s in seeds]
            row = {"cohort": cohort, "budget": budget, "arm": arm, "n_patients": len(S.pts),
                   "assessable_episodes": len(S.episodes), "analyzed_hours": float(S.hours.sum()), "n_seeds": len(seeds)}
            stats = {
                "false_alarms_per_hour": ([r["fa"] for r in rs], [S.hours] * len(rs)),
                "episodes_detected": ([r["det_p"] for r in rs], [r["n_ep_p"] for r in rs]),
                "episode_precision": ([r["pred_eps"] - r["fa"] for r in rs], [r["pred_eps"] for r in rs]),
                "predicted_episodes_per_hour": ([r["pred_eps"] for r in rs], [S.hours] * len(rs)),
            }
            for k, (nums, dens) in stats.items():
                vals = [n.sum() / d.sum() for n, d in zip(nums, dens)]
                bt = np.mean([ratio_boot(n, d, counts) for n, d in zip(nums, dens)], axis=0)
                lo, hi = C.interval(bt)
                row.update({k: float(np.mean(vals)), f"{k}_lo": lo, f"{k}_hi": hi})
            fa_p = np.mean([r["fa"] / S.hours for r in rs], axis=0)
            row["false_alarms_per_hour_patient_median"] = float(np.median(fa_p))
            for b, lab in enumerate(DUR_LABELS):
                m = S.ep_bin == b
                row[f"detected_{lab}"] = float(np.mean([r["det"][m].mean() for r in rs])) if m.any() else np.nan
                row[f"n_{lab}"] = int(m.sum())
            rows.append(row)
            det_p = np.mean([r["det_p"] for r in rs], axis=0)
            pat_rows.append(pd.DataFrame({"cohort": cohort, "budget": budget, "arm": arm, "subject": S.pts,
                                          "analyzed_hours": S.hours, "false_alarms_per_hour": fa_p,
                                          "assessable_episodes": rs[0]["n_ep_p"], "episodes_detected": det_p}))
        print(f"{cohort} done ({time.time() - t_start:.0f} s)", flush=True)

    metrics = pd.DataFrame(rows)
    sw = pd.DataFrame(sweep_rows)
    sweep = (sw.groupby(["cohort", "budget", "arm", "quantile"])[["false_alarms_per_hour", "episodes_detected"]]
             .mean().reset_index())
    op_rows = []  # episodes detected at fixed false-alarm rates, read off each repetition's curve, then averaged
    for (cohort, budget, arm), g in sw.groupby(["cohort", "budget", "arm"]):
        for target in FA_TARGETS:
            vals = []
            for s, gs in g.groupby("seed"):
                gs = gs.sort_values("false_alarms_per_hour")
                if gs.false_alarms_per_hour.min() <= target <= gs.false_alarms_per_hour.max():
                    vals.append(float(np.interp(target, gs.false_alarms_per_hour, gs.episodes_detected)))
            if len(vals) == len(seeds):
                op_rows.append({"cohort": cohort, "budget": budget, "arm": arm, "false_alarms_per_hour": target,
                                "episodes_detected": float(np.mean(vals)), "seed_sd": float(np.std(vals, ddof=1))
                                if len(vals) > 1 else np.nan, "n_seeds": len(vals)})
    operating = pd.DataFrame(op_rows)
    summary = {"commit": C.git_commit(), "date": time.strftime("%Y-%m-%d"), "labels": labels,
               "runs": {str(s): cfgs[s]["run"] for s in seeds}, "run_commits": {str(s): cfgs[s]["commit"] for s in seeds},
               "n_boot": a.n_boot, "seeds": seeds, "cohorts": summ_cohort, "seconds": round(time.time() - t_start, 1)}
    show = metrics[metrics.budget == "all"][["cohort", "arm", "false_alarms_per_hour", "episodes_detected",
                                             "episode_precision", "detected_under 1 s", "detected_2.5 to 5 s"]]
    report = f"""# FOG episodes detected and false alarms per hour (analysis e, post hoc)

Written by `scripts/paper_episodes.py` on {summary['date']} at commit {summary['commit']}; labels: {labels};
{len(seeds)} repetitions; operating point: the stored threshold chosen on validation patients for window F1.
Assessable episodes: lab {summ_cohort['tdcsfog']['assessable_episodes']:,}, home
{summ_cohort['defog']['assessable_episodes']:,}; analyzed hours: lab {summ_cohort['tdcsfog']['analyzed_hours']:.1f}, home
{summ_cohort['defog']['analyzed_hours']:.1f}.

## All training patients, mean over repetitions

{C.md_table(show.set_index(["cohort", "arm"]))}

Intervals, the budget of 10 patients and all duration bins: `episode_metrics.csv`; per patient: `false_alarms.csv`;
threshold sweep: `sweep.csv`.
"""
    out = C.write_outputs(NAME, summary, {"episode_metrics.csv": metrics,
                                          "false_alarms.csv": pd.concat(pat_rows, ignore_index=True),
                                          "sweep.csv": sweep, "operating_points.csv": operating,
                                          "episode_detection.parquet": pd.concat(det_rows, ignore_index=True)},
                          report, base=a.out_base)
    print(report)
    print(f"wrote {out} in {summary['seconds']} s")


if __name__ == "__main__":
    main()
