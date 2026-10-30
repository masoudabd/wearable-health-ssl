"""Post hoc analysis k: how soon after an episode starts a detector flags it, and whether it does so while the episode
is still under way.

The home cohort comes from a trial of on-demand cueing, and the Discussion separates an operating point for a burden
estimate from one for an alert. Analyses e and i count an episode as detected whenever a predicted window overlaps it,
whenever the flag comes; an alert also has to come in time. A window's decision exists only once its 5 s have been
recorded, so for an assessable annotated episode e = [s, f) and the set P of windows the detector called FOG:

    flag(e)     = min over the windows w in P that overlap e of (t0_w + 5 s)
    latency(e)  = flag(e) - s            (defined for detected episodes; always positive)
    in time     flag(e) < f              (the flag exists while the episode is still under way)

No processing delay and no smoothing over windows are assumed, so these are best cases for a real-time system built on
the same windows. Two references bound what the windowing allows, whatever the detector:

  windows   every window overlapping the episode is called FOG: the earliest possible flag with 5 s windows 2.5 s
            apart
  labels    exactly the windows labeled FOG (at least half of the window annotated) are called FOG: a perfect
            classifier of the training target

Operating points: (1) the stored threshold chosen on validation patients for window F1, on the five fixed RQ1/RQ2
runs, every approach, as in analyses e and i; (2) every rule of scripts/paper_val_operating.py (F1, at most 4, 8 or 16
false alarms per hour on the validation patients, and the two burden matches), on the equal-effort runs. The rules'
thresholds are recomputed with that script's own `choose`, since its thresholds.csv keeps six significant digits.
The script checks that (1) reproduces analysis e's share of episodes detected, that (2) reproduces the operating-point
analysis's, and that the F1 rule of (2) equals (1) for the approaches both run sets share.

Pooled as in analysis e: ratios of per-patient sums, mean over repetitions, with patient resamples per cohort shared
by repetitions, approaches and rules (percentile 95% intervals). Medians and quartiles are pooled over the detected
episodes of all repetitions and are descriptive (no interval).

  latency.csv               cohort, runs, budget, approach, rule: share of episodes detected, share flagged in time
                            (all episodes, episodes of 5 s or longer, and detected episodes), mean latency with
                            intervals, median and quartiles; the two references as runs == "reference"
  latency_by_duration.csv   the same by episode duration band (fixed runs at the F1 threshold, and the references)
  episode_latency.parquet   flag time and latency of every assessable episode x repetition x budget x approach (fixed
                            runs)
  summary.json, report.md

Usage: .venv/bin/python scripts/paper_latency.py [--n-boot N]   (run scripts/paper_episodes.py and
       scripts/paper_val_operating.py first, into the same derived tree; a few minutes on the CPU)
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import paper_common as C
import paper_val_operating as V
from paper_episodes import BUDGETS, DUR_LABELS, Cohort, ratio_boot
from paper_ptf_time import patient_time, window_lookup

NAME = "latency"
LONG_S = 5.0  # an episode at least one window long: the cueing case, where a flag can come in time


class LatencyCohort(Cohort):
    """Adds the decision time of every episode-window link (the end of the window)."""

    def __init__(self, cohort: str, W: dict, ep: pd.DataFrame, epw: pd.DataFrame):
        super().__init__(cohort, W, ep, epw)
        self.l_end = self.t0[self.link_pos] + C.WIN
        self.ep_start = self.episodes.start_s.to_numpy(float)
        self.ep_end = self.episodes.end_s.to_numpy(float)
        self.ep_long = self.episodes.duration_s.to_numpy(float) >= LONG_S
        self.y = W["y"][self.win].astype(bool)

    def flag(self, pos: np.ndarray) -> np.ndarray:
        """Time of the first flag of every assessable episode; inf when no overlapping window is called FOG."""
        out = np.full(len(self.episodes), np.inf)
        p = pos[self.link_pos]
        np.minimum.at(out, self.link_ep[p], self.l_end[p])
        return out


def per_patient(S: LatencyCohort, flag: np.ndarray) -> dict:
    """Numerators and denominators of every ratio, per patient (S.pts order)."""
    det = np.isfinite(flag)
    lat = np.where(det, flag - S.ep_start, 0.0)
    in_time = flag < S.ep_end
    n = len(S.pts)

    def b(w):
        return np.bincount(S.ep_col, np.asarray(w, float), minlength=n)

    return {"detected": (b(det), b(np.ones(len(det)))),
            "in_time": (b(in_time), b(np.ones(len(det)))),
            "in_time_long": (b(in_time & S.ep_long), b(S.ep_long)),
            "in_time_of_detected": (b(in_time), b(det)),
            "latency_mean_s": (b(lat), b(det)),
            "latency_mean_long_s": (b(lat * S.ep_long), b(det & S.ep_long))}


def summarize(S: LatencyCohort, flags: list[np.ndarray], counts: np.ndarray) -> dict:
    """Pooled ratios with intervals, mean over repetitions, and pooled quartiles of latency."""
    row = {}
    parts = [per_patient(S, f) for f in flags]
    for k in parts[0]:
        nums, dens = [p[k][0] for p in parts], [p[k][1] for p in parts]
        with np.errstate(invalid="ignore", divide="ignore"):
            vals = [nn.sum() / dd.sum() for nn, dd in zip(nums, dens)]
            bt = np.mean([ratio_boot(nn, dd, counts) for nn, dd in zip(nums, dens)], axis=0)
        lo, hi = C.interval(bt)
        scale = 1 if k.startswith("latency") else 100
        row.update({k if scale == 1 else f"{k}_pct": scale * float(np.mean(vals)),
                    f"{k}_lo" if scale == 1 else f"{k}_pct_lo": scale * lo,
                    f"{k}_hi" if scale == 1 else f"{k}_pct_hi": scale * hi})
    for tag, m in (("", np.ones(len(S.episodes), bool)), ("_long", S.ep_long)):
        lat = np.concatenate([(f - S.ep_start)[np.isfinite(f) & m] for f in flags])
        q = np.quantile(lat, [0.25, 0.5, 0.75]) if len(lat) else [np.nan] * 3
        row.update({f"latency{tag}_q1_s": float(q[0]), f"latency{tag}_median_s": float(q[1]),
                    f"latency{tag}_q3_s": float(q[2])})
    row.update({"n_episodes": int(len(S.episodes)), "n_episodes_long": int(S.ep_long.sum()),
                "n_patients": int(len(S.pts)), "n_seeds": len(flags)})
    return row


def by_duration(S: LatencyCohort, flags: list[np.ndarray]) -> list[dict]:
    out = []
    for b_i, lab in enumerate(DUR_LABELS):
        m = S.ep_bin == b_i
        if not m.any():
            continue
        lat = np.concatenate([(f - S.ep_start)[np.isfinite(f) & m] for f in flags])
        out.append({"duration": lab, "n_episodes": int(m.sum()),
                    "detected_pct": float(np.mean([100 * np.isfinite(f[m]).mean() for f in flags])),
                    "in_time_pct": float(np.mean([100 * (f[m] < S.ep_end[m]).mean() for f in flags])),
                    "latency_median_s": float(np.median(lat)) if len(lat) else np.nan})
    return out


def positions(S: Cohort, d: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    order = np.searchsorted(S.win, d.win.to_numpy())
    if not np.array_equal(np.sort(order), np.arange(len(S.win))):
        raise SystemExit("the windows of a cell do not cover the cohort once")
    pos = np.zeros(len(S.win), bool)
    pos[order] = values
    return pos


def check(name: str, got: float, want: float, rel: float) -> None:
    if not np.isclose(got, want, rtol=rel, atol=0):
        raise SystemExit(f"{name}: {got!r} here against {want!r} registered")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=C.FIXED_RUNS, help="RQ1/RQ2 runs at the F1 threshold")
    ap.add_argument("--rule-runs", nargs="+", default=V.RUNS, help="equal-effort runs for the validation rules")
    ap.add_argument("--n-boot", type=int, default=C.N_BOOT)
    a = ap.parse_args()
    t_start = time.time()
    A = C.DERIVED / "annotations"
    ep, epw = pd.read_csv(A / "episodes.csv"), pd.read_csv(A / "episode_windows.csv")
    ann, recordings = pd.read_parquet(A / "windows.parquet"), pd.read_csv(A / "recordings.csv")
    pt_all = patient_time(ann, ep, recordings)
    lookup = window_lookup(ann, recordings)
    e_reg = pd.read_csv(C.DERIVED / "episodes" / "episode_metrics.csv", dtype={"budget": str})
    v_reg = pd.read_csv(C.DERIVED / "val_operating" / "operating_points.csv", dtype={"budget": str})

    by_seed, labels, cfgs = C.load_runs(a.runs, columns=["cohort", "fold", "budget", "arm", "subject", "y", "pred"])
    cols = ["cohort", "fold", "budget", "arm", "subject", "y", "score", "pred", "window"]
    rs_seed, labels_r, cfgs_r = C.load_runs(a.rule_runs, columns=cols)
    if labels_r != labels:
        raise SystemExit("the two run sets use different labels")
    for s, o in rs_seed.items():
        if not np.array_equal(o.window.to_numpy(), o.win.to_numpy()):
            raise SystemExit(f"seed {s}: stored window indices differ from the rebuilt ones")
    vals = {cfgs_r[s]["seed"]: pd.read_parquet(C.ROOT / cfgs_r[s]["run"] / "validation_scores.parquet")
            .rename(columns={"window": "win"}) for s in rs_seed}
    W = C.window_meta(labels)
    seeds, seeds_r = sorted(by_seed), sorted(rs_seed)
    rows, dur_rows, parq, summ = [], [], [], {}
    for cohort in C.COHORTS:
        S = LatencyCohort(cohort, W, ep, epw)
        counts = C.resamples(len(S.pts), a.n_boot, seed=0)
        base = {"cohort": cohort}
        for ref, pos in (("windows", np.ones(len(S.win), bool)), ("labels", S.y)):
            f = S.flag(pos)
            rows.append({**base, "runs": "reference", "budget": "none", "arm": ref, "rule": "reference",
                         **summarize(S, [f], counts)})
            dur_rows += [{**base, "runs": "reference", "budget": "none", "arm": ref, **r} for r in by_duration(S, [f])]

        # (1) fixed runs, stored F1 threshold, every approach
        fixed = {}
        for s in seeds:
            dc = by_seed[s][(by_seed[s].cohort == cohort) & by_seed[s].budget.isin(BUDGETS)]
            for (budget, arm), d in dc.groupby(["budget", "arm"]):
                f = S.flag(positions(S, d, d.pred.to_numpy().astype(bool)))
                fixed[s, budget, arm] = f
                parq.append(pd.DataFrame({"episode": S.episodes.episode, "seed": s, "budget": budget, "arm": arm,
                                          "flag_s": f, "latency_s": f - S.ep_start, "duration_s":
                                          S.episodes.duration_s.to_numpy(float)}))
        for budget, arm in sorted({k[1:] for k in fixed}):
            flags = [fixed[s, budget, arm] for s in seeds]
            row = {**base, "runs": "fixed", "budget": budget, "arm": arm, "rule": "f1", **summarize(S, flags, counts)}
            reg = e_reg[(e_reg.cohort == cohort) & (e_reg.budget == budget) & (e_reg.arm == arm)]
            check(f"{cohort} {budget} {arm}: episodes detected (analysis e)", row["detected_pct"],
                  100 * reg.episodes_detected.iloc[0], 1e-5)
            rows.append(row)
            dur_rows += [{**base, "runs": "fixed", "budget": budget, "arm": arm, **r} for r in by_duration(S, flags)]
        print(f"{cohort}: fixed runs done ({time.time() - t_start:.0f} s)", flush=True)

        # (2) equal-effort runs, every validation rule, thresholds recomputed exactly as paper_val_operating chooses them
        pt = pt_all[pt_all.cohort == cohort].set_index("subject")
        rule_flags = {}
        for s in seeds_r:
            o, v0 = rs_seed[s], vals[s]
            for budget in V.BUDGETS:
                for arm in V.ARMS:
                    d = o[(o.cohort == cohort) & (o.budget == budget) & (o.arm == arm)]
                    vv = v0[(v0.cohort == cohort) & (v0.budget == budget) & (v0.arm == arm)]
                    pos = {r: np.zeros(len(S.win), bool) for r in V.RULES}
                    for fold, t in d.groupby("fold"):
                        thr, _ = V.choose(S, vv[vv.fold == fold], pt, lookup)
                        if not np.array_equal((t.score.to_numpy() >= thr["f1"]).astype(int), t.pred.to_numpy()):
                            raise SystemExit(f"{cohort} seed {s} {budget} {arm} fold {fold}: stored threshold mismatch")
                        tidx = np.searchsorted(S.win, t.win.to_numpy())
                        for r in V.RULES:
                            pos[r][tidx] = t.score.to_numpy() >= thr[r]
                    for r in V.RULES:
                        rule_flags[s, budget, arm, r] = S.flag(pos[r])
        for budget in V.BUDGETS:
            for arm in V.ARMS:
                for r in V.RULES:
                    flags = [rule_flags[s, budget, arm, r] for s in seeds_r]
                    row = {**base, "runs": "equal_effort", "budget": budget, "arm": arm, "rule": r,
                           **summarize(S, flags, counts)}
                    reg = v_reg[(v_reg.cohort == cohort) & (v_reg.budget == budget) & (v_reg.arm == arm) &
                                (v_reg.rule == r)]
                    check(f"{cohort} {budget} {arm} {r}: episodes detected (operating points)", row["detected_pct"],
                          100 * reg.episodes_detected.iloc[0], 1e-5)
                    if r == "f1" and (seeds_r[0], budget, arm) in fixed and seeds_r == seeds:
                        for s in seeds:  # the equal-effort runs reproduce the fixed runs' base approaches exactly
                            if not np.array_equal(rule_flags[s, budget, arm, r], fixed[s, budget, arm]):
                                raise SystemExit(f"{cohort} seed {s} {budget} {arm}: F1 flags differ between run sets")
                    rows.append(row)
        summ[cohort] = {"assessable_episodes": int(len(S.episodes)), "episodes_5s_or_longer": int(S.ep_long.sum()),
                        "n_patients": int(len(S.pts))}
        print(f"{cohort}: validation rules done ({time.time() - t_start:.0f} s)", flush=True)

    lat = pd.DataFrame(rows)
    dur = pd.DataFrame(dur_rows)
    summary = {"commit": C.git_commit(), "date": time.strftime("%Y-%m-%d"), "labels": labels,
               "runs": {str(s): cfgs[s]["run"] for s in seeds}, "run_commits": {str(s): cfgs[s]["commit"] for s in seeds},
               "rule_runs": {str(s): cfgs_r[s]["run"] for s in seeds_r},
               "rule_run_commits": {str(s): cfgs_r[s]["commit"] for s in seeds_r},
               "n_boot": a.n_boot, "seeds": seeds, "rules": V.RULES, "long_episode_s": LONG_S, "cohorts": summ,
               "checks": "episodes detected reproduce analysis e (fixed runs) and the operating-point analysis "
                         "(every rule); F1 flags identical between run sets for the shared approaches",
               "seconds": round(time.time() - t_start, 1)}
    cols_show = ["detected_pct", "in_time_pct", "in_time_long_pct", "in_time_of_detected_pct", "latency_median_s",
                 "latency_mean_s", "latency_long_median_s"]
    show = lat[(lat.runs == "reference") | ((lat.budget == "all") & lat.arm.isin(["harnet5", "feat60_fi"]) &
                                            ((lat.runs == "fixed") | lat.rule.isin(["fa8", "burden_time"])))]
    report = f"""# Detection latency and flags in time (analysis k, post hoc)

Written by `scripts/paper_latency.py` on {summary['date']} at commit {summary['commit']}; labels: {labels};
{len(seeds)} repetitions; {a.n_boot:,} patient resamples per cohort. A window's flag exists at its end (start + 5 s);
latency runs from episode onset to the first flag of a window overlapping the episode; "in time" means the flag
exists before the episode ends. Long episodes last at least {LONG_S:g} s. References: every overlapping window called
FOG (windows), exactly the windows labeled FOG (labels). {summary['checks']}.

## Budget all (and the references)

{C.md_table(show.set_index(["cohort", "runs", "arm", "rule"])[cols_show], floatfmt=".1f")}

Every approach, budget and rule with intervals: `latency.csv`; duration bands: `latency_by_duration.csv`.
"""
    out = C.write_outputs(NAME, summary, {"latency.csv": lat, "latency_by_duration.csv": dur,
                                          "episode_latency.parquet": pd.concat(parq, ignore_index=True)}, report)
    print(report)
    print(f"wrote {out} in {summary['seconds']} s")


if __name__ == "__main__":
    main()
