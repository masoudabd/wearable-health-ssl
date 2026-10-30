"""Post hoc analysis i: how much of each FOG episode a detector actually covers, not just whether it touches it.

Analysis e counts an annotated episode as detected when any window overlapping it is predicted FOG. With 52.8% of
home-cohort episodes shorter than 2.5 s that rule is permissive: a single predicted window can "detect" an episode
while most of it is missed, and a long episode counts the same whether one second or all of it is covered. The
external review asked for a stricter criterion, so this analysis measures the covered share of every episode and
reports detection under a minimum-coverage rule beside the any-overlap rule.

For an annotated episode e and the set P of windows the detector called FOG:

    covered(e) = | e intersected with the union of the spans of the windows in P |
    coverage   = covered(e) / duration(e)
    detected under the rule "cov>=x" when coverage >= x

The union matters: consecutive windows are 2.5 s apart and 5 s long, so neighbours share 2.5 s and summing their
overlaps would double-count it. Because a window is exactly two hops long, only *adjacent* windows overlap and no
three windows share any instant, so inclusion-exclusion over adjacent pairs is exact:

    covered(e) = sum over predicted windows w of overlap(e, w)
                 - sum over adjacent predicted pairs (w, w+1) of | e intersected with [t0_w + 2.5, t0_w + 5) |

Both terms are precomputed per episode-window link, so each arm costs two weighted bincounts instead of interval
merging per episode. `--verify` checks the identity against exact interval merging on a sample of episodes.

Temporal intersection-over-union is not reported: a predicted run may overlap several annotated episodes and an
episode several runs, so IoU would need a matching rule between the two, which would be a further arbitrary choice.
Coverage of the annotated episode is the quantity the criticism is about and needs no matching.

  strict_metrics.csv    cohort, budget, approach: share of episodes detected under each rule, with intervals
  coverage.csv          cohort, budget, approach: distribution of per-episode coverage (mean, median, quartiles)
  coverage_by_duration.csv  the same split by episode duration band
  episode_coverage.parquet  coverage of every assessable episode x repetition x budget x approach
  summary.json, report.md

Usage: .venv/bin/python scripts/paper_episode_strict.py [--runs RUN ...] [--n-boot N] [--verify]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import paper_common as C
from paper_episodes import BUDGETS, DUR_BINS, DUR_LABELS, Cohort, ratio_boot
from paper_ptf_time import intersect_length

NAME = "episode_strict"
RULES = {"any": 0.0, "cov25": 0.25, "cov50": 0.50, "cov80": 0.80}


class StrictCohort(Cohort):
    """Adds, per episode-window link, the window's overlap with the episode and the part of that overlap it shares
    with the next window, which is all inclusion-exclusion needs."""

    def __init__(self, cohort: str, W: dict, ep: pd.DataFrame, epw: pd.DataFrame):
        super().__init__(cohort, W, ep, epw)
        link = epw[epw.episode.isin(self.episodes.episode)]
        eid = pd.Series(np.arange(len(self.episodes)), index=self.episodes.episode)
        self.l_ep = eid[link.episode].to_numpy()
        pos_of = np.full(int(self.win.max()) + 1, -1)
        pos_of[self.win] = np.arange(len(self.win))
        self.l_pos = pos_of[link.win.to_numpy()]
        # episode_windows.csv stores overlap_s rounded, which would leave the identity accurate only to about 1e-4;
        # recomputing it from the window and episode bounds keeps the fast path exact
        e_start = self.episodes.start_s.to_numpy()[self.l_ep]
        e_end = self.episodes.end_s.to_numpy()[self.l_ep]
        t0 = self.t0[self.l_pos]
        self.l_overlap = np.maximum(np.minimum(e_end, t0 + C.WIN) - np.maximum(e_start, t0), 0.0)
        # the shared span of window w and its successor is [t0_w + hop, t0_w + win)
        lo = np.maximum(e_start, t0 + C.HOP)
        hi = np.minimum(e_end, t0 + C.WIN)
        self.l_shared = np.maximum(hi - lo, 0.0)
        # the successor exists only where the next window of the cache continues the same recording
        nxt = self.l_pos + 1
        ok = (nxt < len(self.win)) & np.r_[self.cont, False][np.minimum(nxt, len(self.win) - 1)]
        self.l_next = np.where(ok, nxt, -1)
        self.ep_duration = self.episodes.duration_s.to_numpy().astype(float)

    def coverage(self, pos: np.ndarray) -> np.ndarray:
        """Covered share of every assessable episode, by inclusion-exclusion over adjacent predicted pairs."""
        p = pos[self.l_pos]
        covered = np.bincount(self.l_ep, self.l_overlap * p, minlength=len(self.episodes))
        both = p & np.where(self.l_next >= 0, pos[np.maximum(self.l_next, 0)], False)
        covered -= np.bincount(self.l_ep, self.l_shared * both, minlength=len(self.episodes))
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.clip(covered / self.ep_duration, 0.0, 1.0)

    def coverage_exact(self, pos: np.ndarray, which: np.ndarray) -> np.ndarray:
        """The same quantity by merging intervals, for `--verify`. Slow; used on a sample only."""
        out = np.zeros(len(which))
        for k, e in enumerate(which):
            sel = self.l_pos[(self.l_ep == e) & pos[self.l_pos]]
            if len(sel) == 0:
                continue
            ws = self.t0[sel]
            inter = intersect_length(ws, ws + C.WIN,
                                     np.array([self.episodes.start_s.to_numpy()[e]]),
                                     np.array([self.episodes.end_s.to_numpy()[e]]))
            out[k] = min(inter / self.ep_duration[e], 1.0)
        return out


def main():
    ap = C.common_args(argparse.ArgumentParser())
    ap.add_argument("--n-boot", type=int, default=C.N_BOOT)
    ap.add_argument("--verify", type=int, default=400, help="episodes per cohort to check against interval merging")
    a = ap.parse_args()
    t_start = time.time()
    A = C.DERIVED / "annotations"
    ep, epw = pd.read_csv(A / "episodes.csv"), pd.read_csv(A / "episode_windows.csv")
    by_seed, labels, cfgs = C.load_runs(a.runs, a.allow_superseded,
                                        columns=["cohort", "fold", "budget", "arm", "subject", "y", "pred"])
    W = C.window_meta(labels)
    seeds = sorted(by_seed)
    rows, cov_rows, dur_rows, parq, summ = [], [], [], [], {}
    for cohort in C.COHORTS:
        S = StrictCohort(cohort, W, ep, epw)
        counts = C.resamples(len(S.pts), a.n_boot, seed=0)
        summ[cohort] = {"assessable_episodes": int(len(S.episodes)), "n_patients": int(len(S.pts))}
        res = {}
        for s in seeds:
            dc = by_seed[s][(by_seed[s].cohort == cohort) & by_seed[s].budget.isin(BUDGETS)]
            for (budget, arm), d in dc.groupby(["budget", "arm"]):
                order = np.searchsorted(S.win, d.win.to_numpy())
                if not np.array_equal(np.sort(order), np.arange(len(S.win))):
                    raise SystemExit(f"seed {s} {cohort} {budget} {arm}: windows do not cover the cohort once")
                pos = np.zeros(len(S.win), bool)
                pos[order] = d.pred.to_numpy().astype(bool)
                cov = S.coverage(pos)
                if a.verify and s == seeds[0] and budget == "all" and arm == "harnet5":
                    rng = np.random.default_rng(0)
                    which = rng.choice(len(S.episodes), size=min(a.verify, len(S.episodes)), replace=False)
                    exact = S.coverage_exact(pos, which)
                    worst = float(np.max(np.abs(cov[which] - exact)))
                    print(f"  {cohort}: coverage identity checked on {len(which)} episodes, "
                          f"largest difference {worst:.2e}", flush=True)
                    if worst > 1e-9:
                        raise SystemExit(f"{cohort}: the fast coverage identity disagrees with interval merging "
                                         f"by {worst}, so it is not exact here")
                res[s, budget, arm] = cov
                parq.append(pd.DataFrame({"episode": S.episodes.episode, "seed": s, "budget": budget, "arm": arm,
                                          "coverage": cov, "duration_s": S.ep_duration}))
        print(f"{cohort} done ({time.time() - t_start:.0f} s)", flush=True)

        cells = sorted({(b, arm) for _, b, arm in res})
        for budget, arm in cells:
            covs = [res[s, budget, arm] for s in seeds]
            for rule, thr in RULES.items():
                det = [(c > thr) if rule == "any" else (c >= thr) for c in covs]
                num = [np.bincount(S.ep_col, dd, minlength=len(S.pts)) for dd in det]
                den = np.bincount(S.ep_col, minlength=len(S.pts))
                boot = np.mean([100 * ratio_boot(n, den, counts) for n in num], axis=0)
                lo, hi = C.interval(boot)
                vals = [100 * dd.mean() for dd in det]
                rows.append({"cohort": cohort, "budget": budget, "arm": arm, "rule": rule, "min_coverage": thr,
                             "detected_pct": float(np.mean(vals)), "lo": lo, "hi": hi,
                             "seed_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
                             "n_episodes": int(len(S.episodes)), "n_patients": int(len(S.pts)),
                             "n_seeds": len(seeds)})
            allc = np.concatenate(covs)
            cov_rows.append({"cohort": cohort, "budget": budget, "arm": arm,
                             "coverage_mean": float(allc.mean()), "coverage_median": float(np.median(allc)),
                             "coverage_q1": float(np.quantile(allc, 0.25)),
                             "coverage_q3": float(np.quantile(allc, 0.75)),
                             "share_zero_pct": float(100 * (allc == 0).mean()),
                             "share_full_pct": float(100 * (allc >= 0.999).mean()), "n_seeds": len(seeds)})
            for b_i, lab in enumerate(DUR_LABELS):
                m = S.ep_bin == b_i
                if not m.any():
                    continue
                dur_rows.append({"cohort": cohort, "budget": budget, "arm": arm, "duration": lab,
                                 "n_episodes": int(m.sum()),
                                 "coverage_mean": float(np.mean([c[m].mean() for c in covs])),
                                 "detected_any_pct": float(np.mean([100 * (c[m] > 0).mean() for c in covs])),
                                 "detected_cov50_pct": float(np.mean([100 * (c[m] >= 0.5).mean() for c in covs]))})

    metrics = pd.DataFrame(rows)
    coverage = pd.DataFrame(cov_rows)
    by_duration = pd.DataFrame(dur_rows)
    episodes_out = pd.concat(parq, ignore_index=True)

    summary = {"commit": C.git_commit(), "date": time.strftime("%Y-%m-%d"), "labels": labels,
               "runs": {str(s): cfgs[s]["run"] for s in seeds},
               "run_commits": {str(s): cfgs[s]["commit"] for s in seeds},
               "n_boot": a.n_boot, "seeds": seeds, "rules": RULES, "cohorts": summ,
               "seconds": round(time.time() - t_start, 1)}
    show = (metrics[(metrics.budget == "all") & metrics.arm.isin(["harnet5", "feat60_fi"])]
            .pivot_table(index=["cohort", "arm"], columns="rule", values="detected_pct").round(1))
    report = f"""# Episode coverage and a minimum-coverage detection rule (analysis i, post hoc)

Written by `scripts/paper_episode_strict.py` on {summary['date']} at commit {summary['commit']}; labels: {labels};
{len(seeds)} repetitions; {a.n_boot:,} patient resamples per cohort. Coverage is the share of an annotated episode
covered by the union of the windows the detector called FOG; `any` is the rule used by analysis e.

## Episodes detected (%), all training patients

{C.md_table(show, floatfmt=".1f")}

## Coverage of an episode, all training patients

{C.md_table(coverage[coverage.budget == "all"].drop(columns=["budget", "n_seeds"]), floatfmt=".3f")}

Per-rule intervals and the split by duration: `strict_metrics.csv`, `coverage_by_duration.csv`.
"""
    out = C.write_outputs(NAME, summary, {"strict_metrics.csv": metrics, "coverage.csv": coverage,
                                          "coverage_by_duration.csv": by_duration,
                                          "episode_coverage.parquet": episodes_out}, report, base=a.out_base)
    print(report)
    print(f"wrote {out} in {summary['seconds']} s")


if __name__ == "__main__":
    main()
