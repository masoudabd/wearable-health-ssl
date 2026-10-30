"""Post hoc analysis g (plan v0 section 10, approved 2026-10-04): detection by FOG manifestation and episode length.

Inputs: the RQ1/RQ2 runs, results/paper_derived/annotations/ (scripts/paper_annotations.py) and
results/paper_derived/episodes/episode_detection.parquet (scripts/paper_episodes.py). Exploratory.

  window level   pooled AUROC of the FOG windows of one kind against all non-FOG windows of the cohort: by FOG class
                 (dominant class among the window's FOG samples: turn, walking, start hesitation, or no class for the
                 notype recordings) and by kinetic flag (events.csv, known for the training recordings only; a window
                 counts as akinetic when most of its FOG samples with a known flag are akinetic)
  episode level  share of assessable episodes detected at the stored operating point, by class, kinetic flag and
                 duration (definitions in scripts/paper_episodes.py)

Means over repetitions; 2,000 patient resamples per cohort shared by repetitions and approaches. Outputs:
window_auroc.csv, episode_detection.csv, summary.json, report.md.

Usage: .venv/bin/python scripts/paper_fog_types.py [--runs RUN ...]   (a few minutes on the CPU)
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import paper_common as C
from paper_episodes import DUR_BINS, DUR_LABELS

NAME = "fog_types"
BUDGETS = ["all", "10"]
CLASSES = ["turn", "walking", "start_hesitation", "no_class"]


def main():
    ap = C.common_args(argparse.ArgumentParser())
    ap.add_argument("--n-boot", type=int, default=C.N_BOOT)
    ap.add_argument("--episodes", default=None, help="folder with episode_detection.parquet (default: paper_derived)")
    a = ap.parse_args()
    t_start = time.time()
    A = C.DERIVED / "annotations"
    wn = pd.read_parquet(A / "windows.parquet").set_index("win")
    ep = pd.read_csv(A / "episodes.csv").set_index("episode")
    det = pd.read_parquet((C.Path(a.episodes) if a.episodes else C.DERIVED / "episodes") / "episode_detection.parquet")
    by_seed, labels, cfgs = C.load_runs(a.runs, a.allow_superseded,
                                        columns=["cohort", "fold", "budget", "arm", "subject", "y", "score"])
    seeds = sorted(by_seed)
    win_rows, ep_rows = [], []
    for cohort in C.COHORTS:
        pts = np.array(sorted(wn[wn.cohort == cohort].subject.unique()))
        counts = C.resamples(len(pts), a.n_boot, seed=0)
        point, boot, npos = {}, {}, {}
        for s in seeds:
            dc = by_seed[s][(by_seed[s].cohort == cohort) & by_seed[s].budget.isin(BUDGETS)]
            for (budget, arm), d in dc.groupby(["budget", "arm"]):
                w = wn.loc[d.win.to_numpy()]
                y, sc = d.y.to_numpy(), d.score.to_numpy()
                col = np.searchsorted(pts, d.subject.to_numpy())
                neg = y == 0
                kin_known = (w.kinetic_known_share.to_numpy() >= 0.5) & (y == 1)
                groups = {f"class_{c}": (y == 1) & (w.dominant_class.to_numpy() == c) for c in CLASSES}
                groups["kinetic"] = kin_known & (w.kinetic_share.to_numpy() >= 0.5)
                groups["akinetic"] = kin_known & (w.kinetic_share.to_numpy() < 0.5)
                groups["all_fog"] = y == 1
                for g, pos in groups.items():
                    if pos.sum() < 10:
                        continue
                    keep = pos | neg
                    point[s, budget, arm, g] = C.auroc(sc[keep], y[keep])
                    boot[s, budget, arm, g] = C.boot_metric(C.weighted_auroc, sc[keep], y[keep], col[keep], counts)
                    npos[budget, arm, g] = (int(pos.sum()), len(np.unique(col[pos])))
        for budget, arm, g in sorted({k[1:] for k in point}):
            ks = [(s, budget, arm, g) for s in seeds]
            if not all(k in point for k in ks):
                continue
            lo, hi = C.interval(np.mean([boot[k] for k in ks], axis=0))
            win_rows.append({"cohort": cohort, "budget": budget, "arm": arm, "group": g,
                             "auroc": float(np.mean([point[k] for k in ks])), "auroc_lo": lo, "auroc_hi": hi,
                             "positive_windows": npos[budget, arm, g][0], "patients_with_positives": npos[budget, arm, g][1],
                             "n_seeds": len(seeds)})
        # episode level
        dd = det[det.episode.isin(ep.index[ep.cohort == cohort]) & det.budget.isin(BUDGETS) & det.seed.isin(seeds)]
        e = ep.loc[dd.episode.unique()]
        e = e.assign(duration_group=pd.cut(e.duration_s, DUR_BINS, right=False, labels=DUR_LABELS).astype(str),
                     kinetic_group=np.where(e.kinetic.isna(), "kinetic flag unknown",
                                            np.where(e.kinetic == 1, "kinetic", "akinetic")))
        groupings = {"class": e.fog_class, "kinetic": e.kinetic_group, "duration": e.duration_group}
        col_e = pd.Series(np.searchsorted(pts, e.subject.to_numpy()), index=e.index)
        for (budget, arm), dba in dd.groupby(["budget", "arm"]):
            piv = dba.pivot_table(index="episode", columns="seed", values="detected", aggfunc="first").loc[e.index]
            for gname, gser in groupings.items():
                for level in sorted(gser.unique()):
                    m = (gser == level).to_numpy()
                    cols = col_e.to_numpy()[m]
                    den = np.bincount(cols, minlength=len(pts)).astype(float)
                    vals, bts = [], []
                    for s in seeds:
                        hit = piv[s].to_numpy()[m].astype(float)
                        num = np.bincount(cols, hit, minlength=len(pts))
                        vals.append(num.sum() / den.sum())
                        with np.errstate(invalid="ignore", divide="ignore"):
                            bts.append((counts @ num) / (counts @ den))
                    lo, hi = C.interval(np.mean(bts, axis=0))
                    ep_rows.append({"cohort": cohort, "budget": budget, "arm": arm, "grouping": gname, "group": level,
                                    "episodes": int(m.sum()), "patients": int((den > 0).sum()),
                                    "detected": float(np.mean(vals)), "detected_lo": lo, "detected_hi": hi,
                                    "n_seeds": len(seeds)})
        print(f"{cohort} done ({time.time() - t_start:.0f} s)", flush=True)

    wa, ed = pd.DataFrame(win_rows), pd.DataFrame(ep_rows)
    summary = {"commit": C.git_commit(), "date": time.strftime("%Y-%m-%d"), "labels": labels,
               "runs": {str(s): cfgs[s]["run"] for s in seeds}, "run_commits": {str(s): cfgs[s]["commit"] for s in seeds},
               "n_boot": a.n_boot, "seeds": seeds, "seconds": round(time.time() - t_start, 1)}
    show = wa[wa.budget == "all"].pivot_table(index=["cohort", "arm"], columns="group", values="auroc")
    show_e = ed[(ed.budget == "all") & (ed.arm.isin(["harnet5", "feat60_fi"]))].pivot_table(
        index=["cohort", "grouping", "group"], columns="arm", values="detected")
    report = f"""# Detection by FOG manifestation and episode length (analysis g, post hoc, exploratory)

Written by `scripts/paper_fog_types.py` on {summary['date']} at commit {summary['commit']}; labels: {labels};
{len(seeds)} repetitions.

## Window-level AUROC of each kind of FOG window against all non-FOG windows, all training patients

{C.md_table(show)}

## Share of assessable episodes detected, all training patients (stored operating point)

{C.md_table(show_e)}

Intervals and counts: `window_auroc.csv`, `episode_detection.csv`.
"""
    out = C.write_outputs(NAME, summary, {"window_auroc.csv": wa, "episode_detection.csv": ed}, report,
                          base=a.out_base)
    print(report)
    print(f"wrote {out} in {summary['seconds']} s")


if __name__ == "__main__":
    main()
