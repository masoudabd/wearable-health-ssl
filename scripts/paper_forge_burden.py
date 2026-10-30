"""Percent time frozen of the released FORGE detector, in elapsed time, so it can be read against our arms.

scripts/forge_detector.py reports the comparator's burden as the share of a patient's analyzed windows classified
FOG. The paper reports the endpoint in elapsed time instead (analysis h), and the two differ by several percentage
points, so the comparator has to be measured the same way before it can be put beside our approaches. This reads the
detector's stored out-of-fold window scores and recomputes the endpoint with the functions of analysis h.

Nothing here is a paired comparison: FORGE's folds, training-set sizes and pretraining corpus are its own, and its
encoder saw the unlabeled recordings of most of these patients. Those caveats are stated in the detector's own report
and are repeated in the paper; this script only changes the unit of the endpoint.

  forge_burden.csv   ensemble, threshold rule: ICC, MAE and bias in elapsed time and, for reference, in windows
  summary.json, report.md

Usage: .venv/bin/python scripts/paper_forge_burden.py [--n-boot N]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

import paper_common as C
from paper_ptf_agreement import STATS, stats
from paper_ptf_time import patient_time, predicted_time

NAME = "forge_burden"
SRC = "forge_detector"
RELEASED_THRESHOLD = 0.35


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=C.N_BOOT)
    ap.add_argument("--out-base", default=None)
    a = ap.parse_args()
    t_start = time.time()

    A = C.DERIVED / "annotations"
    ann = pd.read_parquet(A / "windows.parquet")
    episodes, recordings = pd.read_csv(A / "episodes.csv"), pd.read_csv(A / "recordings.csv")
    pw = pd.read_parquet(C.ROOT / "results" / SRC / "per_window_scores.parquet")
    thr_rows = pd.read_csv(C.ROOT / "results" / SRC / "threshold_metrics.csv")

    test = pw[pw.role == "test"]
    if test.window.nunique() != len(test) // test.seed.nunique():
        raise SystemExit("the stored test scores are not one row per window per seed")

    # the released artifact is the mean of the three seeds' heads; s42 alone is reported beside it
    mean3 = test.groupby(["subject", "window"], sort=False).softmax.mean().reset_index()
    s42 = test[test.seed == 42][["subject", "window", "softmax"]].reset_index(drop=True)
    ensembles = {"mean3": mean3, "s42": s42}

    pt = patient_time(ann, episodes, recordings)
    pt = pt[pt.cohort == "defog"].set_index("subject")
    pts = np.array(sorted(test.subject.unique()))
    missing = set(pts) - set(pt.index)
    if missing:
        raise SystemExit(f"no analyzed time for {len(missing)} patients: {sorted(missing)[:3]}")
    truth = pt.ptf_true_time.reindex(pts).to_numpy()
    analyzed = pt.analyzed_s.reindex(pts).to_numpy()

    t0_of = np.full(len(ann), np.nan)
    t0_of[ann.win.to_numpy()] = ann.t0.to_numpy()
    rec_codes, _ = pd.factorize(ann.rec)
    rec_of = np.full(len(ann), -1, dtype=np.int64)
    rec_of[ann.win.to_numpy()] = rec_codes
    dur_of = np.full(len(ann), np.nan)
    dur_of[ann.win.to_numpy()] = recordings.set_index("rec").duration_s.reindex(ann.rec).to_numpy()

    counts = C.resamples(len(pts), a.n_boot, seed=0)
    idx = np.stack([np.repeat(np.arange(len(pts)), c.astype(int)) for c in counts])

    rows = []
    for name, df in ensembles.items():
        for rule in ("released_0.35", "validation_f1"):
            if rule == "released_0.35":
                pred_mask = df.softmax.to_numpy() >= RELEASED_THRESHOLD
            else:
                # the per-fold validation thresholds the detector chose, as its own report records them
                row = thr_rows[(thr_rows.ensemble == name) & (thr_rows.threshold == rule)]
                if row.empty:
                    continue
                per_fold = {int(k.split("=")[0].replace("fold", "")): float(k.split("=")[1])
                            for k in str(row.threshold_value.iloc[0]).split("; ")}
                fold_of = test.drop_duplicates(["subject", "window"]).set_index(["subject", "window"]).fold
                keys = list(zip(df.subject, df.window))
                thr = np.array([per_fold[int(fold_of.loc[k])] for k in keys])
                pred_mask = df.softmax.to_numpy() >= thr
            d = pd.DataFrame({"subject": df.subject.to_numpy(), "window": df.window.to_numpy(),
                              "pred": pred_mask})
            per_sub = {p: predicted_time(ds, t0_of, rec_of, dur_of) for p, ds in d.groupby("subject")}
            pred_time = np.array([100 * per_sub.get(p, 0.0) / an for p, an in zip(pts, analyzed)])
            win_share = d.groupby("subject").pred.mean().reindex(pts).to_numpy() * 100
            st_t, bt_t = stats(truth[None, :], pred_time[None, :]), stats(truth[idx], pred_time[idx])
            tw = pt.ptf_true_window.reindex(pts).to_numpy() if "ptf_true_window" in pt else None
            r = {"ensemble": name, "threshold": rule, "n_patients": len(pts),
                 "ptf_true_time": float(truth.mean()), "ptf_pred_time": float(pred_time.mean()),
                 "ptf_pred_window": float(win_share.mean())}
            for k in STATS:
                lo, hi = C.interval(bt_t[k])
                r[f"{k}_time"] = float(st_t[k][0])
                r[f"{k}_time_lo"], r[f"{k}_time_hi"] = lo, hi
            rows.append(r)
            print(f"  {name} {rule}: ICC {r['icc_time']:.3f}, bias {r['bias_time']:+.1f} pp "
                  f"(windows: predicted {r['ptf_pred_window']:.1f}%)", flush=True)

    out_df = pd.DataFrame(rows)
    summary = {"commit": C.git_commit(), "date": time.strftime("%Y-%m-%d"), "n_boot": a.n_boot,
               "source": f"results/{SRC}/per_window_scores.parquet",
               "source_commit": json.loads((C.ROOT / "results" / SRC / "summary.json").read_text()).get("commit"),
               "n_patients": int(len(pts)),
               "note": "elapsed-time endpoint (analysis h) for the released FORGE detector; not a paired comparison",
               "seconds": round(time.time() - t_start, 1)}
    report = f"""# The released FORGE detector's freezing burden in elapsed time

Written by `scripts/paper_forge_burden.py` on {summary['date']} at commit {summary['commit']};
{a.n_boot:,} patient resamples; {len(pts)} home patients, out of fold under the authors' own folds.

Percent time frozen here is elapsed FOG time over analyzed time, as analysis h defines it, so these rows can be read
against the paper's arms. The detector's own report measures the same quantity as a share of windows; both are shown.
This is not a paired comparison: FORGE's folds, its 19 training patients per head and its pretraining corpus are its
own, and its encoder was pretrained on the unlabeled recordings of 45 of these 57 patients.

{C.md_table(out_df[['ensemble', 'threshold', 'ptf_true_time', 'ptf_pred_time', 'icc_time', 'mae_time', 'bias_time',
                    'ptf_pred_window']], floatfmt=".3f")}
"""
    out = C.write_outputs(NAME, summary, {"forge_burden.csv": out_df}, report, base=a.out_base)
    print(report)
    print(f"wrote {out} in {summary['seconds']} s")


if __name__ == "__main__":
    main()
