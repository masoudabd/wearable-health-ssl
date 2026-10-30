"""Shared helpers of the post hoc paper analyses (scripts/paper_*.py; writing brief 8.5).

Read-only on data, caches and results: every analysis writes only to results/paper_derived/<name>/.

The RQ1/RQ2 runs do not store window indices. attach_windows() rebuilds them from the same deterministic patient
split (whssl.eval.splits.subject_kfold with the run's folds and seed, stratified on the window labels the run used)
and checks every out-of-fold row against its stored patient and label. Runs made before the notype label fix
(commit 493fb42) used the archived cache; attach_windows() finds the cache that reproduces a run and refuses a
superseded one unless told otherwise (only for testing a script on old outputs).

Uncertainty: patients are resampled with replacement (whssl.eval.paired.patient_counts); one set of resamples per
cohort is shared by every repetition, approach and budget, so differences are paired, and a statistic averaged
over repetitions is averaged within each resample.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from whssl.data import fog_contest as F  # noqa: E402
from whssl.eval.paired import boot_auroc, patient_counts, weighted_auroc  # noqa: E402
from whssl.eval.splits import subject_kfold  # noqa: E402

# WHSSL_DERIVED redirects every post hoc analysis to a parallel tree (used for the 20,000-resample rebuild), so the
# registered outputs stay untouched while the same scripts run side by side.
DERIVED = Path(os.environ.get("WHSSL_DERIVED") or ROOT / "results" / "paper_derived")
CACHES = {"fixed": F.PROC / "windows_30hz_5s_hop2.5.npz",
          "notype_bug": F.PROC / "windows_30hz_5s_hop2.5_notype_bug_2026-10-03.npz"}
META = ["y", "frac", "t0", "rec", "subject", "cohort", "split", "medication", "visit"]
FIXED_RUNS = [f"results/rq12_fog_transfer_mps_fixed_seed{s}" for s in range(5)]
OLD_RUNS = ["results/rq12_fog_transfer_mps"] + [f"results/rq12_fog_transfer_mps_seed{s}" for s in range(1, 5)]
COHORTS = ["tdcsfog", "defog"]
BUDGETS = ["5", "10", "20", "all"]
ARMS = ["harnet5", "feat60_fi", "ssl_indomain", "supervised", "harnet5_rand", "feat60"]
COMPARISONS = [("harnet5", "feat60_fi"), ("ssl_indomain", "feat60_fi"), ("harnet5", "harnet5_rand"),
               ("harnet5", "supervised"), ("ssl_indomain", "harnet5"), ("supervised", "feat60_fi")]
WIN, HOP, FS = 5.0, 2.5, 30.0
N_BOOT = 2000


def git_commit() -> str:
    c = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "src", "scripts"], cwd=ROOT, capture_output=True,
                           text=True).stdout.strip()
    return c + ("+uncommitted" if dirty else "")


_META: dict[str, dict] = {}


def window_meta(version: str = "fixed") -> dict:
    """Per-window metadata of a window cache (the signals are not loaded)."""
    if version not in _META:
        z = np.load(CACHES[version], allow_pickle=False)
        _META[version] = {k: z[k] for k in META}
    return _META[version]


def run_config(run: str | Path) -> dict:
    return json.loads((ROOT / run / "config.json").read_text())


def _rebuild(oof: pd.DataFrame, W: dict, cfg: dict) -> np.ndarray:
    win = np.full(len(oof), -1, dtype=np.int64)
    for cohort, oc in oof.groupby("cohort", sort=False):
        m = np.flatnonzero(W["cohort"] == cohort)
        y, g = W["y"][m].astype(int), W["subject"][m]
        order = m[np.concatenate([te for _, te in subject_kfold(y, g, cfg["folds"], cfg["seed"])])]
        for (budget, arm), d in oc.groupby(["budget", "arm"], sort=False):
            if len(d) != len(order):
                raise AssertionError(f"{cohort}/{budget}/{arm}: {len(d)} rows, {len(order)} rebuilt")
            if not (d.subject.to_numpy() == W["subject"][order]).all():
                raise AssertionError(f"{cohort}/{budget}/{arm}: patients differ")
            if not (d.y.to_numpy() == W["y"][order]).all():
                raise AssertionError(f"{cohort}/{budget}/{arm}: labels differ")
            win[d.index.to_numpy()] = order
    return win


def load_run(run: str | Path, allow_superseded: bool = False, columns: list[str] | None = None):
    """Out-of-fold rows of one RQ1/RQ2 run with the global window index `win`; returns (oof, config, labels)."""
    cfg = run_config(run)
    if cfg.get("quick"):
        raise SystemExit(f"{run}: quick runs are not supported")
    oof = pd.read_parquet(ROOT / run / "out_of_fold_scores.parquet", columns=columns).reset_index(drop=True)
    errors = []
    for version in CACHES:
        try:
            win = _rebuild(oof, window_meta(version), cfg)
        except AssertionError as e:
            errors.append(f"{version} cache: {e}")
            continue
        if version != "fixed" and not allow_superseded:
            raise SystemExit(f"{run} used the superseded labels ({version}); --allow-superseded only for tests")
        return oof.assign(win=win), cfg, version
    raise SystemExit(f"{run}: no window cache reproduces the run ({'; '.join(errors)})")


def load_runs(runs: list[str], allow_superseded: bool = False, columns: list[str] | None = None):
    """{seed: oof} for several runs, all reproduced by the same cache; returns (by_seed, labels, configs)."""
    by_seed, versions, cfgs = {}, set(), {}
    for run in runs:
        oof, cfg, version = load_run(run, allow_superseded, columns)
        if cfg["seed"] in by_seed:
            raise SystemExit(f"two runs with seed {cfg['seed']}")
        by_seed[cfg["seed"]], cfgs[cfg["seed"]] = oof, {**cfg, "run": str(run)}
        versions.add(version)
    if len(versions) > 1:
        raise SystemExit(f"runs mix label versions: {sorted(versions)}")
    return by_seed, versions.pop(), cfgs


def patient_table() -> pd.DataFrame:
    """Clinical scores per patient: means over the patient's rows in subjects.csv (one per visit for DeFOG), as in
    scripts/rq3_severity_fog.py; sex is the value given in most rows (first row on a tie) and flagged when rows
    disagree."""
    s = F.subjects()
    g = s.groupby("Subject")
    t = g.agg(age=("Age", "mean"), years_dx=("YearsSinceDx", "mean"), updrs3_on=("UPDRSIII_On", "mean"),
              updrs3_off=("UPDRSIII_Off", "mean"), nfogq=("NFOGQ", "mean"), rows=("Subject", "size"))
    t["sex"] = g.Sex.agg(lambda x: x.value_counts().index[0] if x.value_counts().is_unique or len(x) == 1
                         else x.iloc[0])
    t["sex_inconsistent"] = g.Sex.agg(lambda x: x.dropna().nunique() > 1)
    return t.rename_axis("subject")


def patient_index(subjects: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sorted unique patients and the column of each window's patient."""
    pts = np.array(sorted(set(subjects)))
    return pts, np.searchsorted(pts, subjects)


def resamples(n_patients: int, n_boot: int = N_BOOT, seed: int = 0) -> np.ndarray:
    return patient_counts(n_patients, n_boot, seed=seed)


def weighted_auprc(scores: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Average precision as sklearn's average_precision_score (precision at each distinct score, weighted by the
    gain in recall), with per-window weights. weights (n,) gives a float; (B, n) one value per row; a row without
    positives gives NaN."""
    w = np.atleast_2d(np.asarray(weights, dtype=np.float64))
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    s, yy, w = np.asarray(scores)[order], np.asarray(y)[order].astype(bool), w[:, order]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    tp = np.cumsum(np.add.reduceat(w * yy, starts, axis=1), axis=1)
    fp = np.cumsum(np.add.reduceat(w * ~yy, starts, axis=1), axis=1)
    pos = tp[:, -1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = tp / (tp + fp)
        rec = tp / pos
    gain = np.diff(np.concatenate([np.zeros((len(w), 1)), rec], axis=1), axis=1)
    ap = np.where(gain > 0, gain * prec, 0.0).sum(axis=1)
    ap[pos[:, 0] == 0] = np.nan
    return ap if np.ndim(weights) == 2 else float(ap[0])


def boot_metric(fn, scores: np.ndarray, y: np.ndarray, col: np.ndarray, counts: np.ndarray,
                chunk: int = 200) -> np.ndarray:
    """fn(scores, y, weights) in every resample; col: each window's column in `counts`.

    AUROC takes the closed form of `boot_auroc`, which gives the same values without expanding the weights to every
    window on every replicate. AUPRC is not bilinear in the weights and still goes replicate by replicate.
    """
    if fn is weighted_auroc:
        return boot_auroc(scores, y, col, counts)
    out = np.empty(len(counts))
    for i in range(0, len(counts), chunk):
        out[i:i + chunk] = fn(scores, y, counts[i:i + chunk][:, col])
    return out


def auroc(scores, y, weights=None):
    return weighted_auroc(scores, y, np.ones(len(y)) if weights is None else weights)


def auprc(scores, y, weights=None):
    return weighted_auprc(scores, y, np.ones(len(y)) if weights is None else weights)


def two_sided_p(d: np.ndarray) -> float:
    d = d[np.isfinite(d)]
    return float(min(1.0, max(2 * min(np.mean(d <= 0), np.mean(d >= 0)), 1.0 / len(d))))


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    adj, running = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def interval(boot: np.ndarray) -> tuple[float, float]:
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def write_outputs(name: str, summary: dict, tables: dict[str, pd.DataFrame], report: str,
                  base: str | Path | None = None) -> Path:
    """Writes results/paper_derived/<name>/ (or <base>/<name>/ for a test on superseded runs)."""
    out = (Path(base) if base else DERIVED) / name
    out.mkdir(parents=True, exist_ok=True)
    for fname, df in tables.items():
        if fname.endswith(".parquet"):
            df.to_parquet(out / fname, index=False)
        else:
            df.to_csv(out / fname, index=False, float_format="%.6g")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=lambda x: x.item() if hasattr(x, "item")
                                                 else str(x)) + "\n")
    (out / "report.md").write_text(report)
    return out


def common_args(ap):
    """--runs, --allow-superseded and --out-base, shared by the analyses that read RQ1/RQ2 runs."""
    ap.add_argument("--runs", nargs="+", default=FIXED_RUNS, help="RQ1/RQ2 runs (default: the five fixed seeds)")
    ap.add_argument("--allow-superseded", action="store_true", help="accept runs on the old labels (tests only)")
    ap.add_argument("--out-base", default=None, help="write <out-base>/<name>/ instead of results/paper_derived/")
    return ap


def md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """A pandas frame as a Markdown pipe table (no extra dependency); the index becomes the first columns."""
    d = df.reset_index() if not isinstance(df.index, pd.RangeIndex) else df
    cells = [[f"{v:{floatfmt}}" if isinstance(v, (float, np.floating)) else str(v) for v in row]
             for row in d.itertuples(index=False)]
    head = [str(c) for c in d.columns]
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
                     + ["| " + " | ".join(r) + " |" for r in cells])
