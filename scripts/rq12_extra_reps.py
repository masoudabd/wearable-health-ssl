"""RQ1/RQ2 with extra frozen foundation-model representations (sensitivity analyses, paper plan v0 section 11 item 5).

The RQ1/RQ2 protocol of scripts/rq12_fog_transfer.py (same folds, budgets, seeds; its base arms reproduce the fixed
runs) with added frozen representations, each probed like harnet5:

  harnet10ctx    the Oxford 10 s network (harnet10, 1,024-dimensional) on a 10 s window centred on every 5 s analysis
                 window (2.5 s of context on each side, edge-padded at recording ends), so the scored windows and their
                 labels are unchanged and paired tests with the 5 s approaches remain valid
  harnet5_swap   harnet5 with the mediolateral and anteroposterior channels swapped (V, AP, ML)
  harnet5_apneg  harnet5 with the anteroposterior channel negated

Orientation: in Capture-24 (wrist AX3, the device family of the model's UK Biobank pretraining; 10 participants),
gravity during walking lies mostly along -x (median x -0.60 g), and our default input already sends the lower-back
vertical axis (about -0.95 g) to x with the same sign. The variants test the two remaining choices: which wrist axes
receive ML and AP, and the AP sign (the two cohorts record AP with opposite signs).

Embeddings are seed-independent and cached in data/processed/fog_contest/. Usage:
  python scripts/rq12_extra_reps.py --device mps --seed 0 --extra tuned_probe [--extra-reps harnet10ctx ...]
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rq12 = _load("rq12_fog_transfer")

from whssl.data import fog_contest as F  # noqa: E402
from whssl.models import pretrained as P  # noqa: E402
from whssl.signal import resample  # noqa: E402

EXTRA = ("harnet10ctx", "harnet5_swap", "harnet5_apneg")


def context_windows(ctx_s: float = 10.0, log=print) -> np.ndarray:
    """(N, ctx_s * 30, 3) windows in the order of the 5 s window cache, each centred on its 5 s window; checked against
    the cache (the centre of every context window equals the cached window exactly)."""
    fs_out, win_s, hop_s = rq12.FS, rq12.WIN, rq12.HOP
    path = F.PROC / f"windows_{int(fs_out)}hz_{win_s:g}s_hop{hop_s:g}_ctx{ctx_s:g}s.npz"
    W = F.load_windows(fs_out, win_s, hop_s)
    T, hop = int(round(win_s * fs_out)), int(round(hop_s * fs_out))
    pad = int(round((ctx_s - win_s) / 2 * fs_out))
    if path.exists():
        Xc = np.load(path)["X"]
    else:
        parts = []
        recs = F.recordings()
        for i, rec in recs.iterrows():
            r = F.load_recording(rec)
            acc = resample(r["acc"], r["fs"], fs_out, axis=0).astype(np.float32)  # as window_recording
            m = len(acc)
            src = np.minimum((np.arange(m) * r["fs"] / fs_out).round().astype(int), len(r["fog"]) - 1)
            valid = r["valid"][src]
            starts = np.arange(0, m - T + 1, hop)
            if len(starts) == 0:
                continue
            keep = valid[starts[:, None] + np.arange(T)[None, :]].mean(axis=1) >= 1.0
            accp = np.pad(acc, ((pad, pad), (0, 0)), mode="edge")
            parts.append(accp[starts[keep][:, None] + np.arange(T + 2 * pad)[None, :]])
            if log and (i + 1) % 200 == 0:
                log(f"  context windows: {i + 1}/{len(recs)} recordings")
        Xc = np.concatenate(parts)
        np.savez(path, X=Xc)
    if Xc.shape[0] != len(W["X"]) or not np.array_equal(Xc[:, pad:pad + T], W["X"]):
        raise SystemExit("context windows do not line up with the 5 s window cache")
    return Xc


def make_frozen(extra: tuple[str, ...]):
    base = rq12.frozen_representations

    def frozen(X, log, device="cpu", seed=0):
        reps = base(X, log, device, seed)
        proc = F.PROC
        t = time.time()
        if "harnet10ctx" in extra:
            reps["harnet10ctx"] = rq12.cached(proc / "emb_harnet10_ctx10s.npz", lambda: P.harnet_embed(
                P.load_harnet("harnet10", True), context_windows(10.0, log), device=device))
        if "harnet5_swap" in extra:
            reps["harnet5_swap"] = rq12.cached(proc / "emb_harnet5_swapMLAP.npz", lambda: P.harnet_embed(
                P.load_harnet("harnet5", True), np.ascontiguousarray(X[:, :, [0, 2, 1]]), device=device))
        if "harnet5_apneg" in extra:
            reps["harnet5_apneg"] = rq12.cached(proc / "emb_harnet5_apneg.npz", lambda: P.harnet_embed(
                P.load_harnet("harnet5", True), X * np.array([1, 1, -1], np.float32), device=device))
        log(f"extra representations ready: {', '.join(extra)} ({time.time() - t:.0f}s)")
        return reps

    return frozen


def main():
    ap = rq12.base_parser()
    ap.add_argument("--extra-reps", nargs="+", default=list(EXTRA), choices=list(EXTRA))
    args = ap.parse_args()
    rq12.frozen_representations = make_frozen(tuple(args.extra_reps))  # run_experiment looks it up at call time
    rq12.run_experiment(args, "rq12_extra_reps")


if __name__ == "__main__":
    main()
