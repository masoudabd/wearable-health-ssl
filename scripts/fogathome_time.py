"""Percent time frozen in elapsed time for the FogAtHome test sets (external test), as analysis h defines it for the
contest cohorts (scripts/paper_ptf_time.py, whose interval functions are used here).

For each participant:

    analyzed time   the union of [t0, t0 + 5) over the participant's windows, clipped to each session's duration
    annotated time  the session's annotated FOG intervals (whssl.data.fogathome.fog_intervals, native rate) intersected
                    with the analyzed time
    predicted time  the union of [t0, t0 + 5) over the windows called FOG

Torch-free, so the post hoc summary (scripts/paper_external.py) can use it as well as the external run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from paper_ptf_time import WIN_S, intersect_length, union_length
from whssl.data import fogathome as FH


class ElapsedTime:
    """Analyzed and annotated FOG seconds per participant of one test set, and predicted FOG seconds on demand.
    Participants are in np.unique(W["subject"]) order, the order the external run's episode counts use."""

    def __init__(self, W: dict, split: str):
        iv, dur = FH.fog_intervals(split, log=None)
        self.pts, col = np.unique(W["subject"], return_inverse=True)
        codes, recs = pd.factorize(pd.Series(W["rec"]))
        self.t0 = np.asarray(W["t0"], float)
        self.end = np.minimum(self.t0 + WIN_S, dur.reindex(recs).to_numpy()[codes])
        order = np.argsort(codes, kind="stable")
        self.rec_idx = np.split(order, np.flatnonzero(np.diff(codes[order])) + 1)  # window indices of each session
        self.codes = codes
        self.rec_col = np.array([col[i[0]] for i in self.rec_idx])                # participant of each session
        ivs = {r: (d.start_s.to_numpy(), d.end_s.to_numpy()) for r, d in iv.groupby("rec")}
        analyzed, annotated = np.zeros(len(recs)), np.zeros(len(recs))
        for k, (r, i) in enumerate(zip(recs, self.rec_idx)):
            analyzed[k] = union_length(self.t0[i], self.end[i])
            if r in ivs:
                annotated[k] = intersect_length(self.t0[i], self.end[i], *ivs[r])
        self.analyzed_s = np.bincount(self.rec_col, analyzed, minlength=len(self.pts))
        self.annotated_s = np.bincount(self.rec_col, annotated, minlength=len(self.pts))
        self.ptf_true = 100 * self.annotated_s / self.analyzed_s

    def predicted_s(self, pred: np.ndarray) -> np.ndarray:
        """Predicted FOG seconds per participant for a boolean prediction over the test set's windows."""
        out = np.zeros(len(self.pts))
        for k in np.unique(self.codes[pred]):
            i = self.rec_idx[k]
            m = i[pred[i]]
            out[self.rec_col[k]] += union_length(self.t0[m], self.end[m])
        return out

    def ptf_pred(self, pred: np.ndarray) -> np.ndarray:
        return 100 * self.predicted_s(pred) / self.analyzed_s
