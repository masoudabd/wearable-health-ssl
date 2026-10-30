"""FOG labels of the contest data (needs data/raw/fog_contest; skipped without it).

The 46 home 'notype' recordings are labelled from events.csv because their Event column holds event times in seconds
written as row numbers (found 2026-10-03). These tests check the events.csv rule against labels it does not share.
"""
import numpy as np
import pandas as pd
import pytest

from whssl.data import fog_contest as F

pytestmark = pytest.mark.skipif(not (F.RAW / "events.csv").exists(), reason="contest data not downloaded")


def test_events_rule_matches_the_typed_home_files():
    recs = F.recordings()
    typed = recs[(recs.cohort == "defog") & (recs.split == "train")].head(15)
    agree, n = 0, 0
    for _, rec in typed.iterrows():
        d = pd.read_csv(rec.path)
        own = d[F.FOG_TYPES].to_numpy().max(axis=1)
        agree += int((own == F.events_fog(rec.Id, len(d), rec.fs)).sum())
        n += len(d)
    assert agree / n > 0.999


def test_notype_labels_equal_the_events_table():
    recs = F.recordings()
    notype = recs[recs.split == "notype"]
    assert len(notype) == 46
    ev = pd.read_csv(F.RAW / "events.csv").dropna(subset=["Init", "Completion"])
    total_loaded, total_events = 0.0, 0.0
    for _, rec in notype.iterrows():
        r = F.load_recording(rec)
        n = len(r["fog"])
        assert np.array_equal(pd.read_csv(rec.path, usecols=["Time"]).Time.to_numpy(), np.arange(n))  # Time = sample
        e = ev[ev.Id == rec.Id]
        expected = float((np.minimum(e.Completion, n / rec.fs) - np.maximum(e.Init, 0)).clip(lower=0).sum())
        assert abs(r["fog"].sum() / rec.fs - expected) <= len(e) / rec.fs + 1e-9  # rounding: at most 1 sample per event
        total_loaded += r["fog"].sum() / rec.fs
        total_events += expected
    assert total_events > 60 * 80  # events.csv lists about 81 min of FOG in these files; the Event column 0.8 min
    assert abs(total_loaded - total_events) < 1.0
