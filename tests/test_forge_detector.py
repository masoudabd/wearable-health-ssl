"""The released FORGE DeFOG detectors: fold membership, pretraining overlap, and the rebuilt model.

The fold checks are what makes the comparison in scripts/forge_detector.py fair, so they are tests, not comments:
a wrong alignment would score patients with a head that was trained on them.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SPLITS = ROOT / "data" / "raw" / "forge" / "code" / "forge-public" / "configs" / "data" / "splits"
HEAD = ROOT / "data" / "raw" / "forge" / "model" / "classification" / "mc_probe_fold0.safetensors"
WINDOWS = ROOT / "data" / "processed" / "fog_contest" / "windows_30hz_5s_hop2.5.npz"


def _script():
    spec = importlib.util.spec_from_file_location("forge_detector", ROOT / "scripts" / "forge_detector.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not SPLITS.exists() or not WINDOWS.exists(),
                    reason="FORGE code or the window cache is not present")
def test_released_folds_partition_the_home_cohort():
    fd = _script()
    from whssl.data import fog_contest as F

    W = F.load_windows()
    subjects = W["subject"][W["cohort"] == "defog"]
    splits = fd.fold_splits()
    fd.check_alignment(splits, subjects)  # raises SystemExit on any overlap or gap
    tests = [set(splits[f]["test"]) for f in fd.FOLDS]
    assert [len(t) for t in tests] == [19, 19, 19]
    assert set().union(*tests) == set(subjects)


@pytest.mark.skipif(not SPLITS.exists() or not WINDOWS.exists(), reason="FORGE code is not present")
def test_check_alignment_refuses_a_leaking_split():
    fd = _script()
    splits = fd.fold_splits()
    subjects = np.array(sorted(set().union(*[set(splits[f]["test"]) for f in fd.FOLDS])))
    leaky = {f: dict(splits[f]) for f in fd.FOLDS}
    leaky[0] = {**leaky[0], "test": leaky[0]["test"][:-1] + [leaky[0]["train"][0]]}
    with pytest.raises(SystemExit):
        fd.check_alignment(leaky, subjects)


@pytest.mark.skipif(not SPLITS.exists(), reason="FORGE code is not present")
def test_pretraining_split_is_the_contest_daily_living_set():
    import pandas as pd

    from whssl.data import fog_contest as F

    meta = F.RAW / "daily_metadata.csv"
    if not meta.exists():
        pytest.skip("contest metadata is not present")
    fd = _script()
    assert fd.pretraining_participants() == set(pd.read_csv(meta).Subject.astype(str))


@pytest.mark.skipif(not HEAD.exists(), reason="FORGE classification heads are not downloaded")
def test_released_detector_loads_strictly_and_ignores_units():
    from whssl.models.forge import forge_detector_probs, load_forge_detector

    det = load_forge_detector("mc_probe_fold0")
    assert det.metadata["splits"] == "kaggle_labeled/kfold_defog_fogcount_valid_mc0"
    assert (det.metadata["phase"], det.metadata["context"], det.metadata["seed"]) == ("probe", "mc", "42")
    x = np.random.default_rng(0).normal(size=(4, 500, 3)).astype(np.float32)
    soft, sig = forge_detector_probs(det, x)
    assert soft.shape == sig.shape == (4,)
    assert np.all((soft >= 0) & (soft <= 1))
    # RevIN rescales every window, so g against m/s^2 changes nothing but its epsilon (1e-5 on the variance).
    scaled, _ = forge_detector_probs(det, x * 9.80665)
    np.testing.assert_allclose(soft, scaled, atol=2e-3)
    # The wavelet power ignores the axis signs, and the released RevIN affine is the identity, so a sign flip is exact.
    flipped, _ = forge_detector_probs(det, x * np.array([1, -1, -1], np.float32))
    np.testing.assert_allclose(soft, flipped, atol=1e-6)
