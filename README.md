# whssl: wearable foundation models and freezing of gait detection

Analysis code for a manuscript under review on detecting freezing of gait (FOG) in Parkinson's disease from a
lower-back accelerometer with a frozen wearable foundation model, clinically informed features, self-supervised and
supervised networks, evaluated with patient-level, leakage-safe protocols. The citation will be added on publication.

Exported on 2026-10-30 from commit `8f9d36a` of the working repository; `EXPORT_MANIFEST.json` lists every file with its
SHA-256.

## Layout

| Path | Content |
|---|---|
| `src/whssl/eval/` | Patient-wise splits, validation carving, metrics, patient-level bootstrap, linear probe |
| `src/whssl/data/` | Loaders: FOG contest cohorts and daily-living recordings, FogAtHome |
| `src/whssl/features.py` | 60 window features and freeze-index features, vectorized |
| `src/whssl/models/` | Wrappers for the Oxford foundation model and FORGE, 1D ResNet encoder, supervised baseline |
| `src/whssl/ssl/` | Multi-task self-supervision (arrow of time, permutation, time warp) |
| `scripts/` | Downloads, experiments including fine-tuning (`rq12_fog_transfer.py` and the other runners), post hoc analyses (`paper_*.py`), re-bootstrap (`rebootstrap.sh`) |
| `tests/` | `python -m pytest tests -q`: leakage, feature fidelity and analysis checks |
| `results/` | Derived results: summary and paired-test tables of every reported analysis, with the per-seed runs |

## Setup

Python 3.11. `python -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .`

Data are not included. `scripts/download_datasets.py` fetches the public datasets and checks sizes and checksums
(`--list` for a dry run); `scripts/download_forge.py` fetches the FORGE release. Scripts read data from `WHSSL_DATA` or `data/`. Set `WHSSL_CONTACT_EMAIL` to identify
yourself to the data servers. The Oxford foundation model's weights come from their authors under their academic
licence.

## Protocol rules every script follows

1. Splits are made on patients, never on windows.
2. Test patients are used once; thresholds, checkpoints and hyperparameters are chosen on validation patients carved
   from the training patients.
3. Scalers, encoders and classifiers are fitted on training patients only; self-supervised pretraining inside a fold
   uses training patients only.
4. Confidence intervals resample patients.

## Licence

Code: MIT (`LICENSE`). Derived results in `results/`: CC BY 4.0, except the tables derived from FogAtHome
(non-commercial research terms of its source studies), which follow those terms. The
datasets and pretrained weights keep their own licences and are not redistributed here.
