# Check of the notype labels (home cohort)

Written by `scripts/paper_check_notype_labels.py` on 2026-10-03 21:57 at commit 2152f68. Read-only check; no data,
cache or result was changed.

## Raw files

- 46 files under `train/notype`. Their Event column flags 5,006 samples as FOG
  (0.83 min; 0.70 min where Valid and Task are true).
- `events.csv` lists 80.9 min of FOG for the same files
  (50.8 min where Valid and Task are true).
- Of the flagged samples, 0 fall inside the `events.csv` intervals read at
  100 Hz, and 5,006 fall inside the same intervals read as row indices. The
  last flagged row in any file is row 4,437 (44 s).
  So the Event column holds the event times in seconds written as row numbers.
- In the typed files (`train/defog`, first 30), the label columns agree with `events.csv` at 100 Hz in
  99.99% of samples.

## Cached windows of the home cohort (5 s at 30 Hz, hop 2.5 s)

- 11,371 of 31,894 home windows (35.7%) come from
  notype recordings, from 25 of 57 patients.
- FOG windows among them: 0 in the cache; 1,172 when
  labeled from `events.csv` with the loader's rule (FOG covers at least half of the window).
- Home cohort: 3,194 FOG windows (10.0%) in the cache;
  4,366 (13.7%) after correction;
  1,172 windows change label.
- Patients without any FOG window: 11 in the cache, 11
  after correction. Mean annotated percent time frozen of the 25 affected patients:
  4.7% in the cache, 8.7%
  after correction.

## Stored scores of `results/rq12_fog_transfer_mps` (seed 0, commit 0a4f88d) against corrected test labels

Indicative only: the models were trained on the uncorrected labels.

| Budget | Arm | AUROC, cached labels | AUROC, corrected labels | AUPRC, cached | AUPRC, corrected | Mean score, notype windows that are FOG | Mean score, notype windows that are not FOG |
|---|---|---|---|---|---|---|---|
| 10 | feat60_fi | 0.732 | 0.745 | 0.197 | 0.313 | 0.59 | 0.18 |
| 10 | harnet5 | 0.747 | 0.735 | 0.213 | 0.281 | 0.35 | 0.12 |
| 10 | ssl_indomain | 0.783 | 0.766 | 0.229 | 0.292 | 0.49 | 0.21 |
| 10 | supervised | 0.691 | 0.672 | 0.169 | 0.203 | 0.19 | 0.13 |
| all | feat60_fi | 0.833 | 0.832 | 0.312 | 0.423 | 0.61 | 0.20 |
| all | harnet5 | 0.832 | 0.834 | 0.303 | 0.401 | 0.60 | 0.19 |
| all | ssl_indomain | 0.829 | 0.821 | 0.300 | 0.400 | 0.59 | 0.24 |
| all | supervised | 0.776 | 0.774 | 0.232 | 0.310 | 0.52 | 0.21 |

## What it means

Every home-cohort result computed from the current window cache is provisional: the label budget experiments (RQ1
and RQ2), the severity analysis (RQ3) and the daily-living pretraining experiment (RQ4). The lab cohort is not
affected. Fix: label the notype recordings from `events.csv` (Init and Completion are in seconds), rebuild the
window cache, and rerun the home cohort. The cached features and embeddings do not depend on the labels.
