# The released FORGE detector on our home cohort (DeFOG, 57 patients)

Written by `scripts/forge_detector.py` on 2026-10-29 at commit c0907f0;
20,000 patient resamples; device mps; 388 s.

## Fold alignment

The nine released heads (`mc_probe`, seeds 42, 43, 44 x folds 0, 1, 2) each name their split in their safetensors
metadata and in `release/manifest.yaml`; the split files list participant codes. The three folds are a
19 / 19 / 19 participant partition of exactly our 57 home
patients, train, val and test disjoint within each fold. Each patient is scored only by the heads of the fold that
holds it in `test`, giving one out-of-fold score per window for all 57 patients, which is the authors'
own DeFOG protocol. Patient by patient in `folds.csv`.

## Pretraining overlap

FORGE's self-supervised pretraining split (`configs/data/splits/kaggle_daily/ssl_pretraining.yaml`) lists
65 participant codes; as a set they are
identical to the
65 participants of the contest's daily-living release. So the encoder
was pretrained on the unlabeled at-home recordings of 45 of our
57 home patients, including 45 of the
57 out-of-fold test patients
(fold 0: 15 of 19, fold 1: 13 of 19, fold 2: 17 of 19).
The labels of the test patients were never used, but their signals were, which this project's own rules forbid
inside a fold. The comparison below therefore favours FORGE.

## Out-of-fold discrimination, authors' reading and our window labels

| source | arm | ensemble | reading | labels | auroc | auroc_lo | auroc_hi | auprc | auprc_lo | auprc_hi |
|---|---|---|---|---|---|---|---|---|---|---|
| forge | forge_mc_probe | mean3 | softmax | ours_majority | 0.835 | 0.786 | 0.876 | 0.412 | 0.278 | 0.504 |
| forge | forge_mc_probe | mean3 | softmax | forge_any_fog | 0.809 | 0.767 | 0.845 | 0.508 | 0.384 | 0.599 |
| forge | forge_mc_probe | mean3 | sigmoid | ours_majority | 0.834 | 0.786 | 0.876 | 0.411 | 0.278 | 0.502 |
| forge | forge_mc_probe | mean3 | sigmoid | forge_any_fog | 0.808 | 0.766 | 0.844 | 0.506 | 0.383 | 0.597 |
| forge | forge_mc_probe | s42 | softmax | ours_majority | 0.836 | 0.789 | 0.877 | 0.411 | 0.280 | 0.502 |
| forge | forge_mc_probe | s42 | softmax | forge_any_fog | 0.810 | 0.769 | 0.846 | 0.506 | 0.384 | 0.597 |
| forge | forge_mc_probe | s42 | sigmoid | ours_majority | 0.835 | 0.789 | 0.875 | 0.409 | 0.278 | 0.500 |
| forge | forge_mc_probe | s42 | sigmoid | forge_any_fog | 0.808 | 0.768 | 0.843 | 0.504 | 0.383 | 0.595 |
| ours | harnet5 |  |  |  | 0.842 | 0.804 | 0.876 | 0.419 | 0.307 | 0.507 |
| ours | feat60_fi |  |  |  | 0.841 | 0.799 | 0.876 | 0.438 | 0.315 | 0.528 |
| ours | ssl_indomain |  |  |  | 0.814 | 0.767 | 0.854 | 0.368 | 0.266 | 0.460 |
| ours | supervised |  |  |  | 0.769 | 0.711 | 0.819 | 0.308 | 0.213 | 0.401 |
| ours | harnet5_tc (equal effort) |  |  |  | 0.851 | 0.811 | 0.885 | 0.438 | 0.323 | 0.525 |
| ours | feat60_fi_tc (equal effort) |  |  |  | 0.829 | 0.779 | 0.869 | 0.438 | 0.312 | 0.530 |
| ours | harnet5_ft (equal effort) |  |  |  | 0.823 | 0.777 | 0.865 | 0.426 | 0.307 | 0.516 |

The arms marked `ours` come from this project's registered home runs (five repetitions, all training patients,
out of fold) and are printed for context only. They are **not** a paired comparison: FORGE's folds are its own,
its heads are trained on 19 patients each against about 45 for ours, and its encoder saw most of these patients
during pretraining. No paired test is reported.

These numbers are not the model card's in-distribution reference of AP 0.730 on the DeFOG validation folds, and
are not meant to be. The windows differ: FORGE cuts 5 s patches every 2 s and every 1 s inside FOG regions and
calls a patch positive if any sample is FOG, which gives the 24.8% FOG share its own fold 0 split file records,
while our windows run every 2.5 s with a majority label and are kept only when every sample is valid
(fold 0: 10.2% majority, 17.8% any FOG, fold 1: 14.7% majority, 22.1% any FOG, fold 2: 16.7% majority, 26.0% any FOG). Average
precision rises with the positive share, which is most of the difference; the card's figure is also on validation
patients rather than the out-of-fold test patients used here.

## At an operating point

| ensemble | threshold | threshold_value | f1_pos | f1_macro | sensitivity | specificity | ptf_icc | ptf_mae_pp | ptf_bias_pp |
|---|---|---|---|---|---|---|---|---|---|
| mean3 | released_0.35 | 0.35 | 0.481 | 0.677 | 0.693 | 0.811 | 0.497 | 14.866 | 13.984 |
| mean3 | validation_f1 | fold0=0.404; fold1=0.362; fold2=0.361 | 0.479 | 0.680 | 0.653 | 0.830 | 0.525 | 13.762 | 11.868 |
| s42 | released_0.35 | 0.35 | 0.476 | 0.684 | 0.596 | 0.857 | 0.597 | 11.176 | 8.283 |
| s42 | validation_f1 | fold0=0.317; fold1=0.337; fold2=0.240 | 0.480 | 0.677 | 0.677 | 0.818 | 0.493 | 15.182 | 13.458 |

Percent time frozen is the share of a patient's analysed windows classified FOG, against the annotated share;
ICC is ICC(A,1), MAE and bias in percentage points. Intervals: `threshold_metrics.csv`.

## What this gives the paper

A published, FOG-specific detector with released weights, trained by its own authors on one of our own cohorts and
run here under its own fold structure, lands in the same band as our arms on window discrimination while
overestimating each patient's freezing burden by 14
percentage points at its released operating point. The state-of-the-art gap an external review raised is therefore
answerable, and the answer points the same way as the paper's own finding: window-level discrimination near 0.84
AUROC does not deliver reliable measurement of freezing at home.

## Files

`summary.csv` (every scoring variant), `threshold_metrics.csv`, `per_fold.csv` (one row per head),
`per_patient.csv`, `per_window_scores.parquet` (every head's score for its fold's test and validation patients),
`folds.csv`, `released_heads.csv`, `overlap.json`.
