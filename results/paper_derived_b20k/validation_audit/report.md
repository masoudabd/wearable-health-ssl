# Validation patients: what they carried and what choosing on them cost (analysis j, post hoc)

Written by `scripts/paper_validation_audit.py` on 2026-10-29 at commit 482fead+uncommitted;
5 repetitions; arms harnet5, feat60_fi, ssl_indomain, supervised, harnet5_rand.

**No validation set was without a FOG window**: 0 of
1000 validation sets across every cohort, budget, fold, arm and repetition contained at
least one. The label budget is drawn until both classes are present, and a validation patient is carved from it
stratified on the label, which is why.

## What the validation patients carried

| cohort | budget | val_patients | windows_median | fog_windows_min | fog_windows_median | sets_without_fog | n_sets |
|---|---|---|---|---|---|---|---|
| defog | 10 | 1 | 789 | 3 | 86 | 0 | 125 |
| defog | 20 | 3 | 2130 | 34 | 238 | 0 | 125 |
| defog | 5 | 1 | 528 | 4 | 63 | 0 | 125 |
| defog | all | 9 | 4811 | 104 | 524 | 0 | 125 |
| tdcsfog | 10 | 2 | 595 | 6 | 83 | 0 | 125 |
| tdcsfog | 20 | 4 | 1269 | 3 | 282 | 0 | 125 |
| tdcsfog | 5 | 1 | 416 | 7 | 71 | 0 | 125 |
| tdcsfog | all | 12 | 4362 | 399 | 1267 | 0 | 125 |

## Cost of choosing the threshold on them

Positive-class F1 on the test patients at the chosen threshold, against the best any single threshold could have
reached on those test windows. The oracle is not attainable in practice; the gap sizes what the choice cost.

| cohort | budget | f1_chosen | f1_oracle | f1_gap | gap_sd |
|---|---|---|---|---|---|
| defog | 10 | 0.313 | 0.376 | 0.064 | 0.039 |
| defog | 20 | 0.365 | 0.412 | 0.047 | 0.031 |
| defog | 5 | 0.290 | 0.354 | 0.064 | 0.029 |
| defog | all | 0.401 | 0.437 | 0.036 | 0.027 |
| tdcsfog | 10 | 0.620 | 0.686 | 0.067 | 0.036 |
| tdcsfog | 20 | 0.679 | 0.717 | 0.038 | 0.024 |
| tdcsfog | 5 | 0.546 | 0.647 | 0.101 | 0.071 |
| tdcsfog | all | 0.728 | 0.757 | 0.029 | 0.021 |

## Spread of the chosen threshold over folds, as a share of the arm's score range

| cohort | budget | threshold_range_rel |
|---|---|---|
| defog | 10 | 0.691 |
| defog | 20 | 0.569 |
| defog | 5 | 0.673 |
| defog | all | 0.309 |
| tdcsfog | 10 | 0.695 |
| tdcsfog | 20 | 0.522 |
| tdcsfog | 5 | 0.856 |
| tdcsfog | all | 0.444 |
