# Calibration of window probabilities (analysis d, post hoc)

Written by `scripts/paper_calibration.py` on 2026-10-28 at commit e37fe68; labels: fixed;
5 repetitions. Calibration in the large is the mean predicted probability minus the prevalence.

## All training patients, mean over repetitions

| cohort | arm | prevalence | brier | citl | intercept | slope |
|---|---|---|---|---|---|---|
| defog | feat60 | 0.137 | 0.166 | 0.210 | -1.678 | 0.758 |
| defog | feat60_fi | 0.137 | 0.157 | 0.192 | -1.632 | 0.740 |
| defog | harnet5 | 0.137 | 0.149 | 0.161 | -1.481 | 0.601 |
| defog | harnet5_rand | 0.137 | 0.177 | 0.165 | -1.494 | 0.256 |
| defog | ssl_indomain | 0.137 | 0.173 | 0.216 | -1.663 | 0.762 |
| defog | supervised | 0.137 | 0.165 | 0.141 | -1.341 | 0.383 |
| tdcsfog | feat60 | 0.348 | 0.113 | 0.049 | -0.419 | 0.856 |
| tdcsfog | feat60_fi | 0.348 | 0.101 | 0.045 | -0.454 | 0.881 |
| tdcsfog | harnet5 | 0.348 | 0.092 | 0.040 | -0.485 | 0.784 |
| tdcsfog | harnet5_rand | 0.348 | 0.140 | 0.010 | -0.143 | 0.554 |
| tdcsfog | ssl_indomain | 0.348 | 0.118 | 0.070 | -0.630 | 0.953 |
| tdcsfog | supervised | 0.348 | 0.109 | 0.007 | -0.128 | 0.685 |

Intervals and the budget of 10 patients: `calibration.csv`; reliability bins: `reliability.csv`.
