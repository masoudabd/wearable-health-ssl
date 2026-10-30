# Episode coverage and a minimum-coverage detection rule (analysis i, post hoc)

Written by `scripts/paper_episode_strict.py` on 2026-10-29 at commit f9c674a+uncommitted; labels: fixed;
5 repetitions; 20,000 patient resamples per cohort. Coverage is the share of an annotated episode
covered by the union of the windows the detector called FOG; `any` is the rule used by analysis e.

## Episodes detected (%), all training patients

| cohort | arm | any | cov25 | cov50 | cov80 |
|---|---|---|---|---|---|
| defog | feat60_fi | 59.8 | 57.7 | 54.6 | 48.0 |
| defog | harnet5 | 66.3 | 64.0 | 59.7 | 51.6 |
| tdcsfog | feat60_fi | 73.7 | 72.3 | 69.9 | 63.1 |
| tdcsfog | harnet5 | 74.2 | 72.5 | 69.9 | 62.9 |

## Coverage of an episode, all training patients

| index | cohort | arm | coverage_mean | coverage_median | coverage_q1 | coverage_q3 | share_zero_pct | share_full_pct |
|---|---|---|---|---|---|---|---|---|
| 6 | tdcsfog | feat60 | 0.697 | 1.000 | 0.058 | 1.000 | 24.897 | 53.330 |
| 7 | tdcsfog | feat60_fi | 0.678 | 1.000 | 0.000 | 1.000 | 26.291 | 51.009 |
| 8 | tdcsfog | harnet5 | 0.678 | 0.997 | 0.000 | 1.000 | 25.800 | 49.604 |
| 9 | tdcsfog | harnet5_rand | 0.675 | 1.000 | 0.000 | 1.000 | 26.023 | 52.370 |
| 10 | tdcsfog | ssl_indomain | 0.723 | 1.000 | 0.458 | 1.000 | 21.450 | 53.397 |
| 11 | tdcsfog | supervised | 0.614 | 0.947 | 0.000 | 1.000 | 32.582 | 44.149 |
| 18 | defog | feat60 | 0.512 | 0.591 | 0.000 | 1.000 | 41.657 | 40.171 |
| 19 | defog | feat60_fi | 0.534 | 0.727 | 0.000 | 1.000 | 40.231 | 42.250 |
| 20 | defog | harnet5 | 0.583 | 0.851 | 0.000 | 1.000 | 33.712 | 44.490 |
| 21 | defog | harnet5_rand | 0.578 | 0.953 | 0.000 | 1.000 | 36.434 | 48.468 |
| 22 | defog | ssl_indomain | 0.567 | 0.809 | 0.000 | 1.000 | 35.259 | 43.596 |
| 23 | defog | supervised | 0.462 | 0.209 | 0.000 | 1.000 | 48.458 | 37.278 |

Per-rule intervals and the split by duration: `strict_metrics.csv`, `coverage_by_duration.csv`.
