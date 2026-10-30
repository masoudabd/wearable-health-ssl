# Percent time frozen in elapsed time (analysis h, post hoc)

Written by `scripts/paper_ptf_time.py` on 2026-10-29 at commit 69c88e9+uncommitted; labels: fixed;
5 repetitions; 20,000 patient resamples per cohort. Percent time frozen is elapsed FOG time over
analyzed time, with overlapping windows and episodes merged exactly, in place of the share of windows used by
analysis b. ICC is ICC(A,1); MAE and bias in percentage points (bias = predicted minus annotated).

## All training patients, mean over repetitions

| cohort | arm | bias | icc | mae | rho |
|---|---|---|---|---|---|
| defog | feat60 | 13.801 | 0.387 | 16.703 | 0.633 |
| defog | feat60_fi | 13.743 | 0.431 | 16.023 | 0.638 |
| defog | harnet5 | 16.313 | 0.437 | 16.733 | 0.758 |
| defog | harnet5_rand | 29.358 | 0.147 | 32.250 | 0.342 |
| defog | ssl_indomain | 18.460 | 0.357 | 19.078 | 0.708 |
| defog | supervised | 15.694 | 0.245 | 21.207 | 0.424 |
| tdcsfog | feat60 | 14.455 | 0.665 | 15.880 | 0.756 |
| tdcsfog | feat60_fi | 11.637 | 0.725 | 13.676 | 0.760 |
| tdcsfog | harnet5 | 9.219 | 0.830 | 11.222 | 0.841 |
| tdcsfog | harnet5_rand | 16.524 | 0.514 | 20.475 | 0.585 |
| tdcsfog | ssl_indomain | 15.160 | 0.693 | 15.942 | 0.764 |
| tdcsfog | supervised | 8.906 | 0.754 | 12.602 | 0.739 |

## The two definitions side by side, all training patients

| index | cohort | arm | ptf_true_time | ptf_pred_time | ptf_true_window | ptf_pred_window | bias_time | bias_window |
|---|---|---|---|---|---|---|---|---|
| 18 | defog | feat60 | 12.77 | 26.57 | 13.17 | 22.98 | 13.80 | 9.81 |
| 19 | defog | feat60_fi | 12.77 | 26.52 | 13.17 | 22.63 | 13.74 | 9.46 |
| 20 | defog | harnet5 | 12.77 | 29.09 | 13.17 | 23.78 | 16.31 | 10.61 |
| 21 | defog | harnet5_rand | 12.77 | 42.13 | 13.17 | 38.51 | 29.36 | 25.34 |
| 22 | defog | ssl_indomain | 12.77 | 31.23 | 13.17 | 25.97 | 18.46 | 12.80 |
| 23 | defog | supervised | 12.77 | 28.47 | 13.17 | 26.13 | 15.69 | 12.96 |
| 42 | tdcsfog | feat60 | 23.10 | 37.55 | 22.89 | 32.15 | 14.45 | 9.26 |
| 43 | tdcsfog | feat60_fi | 23.10 | 34.74 | 22.89 | 29.36 | 11.64 | 6.47 |
| 44 | tdcsfog | harnet5 | 23.10 | 32.32 | 22.89 | 26.60 | 9.22 | 3.71 |
| 45 | tdcsfog | harnet5_rand | 23.10 | 39.62 | 22.89 | 33.97 | 16.52 | 11.08 |
| 46 | tdcsfog | ssl_indomain | 23.10 | 38.26 | 22.89 | 31.52 | 15.16 | 8.63 |
| 47 | tdcsfog | supervised | 23.10 | 32.00 | 22.89 | 27.45 | 8.91 | 4.56 |

Intervals, other budgets and paired differences: `agreement_time.csv`, `paired_time.csv`.
