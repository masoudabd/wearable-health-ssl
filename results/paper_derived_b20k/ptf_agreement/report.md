# Agreement of predicted with annotated percent time frozen (analysis b, post hoc)

Written by `scripts/paper_ptf_agreement.py` on 2026-10-28 at commit e37fe68; labels: fixed;
5 repetitions; 20,000 patient resamples per cohort. ICC is ICC(A,1); MAE and bias in percentage
points (bias = predicted minus annotated).

## All training patients, mean over repetitions

| cohort | arm | bias | icc | mae | rho |
|---|---|---|---|---|---|
| defog | feat60 | 9.814 | 0.446 | 14.500 | 0.584 |
| defog | feat60_fi | 9.457 | 0.519 | 13.393 | 0.608 |
| defog | harnet5 | 10.608 | 0.580 | 11.981 | 0.732 |
| defog | harnet5_rand | 25.344 | 0.175 | 29.456 | 0.316 |
| defog | ssl_indomain | 12.796 | 0.458 | 14.765 | 0.673 |
| defog | supervised | 12.963 | 0.274 | 19.937 | 0.397 |
| tdcsfog | feat60 | 9.260 | 0.732 | 13.071 | 0.768 |
| tdcsfog | feat60_fi | 6.470 | 0.789 | 11.121 | 0.771 |
| tdcsfog | harnet5 | 3.714 | 0.887 | 8.560 | 0.854 |
| tdcsfog | harnet5_rand | 11.083 | 0.581 | 17.299 | 0.606 |
| tdcsfog | ssl_indomain | 8.630 | 0.799 | 11.817 | 0.777 |
| tdcsfog | supervised | 4.559 | 0.798 | 11.019 | 0.750 |

Intervals, other budgets and paired differences: `agreement.csv`, `paired.csv`.
