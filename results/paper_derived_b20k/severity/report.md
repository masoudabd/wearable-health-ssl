# Severity across repetitions and by sex (analysis f, post hoc, exploratory)

Written by `scripts/paper_severity.py` on 2026-10-28 at commit e37fe68; labels: fixed;
5 repetitions; 20,000 patient resamples per cohort shared by repetitions and approaches.

## Contrasts in pooled AUROC, all training patients (mean over repetitions)

| cohort | arm | med_on_minus_off | nfogq_t1_minus_t3 | sex_female_minus_male | updrs3on_t1_minus_t3 |
|---|---|---|---|---|---|
| defog | feat60 | 0.050 | 0.100 | -0.007 | 0.009 |
| defog | feat60_fi | 0.034 | 0.108 | -0.004 | 0.029 |
| defog | harnet5 | 0.037 | 0.112 | -0.015 | 0.062 |
| defog | harnet5_rand | -0.042 | 0.111 | -0.066 | 0.071 |
| defog | ssl_indomain | 0.015 | 0.145 | -0.014 | 0.075 |
| defog | supervised | 0.037 | 0.095 | -0.055 | 0.093 |
| tdcsfog | feat60 | 0.024 | -0.016 | -0.055 | 0.005 |
| tdcsfog | feat60_fi | 0.026 | -0.032 | -0.063 | -0.010 |
| tdcsfog | harnet5 | 0.025 | -0.053 | -0.035 | -0.010 |
| tdcsfog | harnet5_rand | 0.006 | 0.020 | -0.028 | 0.058 |
| tdcsfog | ssl_indomain | 0.037 | -0.063 | -0.041 | -0.034 |
| tdcsfog | supervised | 0.032 | -0.047 | -0.050 | 0.005 |

Intervals: `contrasts.csv`; strata: `strata.csv`; correlations: `correlations.csv`; per patient: `per_patient.csv`.
