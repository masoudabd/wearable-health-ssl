# Patient-bootstrap intervals over the repetitions (analysis a, post hoc)

Written by `scripts/paper_intervals.py` on 2026-10-28 at commit e37fe68; labels: fixed;
5 repetitions; 20,000 patient resamples per cohort shared by repetitions, approaches and budgets.
Check: 48 AUROC tests compared with results/rq12_seeds_mps_fixed/paired_tests.csv: largest deviation 3.00e-03.

## Pooled AUROC and AUPRC, mean over repetitions (95% patient-bootstrap interval), budgets 10 and all

| Cohort | Budget | Approach | AUROC | AUPRC | Prevalence |
|---|---|---|---|---|---|
| defog | 10 | ssl_indomain | 0.762 (0.722 to 0.801) | 0.300 (0.221 to 0.374) | 0.137 |
| defog | 10 | feat60_fi | 0.761 (0.714 to 0.804) | 0.308 (0.224 to 0.390) | 0.137 |
| defog | 10 | feat60 | 0.746 (0.695 to 0.792) | 0.288 (0.205 to 0.372) | 0.137 |
| defog | 10 | harnet5 | 0.734 (0.696 to 0.771) | 0.287 (0.210 to 0.358) | 0.137 |
| defog | 10 | supervised | 0.693 (0.634 to 0.748) | 0.240 (0.155 to 0.331) | 0.137 |
| defog | 10 | harnet5_rand | 0.656 (0.586 to 0.719) | 0.216 (0.140 to 0.303) | 0.137 |
| defog | all | harnet5 | 0.842 (0.804 to 0.876) | 0.419 (0.307 to 0.507) | 0.137 |
| defog | all | feat60_fi | 0.841 (0.799 to 0.876) | 0.438 (0.315 to 0.528) | 0.137 |
| defog | all | feat60 | 0.827 (0.783 to 0.864) | 0.408 (0.296 to 0.496) | 0.137 |
| defog | all | ssl_indomain | 0.814 (0.767 to 0.854) | 0.368 (0.266 to 0.460) | 0.137 |
| defog | all | supervised | 0.769 (0.711 to 0.819) | 0.308 (0.213 to 0.401) | 0.137 |
| defog | all | harnet5_rand | 0.728 (0.660 to 0.785) | 0.266 (0.176 to 0.357) | 0.137 |
| tdcsfog | 10 | harnet5 | 0.902 (0.858 to 0.929) | 0.825 (0.632 to 0.915) | 0.348 |
| tdcsfog | 10 | feat60_fi | 0.901 (0.846 to 0.934) | 0.835 (0.611 to 0.928) | 0.348 |
| tdcsfog | 10 | ssl_indomain | 0.887 (0.845 to 0.912) | 0.791 (0.584 to 0.892) | 0.348 |
| tdcsfog | 10 | feat60 | 0.877 (0.818 to 0.915) | 0.782 (0.536 to 0.899) | 0.348 |
| tdcsfog | 10 | supervised | 0.762 (0.730 to 0.809) | 0.647 (0.484 to 0.757) | 0.348 |
| tdcsfog | 10 | harnet5_rand | 0.750 (0.712 to 0.802) | 0.622 (0.464 to 0.731) | 0.348 |
| tdcsfog | all | harnet5 | 0.943 (0.907 to 0.964) | 0.903 (0.753 to 0.960) | 0.348 |
| tdcsfog | all | feat60_fi | 0.932 (0.895 to 0.954) | 0.870 (0.701 to 0.941) | 0.348 |
| tdcsfog | all | supervised | 0.918 (0.875 to 0.942) | 0.845 (0.666 to 0.926) | 0.348 |
| tdcsfog | all | ssl_indomain | 0.916 (0.874 to 0.940) | 0.841 (0.644 to 0.926) | 0.348 |
| tdcsfog | all | feat60 | 0.915 (0.876 to 0.940) | 0.816 (0.622 to 0.913) | 0.348 |
| tdcsfog | all | harnet5_rand | 0.868 (0.843 to 0.895) | 0.752 (0.586 to 0.850) | 0.348 |

## Lab minus home cohort, AUROC (independent resamples)

| Budget | Approach | Difference (95% interval) |
|---|---|---|
| 5 | harnet5 | +0.152 (+0.087 to +0.203) |
| 5 | feat60_fi | +0.177 (+0.105 to +0.232) |
| 5 | ssl_indomain | +0.148 (+0.079 to +0.202) |
| 5 | supervised | +0.031 (-0.031 to +0.114) |
| 5 | harnet5_rand | +0.115 (+0.045 to +0.182) |
| 5 | feat60 | +0.177 (+0.104 to +0.233) |
| 10 | harnet5 | +0.168 (+0.109 to +0.215) |
| 10 | feat60_fi | +0.140 (+0.068 to +0.197) |
| 10 | ssl_indomain | +0.125 (+0.067 to +0.172) |
| 10 | supervised | +0.070 (+0.008 to +0.146) |
| 10 | harnet5_rand | +0.093 (+0.022 to +0.185) |
| 10 | feat60 | +0.131 (+0.053 to +0.195) |
| 20 | harnet5 | +0.103 (+0.049 to +0.149) |
| 20 | feat60_fi | +0.085 (+0.028 to +0.136) |
| 20 | ssl_indomain | +0.105 (+0.042 to +0.158) |
| 20 | supervised | +0.117 (+0.059 to +0.181) |
| 20 | harnet5_rand | +0.107 (+0.029 to +0.212) |
| 20 | feat60 | +0.083 (+0.023 to +0.134) |
| all | harnet5 | +0.101 (+0.050 to +0.145) |
| all | feat60_fi | +0.090 (+0.039 to +0.138) |
| all | ssl_indomain | +0.102 (+0.043 to +0.153) |
| all | supervised | +0.149 (+0.083 to +0.211) |
| all | harnet5_rand | +0.139 (+0.077 to +0.214) |
| all | feat60 | +0.088 (+0.033 to +0.138) |

Other tables: `paired_auprc.csv`, `budget_differences.csv`, `paired_auroc_check.csv`, `cohort_gap.csv` (AUPRC rows).
