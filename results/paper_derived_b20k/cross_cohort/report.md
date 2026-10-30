# Cross-cohort transfer, summary (post hoc)

Written by `scripts/paper_cross_cohort.py` on 2026-10-28 at commit e37fe68. Pooled AUROC on the
target cohort, mean over five repetitions; within-cohort reference from the RQ1/RQ2 runs of the same seeds.

| mode | source | budget | base_arm | transfer_auroc_mean | within_auroc_mean | within_minus_transfer |
|---|---|---|---|---|---|---|
| ap | defog | 10 | feat60_fi | 0.790 | 0.901 | 0.110 |
| ap | defog | 10 | harnet5 | 0.804 | 0.902 | 0.098 |
| ap | defog | 20 | feat60_fi | 0.847 | 0.914 | 0.067 |
| ap | defog | 20 | harnet5 | 0.871 | 0.923 | 0.051 |
| ap | defog | 5 | feat60_fi | 0.784 | 0.871 | 0.087 |
| ap | defog | 5 | harnet5 | 0.772 | 0.868 | 0.095 |
| ap | defog | all | feat60_fi | 0.892 | 0.932 | 0.039 |
| ap | defog | all | harnet5 | 0.915 | 0.943 | 0.028 |
| ap | tdcsfog | 10 | feat60_fi | 0.781 | 0.761 | -0.020 |
| ap | tdcsfog | 10 | harnet5 | 0.795 | 0.734 | -0.061 |
| ap | tdcsfog | 20 | feat60_fi | 0.783 | 0.829 | 0.046 |
| ap | tdcsfog | 20 | harnet5 | 0.823 | 0.820 | -0.004 |
| ap | tdcsfog | 5 | feat60_fi | 0.768 | 0.693 | -0.074 |
| ap | tdcsfog | 5 | harnet5 | 0.789 | 0.716 | -0.073 |
| ap | tdcsfog | all | feat60_fi | 0.827 | 0.841 | 0.014 |
| ap | tdcsfog | all | harnet5 | 0.846 | 0.842 | -0.004 |
| none | defog | 10 | feat60_fi | 0.605 | 0.901 | 0.296 |
| none | defog | 10 | harnet5 | 0.783 | 0.902 | 0.119 |
| none | defog | 20 | feat60_fi | 0.678 | 0.914 | 0.236 |
| none | defog | 20 | harnet5 | 0.870 | 0.923 | 0.053 |
| none | defog | 5 | feat60_fi | 0.742 | 0.871 | 0.129 |
| none | defog | 5 | harnet5 | 0.756 | 0.868 | 0.111 |
| none | defog | all | feat60_fi | 0.680 | 0.932 | 0.252 |
| none | defog | all | harnet5 | 0.905 | 0.943 | 0.039 |
| none | tdcsfog | 10 | feat60_fi | 0.500 | 0.761 | 0.261 |
| none | tdcsfog | 10 | harnet5 | 0.797 | 0.734 | -0.063 |
| none | tdcsfog | 20 | feat60_fi | 0.465 | 0.829 | 0.363 |
| none | tdcsfog | 20 | harnet5 | 0.820 | 0.820 | -0.001 |
| none | tdcsfog | 5 | feat60_fi | 0.520 | 0.693 | 0.174 |
| none | tdcsfog | 5 | harnet5 | 0.783 | 0.716 | -0.067 |
| none | tdcsfog | all | feat60_fi | 0.597 | 0.841 | 0.244 |
| none | tdcsfog | all | harnet5 | 0.845 | 0.842 | -0.003 |
