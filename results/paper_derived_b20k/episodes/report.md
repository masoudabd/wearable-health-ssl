# FOG episodes detected and false alarms per hour (analysis e, post hoc)

Written by `scripts/paper_episodes.py` on 2026-10-28 at commit e37fe68; labels: fixed;
5 repetitions; operating point: the stored threshold chosen on validation patients for window F1.
Assessable episodes: lab 1,793, home
1,991; analyzed hours: lab 19.7, home
24.4.

## All training patients, mean over repetitions

| cohort | arm | false_alarms_per_hour | episodes_detected | episode_precision | detected_under 1 s | detected_2.5 to 5 s |
|---|---|---|---|---|---|---|
| tdcsfog | feat60 | 31.645 | 0.751 | 0.665 | 0.424 | 0.751 |
| tdcsfog | feat60_fi | 28.237 | 0.737 | 0.690 | 0.435 | 0.738 |
| tdcsfog | harnet5 | 27.872 | 0.742 | 0.689 | 0.434 | 0.732 |
| tdcsfog | harnet5_rand | 37.051 | 0.740 | 0.633 | 0.434 | 0.732 |
| tdcsfog | ssl_indomain | 41.848 | 0.785 | 0.615 | 0.536 | 0.766 |
| tdcsfog | supervised | 23.602 | 0.674 | 0.693 | 0.370 | 0.661 |
| defog | feat60 | 46.348 | 0.583 | 0.417 | 0.348 | 0.590 |
| defog | feat60_fi | 49.937 | 0.598 | 0.409 | 0.359 | 0.632 |
| defog | harnet5 | 65.523 | 0.663 | 0.406 | 0.401 | 0.741 |
| defog | harnet5_rand | 70.475 | 0.636 | 0.308 | 0.493 | 0.663 |
| defog | ssl_indomain | 68.274 | 0.647 | 0.388 | 0.399 | 0.709 |
| defog | supervised | 42.201 | 0.515 | 0.376 | 0.350 | 0.535 |

Intervals, the budget of 10 patients and all duration bins: `episode_metrics.csv`; per patient: `false_alarms.csv`;
threshold sweep: `sweep.csv`.
