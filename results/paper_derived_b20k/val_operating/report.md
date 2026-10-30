# Operating points chosen on validation patients (post hoc)

Written by `scripts/paper_val_operating.py` on 2026-10-29 at commit 3df4ee5; labels: fixed;
5 repetitions of the equal-effort runs; 20,000 patient resamples per cohort. Rules: f1 (stored
threshold, analysis e), fa4/fa8/fa16 (validation false alarms per hour at most 4, 8, 16), burden (validation share of
windows predicted equals share annotated), burden_time (validation predicted FOG time equals annotated FOG time,
elapsed time over analyzed time). Percent time frozen in elapsed time (ptf_time_*, analysis h's definition, the
reported endpoint) and as a share of windows (ptf_*, analysis b, for comparison); bias and MAE in percentage points.
F1 rule reproduces analysis h in 108 values (points and intervals; largest difference 4.8e-05, within the six significant digits h stores).

## All training patients, mean over repetitions

| cohort | arm | rule | false_alarms_per_hour | episodes_detected | episode_precision | ptf_time_bias | ptf_time_mae | ptf_time_icc | ptf_bias | ptf_icc |
|---|---|---|---|---|---|---|---|---|---|---|
| tdcsfog | harnet5 | f1 | 27.872 | 0.742 | 0.689 | 9.219 | 11.222 | 0.830 | 3.714 | 0.887 |
| tdcsfog | harnet5 | fa4 | 3.986 | 0.380 | 0.945 | -9.953 | 10.743 | 0.761 | -13.215 | 0.615 |
| tdcsfog | harnet5 | fa8 | 7.577 | 0.492 | 0.900 | -4.760 | 7.336 | 0.886 | -8.693 | 0.792 |
| tdcsfog | harnet5 | fa16 | 14.747 | 0.621 | 0.815 | 1.553 | 6.938 | 0.913 | -3.061 | 0.890 |
| tdcsfog | harnet5 | burden | 22.344 | 0.714 | 0.742 | 6.579 | 9.006 | 0.881 | 1.369 | 0.909 |
| tdcsfog | harnet5 | burden_time | 13.358 | 0.613 | 0.834 | 0.626 | 6.502 | 0.921 | -3.939 | 0.884 |
| tdcsfog | feat60_fi | f1 | 28.237 | 0.737 | 0.690 | 11.637 | 13.676 | 0.725 | 6.470 | 0.789 |
| tdcsfog | feat60_fi | fa4 | 4.392 | 0.367 | 0.940 | -10.119 | 11.991 | 0.695 | -13.322 | 0.554 |
| tdcsfog | feat60_fi | fa8 | 8.216 | 0.484 | 0.900 | -4.413 | 8.614 | 0.850 | -8.471 | 0.753 |
| tdcsfog | feat60_fi | fa16 | 14.585 | 0.595 | 0.829 | 1.836 | 8.405 | 0.863 | -2.721 | 0.841 |
| tdcsfog | feat60_fi | burden | 21.137 | 0.682 | 0.760 | 7.027 | 10.132 | 0.819 | 2.097 | 0.851 |
| tdcsfog | feat60_fi | burden_time | 14.362 | 0.598 | 0.833 | 1.540 | 8.056 | 0.875 | -3.093 | 0.850 |
| tdcsfog | ssl_indomain | f1 | 41.848 | 0.785 | 0.615 | 15.160 | 15.942 | 0.693 | 8.630 | 0.799 |
| tdcsfog | ssl_indomain | fa4 | 4.493 | 0.290 | 0.937 | -14.311 | 15.008 | 0.533 | -17.029 | 0.375 |
| tdcsfog | ssl_indomain | fa8 | 9.068 | 0.428 | 0.900 | -8.587 | 10.988 | 0.756 | -12.619 | 0.594 |
| tdcsfog | ssl_indomain | fa16 | 18.612 | 0.577 | 0.815 | -0.456 | 8.547 | 0.873 | -5.612 | 0.807 |
| tdcsfog | ssl_indomain | burden | 32.527 | 0.721 | 0.696 | 9.278 | 11.537 | 0.800 | 3.179 | 0.852 |
| tdcsfog | ssl_indomain | burden_time | 23.216 | 0.630 | 0.784 | 2.683 | 8.922 | 0.869 | -2.965 | 0.838 |
| tdcsfog | harnet5_tc | f1 | 23.318 | 0.739 | 0.719 | 7.810 | 10.335 | 0.854 | 2.865 | 0.894 |
| tdcsfog | harnet5_tc | fa4 | 3.824 | 0.411 | 0.942 | -8.744 | 9.705 | 0.802 | -11.903 | 0.682 |
| tdcsfog | harnet5_tc | fa8 | 7.972 | 0.541 | 0.888 | -3.034 | 6.793 | 0.905 | -6.870 | 0.840 |
| tdcsfog | harnet5_tc | fa16 | 14.666 | 0.653 | 0.804 | 2.464 | 7.127 | 0.912 | -1.860 | 0.905 |
| tdcsfog | harnet5_tc | burden | 19.352 | 0.719 | 0.760 | 5.801 | 8.627 | 0.890 | 1.063 | 0.912 |
| tdcsfog | harnet5_tc | burden_time | 12.191 | 0.623 | 0.839 | 0.653 | 6.633 | 0.919 | -3.612 | 0.890 |
| tdcsfog | harnet5_ft | f1 | 19.007 | 0.711 | 0.738 | 7.096 | 10.286 | 0.835 | 2.883 | 0.868 |
| tdcsfog | harnet5_ft | fa4 | 4.787 | 0.460 | 0.913 | -5.866 | 8.109 | 0.848 | -8.773 | 0.778 |
| tdcsfog | harnet5_ft | fa8 | 8.824 | 0.571 | 0.857 | -0.671 | 7.073 | 0.895 | -4.087 | 0.867 |
| tdcsfog | harnet5_ft | fa16 | 14.210 | 0.656 | 0.785 | 3.790 | 8.597 | 0.873 | 0.039 | 0.882 |
| tdcsfog | harnet5_ft | burden | 15.772 | 0.684 | 0.774 | 5.170 | 8.873 | 0.870 | 1.171 | 0.889 |
| tdcsfog | harnet5_ft | burden_time | 10.467 | 0.604 | 0.837 | 0.752 | 7.305 | 0.888 | -2.824 | 0.871 |
| defog | harnet5 | f1 | 65.523 | 0.663 | 0.406 | 16.313 | 16.733 | 0.437 | 10.608 | 0.580 |
| defog | harnet5 | fa4 | 4.779 | 0.097 | 0.640 | -11.159 | 11.209 | 0.135 | -12.228 | 0.072 |
| defog | harnet5 | fa8 | 9.575 | 0.168 | 0.616 | -9.352 | 9.694 | 0.278 | -11.071 | 0.154 |
| defog | harnet5 | fa16 | 18.699 | 0.284 | 0.585 | -5.523 | 7.331 | 0.549 | -8.386 | 0.345 |
| defog | harnet5 | burden | 42.201 | 0.491 | 0.491 | 4.816 | 7.974 | 0.701 | -0.059 | 0.711 |
| defog | harnet5 | burden_time | 30.828 | 0.390 | 0.532 | -0.241 | 6.164 | 0.744 | -4.327 | 0.603 |
| defog | feat60_fi | f1 | 49.937 | 0.598 | 0.409 | 13.743 | 16.023 | 0.431 | 9.457 | 0.519 |
| defog | feat60_fi | fa4 | 4.590 | 0.098 | 0.653 | -10.224 | 10.724 | 0.200 | -11.377 | 0.144 |
| defog | feat60_fi | fa8 | 8.762 | 0.164 | 0.619 | -8.238 | 9.641 | 0.333 | -9.828 | 0.254 |
| defog | feat60_fi | fa16 | 18.740 | 0.294 | 0.567 | -3.529 | 8.270 | 0.555 | -6.104 | 0.472 |
| defog | feat60_fi | burden | 31.033 | 0.425 | 0.494 | 3.215 | 9.792 | 0.593 | -0.140 | 0.610 |
| defog | feat60_fi | burden_time | 24.669 | 0.363 | 0.534 | -0.304 | 8.520 | 0.614 | -3.324 | 0.573 |
| defog | ssl_indomain | f1 | 68.274 | 0.647 | 0.388 | 18.460 | 19.078 | 0.357 | 12.796 | 0.458 |
| defog | ssl_indomain | fa4 | 4.517 | 0.061 | 0.551 | -11.308 | 11.398 | 0.093 | -12.291 | 0.049 |
| defog | ssl_indomain | fa8 | 10.010 | 0.146 | 0.576 | -9.155 | 9.712 | 0.232 | -10.889 | 0.127 |
| defog | ssl_indomain | fa16 | 19.470 | 0.250 | 0.546 | -5.405 | 7.588 | 0.455 | -8.208 | 0.292 |
| defog | ssl_indomain | burden | 44.533 | 0.464 | 0.465 | 5.321 | 9.458 | 0.599 | 0.455 | 0.615 |
| defog | ssl_indomain | burden_time | 32.930 | 0.372 | 0.498 | 0.074 | 7.374 | 0.634 | -3.962 | 0.521 |
| defog | harnet5_tc | f1 | 56.260 | 0.629 | 0.403 | 14.747 | 15.616 | 0.456 | 10.094 | 0.572 |
| defog | harnet5_tc | fa4 | 5.346 | 0.129 | 0.674 | -10.471 | 10.679 | 0.191 | -11.735 | 0.112 |
| defog | harnet5_tc | fa8 | 10.125 | 0.206 | 0.642 | -8.150 | 8.923 | 0.378 | -10.110 | 0.236 |
| defog | harnet5_tc | fa16 | 18.444 | 0.313 | 0.594 | -3.674 | 6.988 | 0.632 | -6.584 | 0.477 |
| defog | harnet5_tc | burden | 33.652 | 0.457 | 0.503 | 4.124 | 7.982 | 0.687 | 0.208 | 0.692 |
| defog | harnet5_tc | burden_time | 25.597 | 0.382 | 0.548 | -0.024 | 6.704 | 0.713 | -3.507 | 0.623 |
| defog | harnet5_ft | f1 | 49.501 | 0.642 | 0.419 | 14.619 | 16.033 | 0.440 | 10.568 | 0.531 |
| defog | harnet5_ft | fa4 | 5.814 | 0.121 | 0.644 | -9.919 | 10.451 | 0.217 | -11.153 | 0.151 |
| defog | harnet5_ft | fa8 | 10.183 | 0.202 | 0.626 | -7.268 | 9.046 | 0.385 | -9.068 | 0.290 |
| defog | harnet5_ft | fa16 | 19.438 | 0.336 | 0.566 | -1.596 | 8.169 | 0.575 | -4.266 | 0.513 |
| defog | harnet5_ft | burden | 28.619 | 0.446 | 0.513 | 3.974 | 8.812 | 0.637 | 0.690 | 0.653 |
| defog | harnet5_ft | burden_time | 23.634 | 0.381 | 0.541 | 0.561 | 7.793 | 0.655 | -2.494 | 0.616 |

Intervals, the paired ICC differences against the F1 rule and the budget of 10 patients: `operating_points.csv`;
per-fold thresholds, their validation false-alarm rates and the validation burden in elapsed time: `thresholds.csv`.
