# Detection by FOG manifestation and episode length (analysis g, post hoc, exploratory)

Written by `scripts/paper_fog_types.py` on 2026-10-28 at commit e37fe68; labels: fixed;
5 repetitions.

## Window-level AUROC of each kind of FOG window against all non-FOG windows, all training patients

| cohort | arm | akinetic | all_fog | class_no_class | class_start_hesitation | class_turn | class_walking | kinetic |
|---|---|---|---|---|---|---|---|---|
| defog | feat60 | 0.820 | 0.827 | 0.820 | nan | 0.839 | 0.794 | 0.817 |
| defog | feat60_fi | 0.846 | 0.841 | 0.830 | nan | 0.853 | 0.813 | 0.834 |
| defog | harnet5 | 0.827 | 0.842 | 0.825 | nan | 0.859 | 0.803 | 0.838 |
| defog | harnet5_rand | 0.696 | 0.728 | 0.703 | nan | 0.729 | 0.772 | 0.689 |
| defog | ssl_indomain | 0.807 | 0.814 | 0.767 | nan | 0.836 | 0.808 | 0.821 |
| defog | supervised | 0.720 | 0.769 | 0.740 | nan | 0.781 | 0.773 | 0.770 |
| tdcsfog | feat60 | 0.883 | 0.915 | nan | 0.931 | 0.912 | 0.921 | 0.928 |
| tdcsfog | feat60_fi | 0.858 | 0.932 | nan | 0.947 | 0.930 | 0.931 | 0.940 |
| tdcsfog | harnet5 | 0.847 | 0.943 | nan | 0.955 | 0.942 | 0.946 | 0.948 |
| tdcsfog | harnet5_rand | 0.844 | 0.868 | nan | 0.873 | 0.870 | 0.839 | 0.869 |
| tdcsfog | ssl_indomain | 0.829 | 0.916 | nan | 0.937 | 0.913 | 0.913 | 0.927 |
| tdcsfog | supervised | 0.850 | 0.918 | nan | 0.946 | 0.915 | 0.916 | 0.926 |

## Share of assessable episodes detected, all training patients (stored operating point)

| cohort | grouping | group | feat60_fi | harnet5 |
|---|---|---|---|---|
| defog | class | no_class | 0.654 | 0.741 |
| defog | class | start_hesitation | 0.000 | 0.000 |
| defog | class | turn | 0.621 | 0.695 |
| defog | class | walking | 0.473 | 0.487 |
| defog | duration | 1 to 2.5 s | 0.504 | 0.560 |
| defog | duration | 10 to 30 s | 0.903 | 0.957 |
| defog | duration | 2.5 to 5 s | 0.632 | 0.741 |
| defog | duration | 30 s or longer | 0.964 | 1.000 |
| defog | duration | 5 to 10 s | 0.800 | 0.877 |
| defog | duration | under 1 s | 0.359 | 0.401 |
| defog | kinetic | akinetic | 0.738 | 0.783 |
| defog | kinetic | kinetic | 0.583 | 0.648 |
| defog | kinetic | kinetic flag unknown | 0.575 | 0.645 |
| tdcsfog | class | start_hesitation | 0.599 | 0.591 |
| tdcsfog | class | turn | 0.741 | 0.755 |
| tdcsfog | class | walking | 0.821 | 0.769 |
| tdcsfog | duration | 1 to 2.5 s | 0.553 | 0.568 |
| tdcsfog | duration | 10 to 30 s | 0.956 | 0.961 |
| tdcsfog | duration | 2.5 to 5 s | 0.738 | 0.732 |
| tdcsfog | duration | 30 s or longer | 0.997 | 0.995 |
| tdcsfog | duration | 5 to 10 s | 0.907 | 0.911 |
| tdcsfog | duration | under 1 s | 0.435 | 0.434 |
| tdcsfog | kinetic | akinetic | 0.685 | 0.662 |
| tdcsfog | kinetic | kinetic | 0.754 | 0.757 |
| tdcsfog | kinetic | kinetic flag unknown | 0.713 | 0.723 |

Intervals and counts: `window_auroc.csv`, `episode_detection.csv`.
