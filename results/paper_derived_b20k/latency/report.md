# Detection latency and flags in time (analysis k, post hoc)

Written by `scripts/paper_latency.py` on 2026-10-29 at commit 329c380; labels: fixed;
5 repetitions; 20,000 patient resamples per cohort. A window's flag exists at its end (start + 5 s);
latency runs from episode onset to the first flag of a window overlapping the episode; "in time" means the flag
exists before the episode ends. Long episodes last at least 5 s. References: every overlapping window called
FOG (windows), exactly the windows labeled FOG (labels). episodes detected reproduce analysis e (fixed runs) and the operating-point analysis (every rule); F1 flags identical between run sets for the shared approaches.

## Budget all (and the references)

| cohort | runs | arm | rule | detected_pct | in_time_pct | in_time_long_pct | in_time_of_detected_pct | latency_median_s | latency_mean_s | latency_long_median_s |
|---|---|---|---|---|---|---|---|---|---|---|
| tdcsfog | reference | windows | reference | 100.0 | 82.6 | 100.0 | 82.6 | 1.3 | 1.3 | 1.3 |
| tdcsfog | reference | labels | reference | 68.4 | 52.6 | 100.0 | 77.0 | 3.3 | 3.1 | 3.3 |
| tdcsfog | fixed | feat60_fi | f1 | 73.7 | 55.3 | 91.7 | 75.0 | 2.2 | 2.9 | 2.5 |
| tdcsfog | fixed | harnet5 | f1 | 74.2 | 54.7 | 91.3 | 73.7 | 2.3 | 2.9 | 2.5 |
| tdcsfog | equal_effort | harnet5 | fa8 | 49.2 | 36.5 | 75.2 | 74.4 | 3.8 | 4.4 | 4.4 |
| tdcsfog | equal_effort | harnet5 | burden_time | 61.3 | 44.8 | 84.2 | 73.0 | 3.0 | 3.7 | 3.7 |
| tdcsfog | equal_effort | feat60_fi | fa8 | 48.4 | 37.0 | 74.6 | 76.7 | 3.8 | 5.0 | 4.5 |
| tdcsfog | equal_effort | feat60_fi | burden_time | 59.8 | 44.8 | 83.4 | 74.9 | 3.1 | 4.0 | 3.9 |
| defog | reference | windows | reference | 100.0 | 68.4 | 99.7 | 68.4 | 1.4 | 1.5 | 1.4 |
| defog | reference | labels | reference | 61.9 | 44.5 | 99.7 | 71.9 | 3.0 | 2.9 | 2.9 |
| defog | fixed | feat60_fi | f1 | 59.8 | 41.1 | 82.0 | 68.8 | 1.9 | 2.6 | 2.0 |
| defog | fixed | harnet5 | f1 | 66.3 | 44.1 | 88.2 | 66.6 | 2.0 | 2.6 | 2.1 |
| defog | equal_effort | harnet5 | fa8 | 16.8 | 11.7 | 31.3 | 69.3 | 5.0 | 6.6 | 6.2 |
| defog | equal_effort | harnet5 | burden_time | 39.0 | 26.9 | 63.5 | 69.1 | 3.2 | 4.2 | 4.1 |
| defog | equal_effort | feat60_fi | fa8 | 16.4 | 12.0 | 30.8 | 73.2 | 3.8 | 5.8 | 5.0 |
| defog | equal_effort | feat60_fi | burden_time | 36.3 | 25.7 | 59.3 | 71.0 | 2.4 | 3.8 | 3.1 |

Every approach, budget and rule with intervals: `latency.csv`; duration bands: `latency_by_duration.csv`.
