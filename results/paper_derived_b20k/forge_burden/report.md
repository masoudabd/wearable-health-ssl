# The released FORGE detector's freezing burden in elapsed time

Written by `scripts/paper_forge_burden.py` on 2026-10-29 at commit 5d34031+uncommitted;
20,000 patient resamples; 57 home patients, out of fold under the authors' own folds.

Percent time frozen here is elapsed FOG time over analyzed time, as analysis h defines it, so these rows can be read
against the paper's arms. The detector's own report measures the same quantity as a share of windows; both are shown.
This is not a paired comparison: FORGE's folds, its 19 training patients per head and its pretraining corpus are its
own, and its encoder was pretrained on the unlabeled recordings of 45 of these 57 patients.

| ensemble | threshold | ptf_true_time | ptf_pred_time | icc_time | mae_time | bias_time | ptf_pred_window |
|---|---|---|---|---|---|---|---|
| mean3 | released_0.35 | 12.772 | 31.035 | 0.393 | 18.312 | 18.263 | 27.153 |
| mean3 | validation_f1 | 12.772 | 28.830 | 0.423 | 16.736 | 16.058 | 25.053 |
| s42 | released_0.35 | 12.772 | 25.369 | 0.500 | 13.505 | 12.597 | 21.452 |
| s42 | validation_f1 | 12.772 | 30.542 | 0.393 | 18.158 | 17.770 | 26.638 |
