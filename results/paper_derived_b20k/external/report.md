# External test on FogAtHome, summary (post hoc)

Written by `scripts/paper_external.py` on 2026-10-29 at commit ff79be3. Test sets: {'structured': {'participants': 12, 'windows': 804, 'sessions': 94, 'hours': 0.6236111111111111, 'fog_windows_pct': 37.06467661691542, 'fog_time_pct': 36.71224944320713}, 'daily': {'participants': 11, 'windows': 423912, 'sessions': 2814, 'hours': 296.3375, 'fog_windows_pct': 0.9072637717262074, 'fog_time_pct': 1.0640059729221905}, 'daily_walkstand': {'participants': 11, 'windows': 81314, 'windows_pct_of_daily': 19.18181131933043, 'fog_windows_pct': 2.3710554148117176}}.

Main paired comparisons (foundation model minus clinical features), Holm over the 12:

| source | axes | test | arm | auroc | auroc_lo | auroc_hi | p | p_holm_main12 |
|---|---|---|---|---|---|---|---|---|
| defog | Vflip | structured | harnet5 - feat60_fi | 0.030 | -0.007 | 0.113 | 0.153 | 0.916 |
| defog | Vflip | structured | harnet5_tc - feat60_fi_tc | 0.139 | 0.031 | 0.316 | 0.001 | 0.006 |
| defog | Vflip | daily | harnet5 - feat60_fi | 0.019 | -0.016 | 0.071 | 0.380 | 1.000 |
| defog | Vflip | daily | harnet5_tc - feat60_fi_tc | 0.118 | -0.008 | 0.306 | 0.113 | 0.794 |
| defog | Vflip | daily_walkstand | harnet5 - feat60_fi | -0.037 | -0.102 | 0.024 | 0.295 | 1.000 |
| defog | Vflip | daily_walkstand | harnet5_tc - feat60_fi_tc | 0.070 | -0.033 | 0.225 | 0.227 | 1.000 |
| tdcsfog | VAPflip | structured | harnet5 - feat60_fi | 0.114 | 0.008 | 0.311 | 0.008 | 0.076 |
| tdcsfog | VAPflip | structured | harnet5_tc - feat60_fi_tc | 0.132 | 0.013 | 0.341 | 0.004 | 0.048 |
| tdcsfog | VAPflip | daily | harnet5 - feat60_fi | 0.180 | 0.020 | 0.382 | 0.015 | 0.139 |
| tdcsfog | VAPflip | daily | harnet5_tc - feat60_fi_tc | 0.185 | 0.020 | 0.390 | 0.017 | 0.139 |
| tdcsfog | VAPflip | daily_walkstand | harnet5 - feat60_fi | 0.105 | -0.085 | 0.320 | 0.281 | 1.000 |
| tdcsfog | VAPflip | daily_walkstand | harnet5_tc - feat60_fi_tc | 0.115 | -0.079 | 0.330 | 0.248 | 1.000 |
