# Reproducibility across hardware (post hoc)

Written by `scripts/paper_hardware.py` on 2026-10-13 at commit 42ac5ed.

| arm | kind | max_abs_diff_laptop_cpu_vs_mac_cpu | cells_laptop_cpu_vs_mac_cpu | max_abs_diff_mac_cpu_vs_mac_gpu | cells_mac_cpu_vs_mac_gpu |
|---|---|---|---|---|---|
| feat60 | frozen | 0.000 | 8 | 0.000 | 8 |
| feat60_fi | frozen | 0.000 | 8 | 0.000 | 8 |
| harnet5 | frozen | 0.000 | 8 | 0.000 | 8 |
| harnet5_rand | frozen | 0.001 | 8 | 0.000 | 8 |
| ssl_indomain | trained | 0.013 | 8 | 0.018 | 8 |
| supervised | trained | 0.074 | 8 | 0.035 | 8 |
