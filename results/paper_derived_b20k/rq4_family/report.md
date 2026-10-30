# RQ4 tests under one Holm family

Commit e37fe68. 96 tests listed in the three files, 92 unique (4 listed twice, identical). Smallest attainable Holm-adjusted P with 20,000 resamples: 0.005.

Tests with Holm-adjusted P below 0.05 over all RQ4 tests:

```
cohort budget                          comparison    experiment also_in  d_mean  ci_lo  ci_hi      p  p_holm_experiment  n_tests_experiment  p_holm_all  seeds_positive  n_seeds  n_patients
 defog     20      ssl_dailyliving - ssl_indomain           rq4     NaN  0.0326 0.0174 0.0492 0.0000             0.0014                  28      0.0046               5        5          57
 defog    all      ssl_dailyliving - ssl_indomain           rq4     NaN  0.0244 0.0095 0.0428 0.0004             0.0104                  28      0.0348               5        5          57
 defog     10           ssl_dailyliving - harnet5           rq4     NaN  0.0388 0.0202 0.0586 0.0001             0.0027                  28      0.0091               5        5          57
 defog    all           harnet5_dl - ssl_indomain           rq4     NaN  0.0264 0.0089 0.0477 0.0004             0.0104                  28      0.0348               5        5          57
 defog     20 ssl_dailyliving - ssl_indomain_long equal_compute     NaN  0.0155 0.0077 0.0235 0.0002             0.0048                  24      0.0180               5        5          57
 defog      5           ssl_dailygait - feat60_fi       walking     NaN  0.0589 0.0276 0.0882 0.0002             0.0048                  24      0.0180               5        5          57
 defog     20        ssl_dailygait - ssl_indomain       walking     NaN  0.0299 0.0145 0.0447 0.0002             0.0048                  24      0.0180               5        5          57
```

All tests:

```
cohort budget                              comparison      experiment       also_in  d_mean   ci_lo   ci_hi      p  p_holm_experiment  n_tests_experiment  p_holm_all  seeds_positive  n_seeds  n_patients
 defog      5          ssl_dailyliving - ssl_indomain             rq4           NaN  0.0059 -0.0147  0.0278 0.5577             1.0000                  28      1.0000               3        5          57
 defog     10          ssl_dailyliving - ssl_indomain             rq4           NaN  0.0105 -0.0073  0.0291 0.2379             1.0000                  28      1.0000               4        5          57
 defog     20          ssl_dailyliving - ssl_indomain             rq4           NaN  0.0326  0.0174  0.0492 0.0000             0.0014                  28      0.0046               5        5          57
 defog    all          ssl_dailyliving - ssl_indomain             rq4           NaN  0.0244  0.0095  0.0428 0.0004             0.0104                  28      0.0348               5        5          57
 defog      5             ssl_dailyliving - feat60_fi             rq4 equal_compute  0.0419  0.0122  0.0731 0.0054             0.1296                  28      0.4374               4        5          57
 defog     10             ssl_dailyliving - feat60_fi             rq4 equal_compute  0.0115 -0.0173  0.0429 0.4170             1.0000                  28      1.0000               4        5          57
 defog     20             ssl_dailyliving - feat60_fi             rq4 equal_compute  0.0024 -0.0123  0.0196 0.7576             1.0000                  28      1.0000               3        5          57
 defog    all             ssl_dailyliving - feat60_fi             rq4 equal_compute -0.0033 -0.0160  0.0131 0.6447             1.0000                  28      1.0000               1        5          57
 defog      5               ssl_dailyliving - harnet5             rq4           NaN  0.0192 -0.0042  0.0458 0.1095             1.0000                  28      1.0000               4        5          57
 defog     10               ssl_dailyliving - harnet5             rq4           NaN  0.0388  0.0202  0.0586 0.0001             0.0027                  28      0.0091               5        5          57
 defog     20               ssl_dailyliving - harnet5             rq4           NaN  0.0115  0.0024  0.0217 0.0131             0.2751                  28      0.9825               5        5          57
 defog    all               ssl_dailyliving - harnet5             rq4           NaN -0.0040 -0.0136  0.0058 0.4200             1.0000                  28      1.0000               1        5          57
 defog      5                    harnet5_dl - harnet5             rq4           NaN  0.0097 -0.0123  0.0349 0.4177             1.0000                  28      1.0000               4        5          57
 defog     10                    harnet5_dl - harnet5             rq4           NaN  0.0111 -0.0107  0.0358 0.3348             1.0000                  28      1.0000               4        5          57
 defog     20                    harnet5_dl - harnet5             rq4           NaN -0.0029 -0.0140  0.0101 0.6810             1.0000                  28      1.0000               1        5          57
 defog    all                    harnet5_dl - harnet5             rq4           NaN -0.0020 -0.0113  0.0088 0.7139             1.0000                  28      1.0000               1        5          57
 defog      5                  harnet5_dl - feat60_fi             rq4           NaN  0.0325 -0.0006  0.0628 0.0546             1.0000                  28      1.0000               4        5          57
 defog     10                  harnet5_dl - feat60_fi             rq4           NaN -0.0162 -0.0490  0.0202 0.4143             1.0000                  28      1.0000               1        5          57
 defog     20                  harnet5_dl - feat60_fi             rq4           NaN -0.0120 -0.0295  0.0092 0.2530             1.0000                  28      1.0000               0        5          57
 defog    all                  harnet5_dl - feat60_fi             rq4           NaN -0.0013 -0.0159  0.0173 0.8864             1.0000                  28      1.0000               2        5          57
 defog      5               harnet5_dl - ssl_indomain             rq4           NaN -0.0035 -0.0282  0.0205 0.7935             1.0000                  28      1.0000               3        5          57
 defog     10               harnet5_dl - ssl_indomain             rq4           NaN -0.0172 -0.0423  0.0113 0.2424             1.0000                  28      1.0000               2        5          57
 defog     20               harnet5_dl - ssl_indomain             rq4           NaN  0.0181 -0.0017  0.0411 0.0777             1.0000                  28      1.0000               5        5          57
 defog    all               harnet5_dl - ssl_indomain             rq4           NaN  0.0264  0.0089  0.0477 0.0004             0.0104                  28      0.0348               5        5          57
 defog      5                ssl_indomain - feat60_fi             rq4           NaN  0.0360  0.0014  0.0688 0.0429             0.8580                  28      1.0000               4        5          57
 defog     10                ssl_indomain - feat60_fi             rq4           NaN  0.0010 -0.0191  0.0268 0.9405             1.0000                  28      1.0000               3        5          57
 defog     20                ssl_indomain - feat60_fi             rq4           NaN -0.0302 -0.0487 -0.0093 0.0058             0.1334                  28      0.4640               0        5          57
 defog    all                ssl_indomain - feat60_fi             rq4           NaN -0.0277 -0.0454 -0.0072 0.0096             0.2112                  28      0.7488               0        5          57
 defog      5     ssl_dailyliving - ssl_indomain_long   equal_compute           NaN  0.0162 -0.0031  0.0371 0.1096             1.0000                  24      1.0000               2        5          57
 defog     10     ssl_dailyliving - ssl_indomain_long   equal_compute           NaN  0.0130  0.0003  0.0254 0.0450             0.7650                  24      1.0000               4        5          57
 defog     20     ssl_dailyliving - ssl_indomain_long   equal_compute           NaN  0.0155  0.0077  0.0235 0.0002             0.0048                  24      0.0180               5        5          57
 defog    all     ssl_dailyliving - ssl_indomain_long   equal_compute           NaN  0.0118  0.0046  0.0197 0.0015             0.0345                  24      0.1275               5        5          57
 defog      5        ssl_indomain_long - ssl_indomain   equal_compute           NaN -0.0102 -0.0272  0.0072 0.2558             1.0000                  24      1.0000               2        5          57
 defog     10        ssl_indomain_long - ssl_indomain   equal_compute           NaN -0.0025 -0.0152  0.0128 0.7371             1.0000                  24      1.0000               3        5          57
 defog     20        ssl_indomain_long - ssl_indomain   equal_compute           NaN  0.0171  0.0053  0.0292 0.0042             0.0882                  24      0.3444               4        5          57
 defog    all        ssl_indomain_long - ssl_indomain   equal_compute           NaN  0.0126  0.0006  0.0262 0.0381             0.6858                  24      1.0000               4        5          57
 defog      5    ssl_dailyliving_short - ssl_indomain   equal_compute           NaN -0.0117 -0.0294  0.0044 0.1564             1.0000                  24      1.0000               2        5          57
 defog     10    ssl_dailyliving_short - ssl_indomain   equal_compute           NaN  0.0066 -0.0106  0.0205 0.4043             1.0000                  24      1.0000               3        5          57
 defog     20    ssl_dailyliving_short - ssl_indomain   equal_compute           NaN  0.0149 -0.0050  0.0356 0.1442             1.0000                  24      1.0000               5        5          57
 defog    all    ssl_dailyliving_short - ssl_indomain   equal_compute           NaN  0.0071 -0.0103  0.0234 0.4209             1.0000                  24      1.0000               4        5          57
 defog      5 ssl_dailyliving - ssl_dailyliving_short   equal_compute           NaN  0.0176 -0.0034  0.0406 0.1030             1.0000                  24      1.0000               3        5          57
 defog     10 ssl_dailyliving - ssl_dailyliving_short   equal_compute           NaN  0.0038 -0.0158  0.0243 0.6538             1.0000                  24      1.0000               2        5          57
 defog     20 ssl_dailyliving - ssl_dailyliving_short   equal_compute           NaN  0.0177  0.0041  0.0321 0.0101             0.1919                  24      0.7700               4        5          57
 defog    all ssl_dailyliving - ssl_dailyliving_short   equal_compute           NaN  0.0173  0.0055  0.0308 0.0022             0.0484                  24      0.1826               5        5          57
 defog      5           ssl_indomain_long - feat60_fi   equal_compute           NaN  0.0258 -0.0087  0.0583 0.1381             1.0000                  24      1.0000               3        5          57
 defog     10           ssl_indomain_long - feat60_fi   equal_compute           NaN -0.0015 -0.0251  0.0274 0.9292             1.0000                  24      1.0000               3        5          57
 defog     20           ssl_indomain_long - feat60_fi   equal_compute           NaN -0.0130 -0.0278  0.0046 0.1363             1.0000                  24      1.0000               1        5          57
 defog    all           ssl_indomain_long - feat60_fi   equal_compute           NaN -0.0151 -0.0283  0.0012 0.0662             1.0000                  24      1.0000               0        5          57
 defog      5         ssl_dailygait - ssl_dailyliving         walking           NaN  0.0169  0.0007  0.0331 0.0396             0.7524                  24      1.0000               4        5          57
 defog     10         ssl_dailygait - ssl_dailyliving         walking           NaN  0.0056 -0.0054  0.0183 0.4181             1.0000                  24      1.0000               3        5          57
 defog     20         ssl_dailygait - ssl_dailyliving         walking           NaN -0.0027 -0.0120  0.0062 0.5916             1.0000                  24      1.0000               2        5          57
 defog    all         ssl_dailygait - ssl_dailyliving         walking           NaN -0.0038 -0.0125  0.0045 0.4200             1.0000                  24      1.0000               2        5          57
 defog      5             harnet5_dlgait - harnet5_dl         walking           NaN  0.0020 -0.0091  0.0125 0.7255             1.0000                  24      1.0000               2        5          57
 defog     10             harnet5_dlgait - harnet5_dl         walking           NaN  0.0019 -0.0078  0.0105 0.7108             1.0000                  24      1.0000               1        5          57
 defog     20             harnet5_dlgait - harnet5_dl         walking           NaN  0.0047 -0.0017  0.0098 0.1473             1.0000                  24      1.0000               4        5          57
 defog    all             harnet5_dlgait - harnet5_dl         walking           NaN  0.0010 -0.0047  0.0053 0.6919             1.0000                  24      1.0000               3        5          57
 defog      5                harnet5_dlgait - harnet5         walking           NaN  0.0117 -0.0101  0.0372 0.3234             1.0000                  24      1.0000               3        5          57
 defog     10                harnet5_dlgait - harnet5         walking           NaN  0.0131 -0.0046  0.0336 0.1565             1.0000                  24      1.0000               5        5          57
 defog     20                harnet5_dlgait - harnet5         walking           NaN  0.0018 -0.0077  0.0129 0.7198             1.0000                  24      1.0000               3        5          57
 defog    all                harnet5_dlgait - harnet5         walking           NaN -0.0010 -0.0098  0.0092 0.8280             1.0000                  24      1.0000               2        5          57
 defog      5               ssl_dailygait - feat60_fi         walking           NaN  0.0589  0.0276  0.0882 0.0002             0.0048                  24      0.0180               5        5          57
 defog     10               ssl_dailygait - feat60_fi         walking           NaN  0.0171 -0.0036  0.0430 0.1105             1.0000                  24      1.0000               4        5          57
 defog     20               ssl_dailygait - feat60_fi         walking           NaN -0.0003 -0.0140  0.0178 0.9643             1.0000                  24      1.0000               2        5          57
 defog    all               ssl_dailygait - feat60_fi         walking           NaN -0.0071 -0.0191  0.0100 0.3577             1.0000                  24      1.0000               0        5          57
 defog      5              harnet5_dlgait - feat60_fi         walking           NaN  0.0345  0.0007  0.0655 0.0454             0.8172                  24      1.0000               4        5          57
 defog     10              harnet5_dlgait - feat60_fi         walking           NaN -0.0143 -0.0436  0.0182 0.4143             1.0000                  24      1.0000               2        5          57
 defog     20              harnet5_dlgait - feat60_fi         walking           NaN -0.0073 -0.0235  0.0122 0.4398             1.0000                  24      1.0000               1        5          57
 defog    all              harnet5_dlgait - feat60_fi         walking           NaN -0.0003 -0.0149  0.0183 0.9611             1.0000                  24      1.0000               2        5          57
 defog      5            ssl_dailygait - ssl_indomain         walking           NaN  0.0229  0.0090  0.0365 0.0017             0.0374                  24      0.1428               5        5          57
 defog     10            ssl_dailygait - ssl_indomain         walking           NaN  0.0160  0.0030  0.0306 0.0161             0.3220                  24      1.0000               5        5          57
 defog     20            ssl_dailygait - ssl_indomain         walking           NaN  0.0299  0.0145  0.0447 0.0002             0.0048                  24      0.0180               5        5          57
 defog    all            ssl_dailygait - ssl_indomain         walking           NaN  0.0207  0.0051  0.0360 0.0100             0.2100                  24      0.7700               5        5          57
 defog      5                harnet5_dl_e10 - harnet5 long_adaptation           NaN  0.0011 -0.0267  0.0326 0.9354             1.0000                  20      1.0000               1        3          57
 defog     10                harnet5_dl_e10 - harnet5 long_adaptation           NaN  0.0175 -0.0079  0.0460 0.1900             1.0000                  20      1.0000               3        3          57
 defog     20                harnet5_dl_e10 - harnet5 long_adaptation           NaN  0.0018 -0.0110  0.0172 0.7819             1.0000                  20      1.0000               3        3          57
 defog    all                harnet5_dl_e10 - harnet5 long_adaptation           NaN  0.0048 -0.0048  0.0155 0.3300             1.0000                  20      1.0000               2        3          57
 defog      5             harnet5_dl_e10 - harnet5_dl long_adaptation           NaN -0.0049 -0.0188  0.0104 0.5255             1.0000                  20      1.0000               1        3          57
 defog     10             harnet5_dl_e10 - harnet5_dl long_adaptation           NaN  0.0078  0.0007  0.0144 0.0343             0.6517                  20      1.0000               2        3          57
 defog     20             harnet5_dl_e10 - harnet5_dl long_adaptation           NaN  0.0078  0.0005  0.0148 0.0382             0.6876                  20      1.0000               3        3          57
 defog    all             harnet5_dl_e10 - harnet5_dl long_adaptation           NaN  0.0073  0.0022  0.0116 0.0089             0.1780                  20      0.7031               3        3          57
 defog      5              harnet5_dl_e10 - feat60_fi long_adaptation           NaN  0.0316 -0.0088  0.0728 0.1276             1.0000                  20      1.0000               2        3          57
 defog     10              harnet5_dl_e10 - feat60_fi long_adaptation           NaN  0.0019 -0.0296  0.0385 0.8921             1.0000                  20      1.0000               2        3          57
 defog     20              harnet5_dl_e10 - feat60_fi long_adaptation           NaN -0.0091 -0.0260  0.0116 0.3651             1.0000                  20      1.0000               0        3          57
 defog    all              harnet5_dl_e10 - feat60_fi long_adaptation           NaN  0.0020 -0.0120  0.0203 0.7874             1.0000                  20      1.0000               3        3          57
 defog      5                 harnet5_dl_e6 - harnet5 long_adaptation           NaN -0.0016 -0.0244  0.0251 0.8974             1.0000                  20      1.0000               2        3          57
 defog     10                 harnet5_dl_e6 - harnet5 long_adaptation           NaN  0.0077 -0.0216  0.0394 0.5943             1.0000                  20      1.0000               2        3          57
 defog     20                 harnet5_dl_e6 - harnet5 long_adaptation           NaN  0.0042 -0.0084  0.0191 0.5158             1.0000                  20      1.0000               2        3          57
 defog    all                 harnet5_dl_e6 - harnet5 long_adaptation           NaN  0.0014 -0.0090  0.0128 0.7459             1.0000                  20      1.0000               2        3          57
 defog      5                 harnet5_dl_e4 - harnet5 long_adaptation           NaN -0.0040 -0.0251  0.0204 0.7507             1.0000                  20      1.0000               1        3          57
 defog     10                 harnet5_dl_e4 - harnet5 long_adaptation           NaN  0.0100 -0.0188  0.0412 0.4971             1.0000                  20      1.0000               2        3          57
 defog     20                 harnet5_dl_e4 - harnet5 long_adaptation           NaN  0.0006 -0.0125  0.0163 0.9028             1.0000                  20      1.0000               2        3          57
 defog    all                 harnet5_dl_e4 - harnet5 long_adaptation           NaN  0.0002 -0.0091  0.0110 0.9394             1.0000                  20      1.0000               2        3          57
```
