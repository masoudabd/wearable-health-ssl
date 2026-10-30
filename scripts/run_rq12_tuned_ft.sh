#!/bin/bash
# Equal effort for the baselines (paper plan v0 section 11 item 1), both cohorts, seeds 0 to 4 on the Apple GPU:
# RQ1/RQ2 plus the optional arms tuned_probe (<arm>_tc), supervised_es and harnet5_ft. The base arms are computed
# first in every budget and must reproduce the fixed-label runs exactly (checked per seed with compare_runs.py,
# level 2: per-fold AUROC differences of the shared arms). Then the 5-seed aggregate and the paired tests.
# Progress: results/rq12_tuned_ft_STATUS.txt. Launch detached:
#   screen -dmS whssl_tuned caffeinate -i /bin/bash scripts/run_rq12_tuned_ft.sh
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ST=results/rq12_tuned_ft_STATUS.txt
note() { echo "$(date '+%Y-%m-%d %H:%M') $*" >> "$ST"; }
note "start at commit $(git rev-parse --short HEAD)"

R=""
for s in 0 1 2 3 4; do
  $PY -u scripts/rq12_fog_transfer.py --device mps --threads 4 --seed $s --out rq12_tuned_ft_seed$s \
    --extra tuned_probe supervised_es harnet5_ft \
    > results/rq12_tuned_ft_seed${s}_stdout.txt 2>&1 || { note "seed $s FAILED"; exit 1; }
  R="$R results/rq12_tuned_ft_seed$s"
  $PY scripts/compare_runs.py results/rq12_fog_transfer_mps_fixed_seed$s results/rq12_tuned_ft_seed$s \
    > results/rq12_tuned_ft_seed$s/compare_vs_fixed_stdout.txt 2>&1
  note "seed $s done"
done
$PY scripts/aggregate_seeds.py $R --out results/rq12_tuned_ft_seeds > results/rq12_tuned_ft_aggregate_stdout.txt 2>&1
$PY scripts/paired_tests.py $R --out results/rq12_tuned_ft_seeds --name paired_tests --comparisons \
  harnet5_ft:feat60_fi harnet5_ft:harnet5 harnet5_ft:supervised_es \
  harnet5_tc:feat60_fi_tc harnet5_tc:harnet5 feat60_fi_tc:feat60_fi ssl_indomain_tc:feat60_fi_tc \
  supervised_es:supervised supervised_es:feat60_fi \
  > results/rq12_tuned_ft_paired_stdout.txt 2>&1
note "ALL DONE (results/rq12_tuned_ft_seeds/)"
