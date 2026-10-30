#!/bin/bash
# RQ4 gait-focused variant on the fixed labels: seeds 0 to 4 on the Apple GPU, then the 5-seed aggregate and paired
# tests against the RQ4 arms (each seed's run merged with the RQ4 run of the same seed). Progress:
# results/rq4_gait_STATUS.txt. About 8 h. Launch detached:
#   screen -dmS whssl_gait caffeinate -i /bin/bash scripts/run_rq4_gait.sh
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ST=results/rq4_gait_STATUS.txt
note() { echo "$(date '+%Y-%m-%d %H:%M') $*" >> "$ST"; }
note "start at commit $(git rev-parse --short HEAD)"

R=""
M=""
for s in 0 1 2 3 4; do
  $PY -u scripts/rq4_gait.py --device mps --threads 4 --seed $s --out rq4_gait_seed$s \
    > results/rq4_gait_seed${s}_stdout.txt 2>&1 || { note "seed $s FAILED"; exit 1; }
  R="$R results/rq4_gait_seed$s"
  M="$M results/rq4_daily_living_fixed_seed$s+results/rq4_gait_seed$s"
  note "seed $s done"
done
$PY scripts/aggregate_seeds.py $R --out results/rq4_gait_seeds > results/rq4_gait_aggregate_stdout.txt 2>&1
$PY scripts/paired_tests.py $M --out results/rq4_gait_seeds --name paired_tests --comparisons \
  ssl_dailygait:ssl_dailyliving harnet5_dlgait:harnet5_dl harnet5_dlgait:harnet5 \
  ssl_dailygait:feat60_fi harnet5_dlgait:feat60_fi ssl_dailygait:ssl_indomain \
  > results/rq4_gait_paired_stdout.txt 2>&1
note "ALL DONE (results/rq4_gait_seeds/)"
