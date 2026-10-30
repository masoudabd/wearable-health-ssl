#!/bin/bash
# GPU queue: RQ4 longer adaptation of harnet5 on daily living (10 epochs, checkpoints 2, 4, 6, 10), seeds 0 to 2;
# waits for the extra-representation queue (whssl_extra). Paired tests merge each seed with its RQ4 run.
# Progress: results/rq4_long_adapt_STATUS.txt. About 5 h per seed.
#   screen -dmS whssl_long caffeinate -i /bin/bash scripts/run_rq4_long_adapt.sh
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ST=results/rq4_long_adapt_STATUS.txt
note() { echo "$(date '+%Y-%m-%d %H:%M') $*" >> "$ST"; }
note "queued at commit $(git rev-parse --short HEAD); waiting for whssl_extra"
while screen -ls | grep -q "whssl_extra\b"; do sleep 60; done
note "start at commit $(git rev-parse --short HEAD)"
R=""
M=""
for s in 0 1 2; do
  $PY -u scripts/rq4_long_adapt.py --device mps --threads 4 --seed $s --out rq4_long_adapt_seed$s \
    > results/rq4_long_adapt_seed${s}_stdout.txt 2>&1 || { note "seed $s FAILED"; exit 1; }
  R="$R results/rq4_long_adapt_seed$s"
  M="$M results/rq4_daily_living_fixed_seed$s+results/rq4_long_adapt_seed$s"
  note "seed $s done"
done
$PY scripts/aggregate_seeds.py $R --out results/rq4_long_adapt_seeds > results/rq4_long_adapt_aggregate_stdout.txt 2>&1
$PY scripts/paired_tests.py $M --out results/rq4_long_adapt_seeds --name paired_tests --comparisons \
  harnet5_dl_e10:harnet5 harnet5_dl_e10:harnet5_dl harnet5_dl_e10:feat60_fi harnet5_dl_e6:harnet5 \
  harnet5_dl_e4:harnet5 > results/rq4_long_adapt_paired_stdout.txt 2>&1
note "ALL DONE (results/rq4_long_adapt_seeds/)"
