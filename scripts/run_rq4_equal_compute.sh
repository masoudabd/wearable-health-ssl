#!/bin/bash
# RQ4 equal-compute control on the fixed labels: seeds 0 to 4 on the Apple GPU, then the 5-seed aggregate and
# paired tests. Progress: results/rq4_equal_compute_STATUS.txt. About 6.5 h. Launch detached:
#   screen -dmS whssl_eqc caffeinate -i /bin/bash scripts/run_rq4_equal_compute.sh
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ST=results/rq4_equal_compute_STATUS.txt
note() { echo "$(date '+%Y-%m-%d %H:%M') $*" >> "$ST"; }
note "start at commit $(git rev-parse --short HEAD)"

R=""
for s in 0 1 2 3 4; do
  $PY -u scripts/rq4_equal_compute.py --device mps --threads 4 --seed $s --out rq4_equal_compute_seed$s \
    > results/rq4_equal_compute_seed${s}_stdout.txt 2>&1 || { note "seed $s FAILED"; exit 1; }
  R="$R results/rq4_equal_compute_seed$s"
  note "seed $s done"
done
$PY scripts/aggregate_seeds.py $R --out results/rq4_equal_compute_seeds > results/rq4_equal_compute_aggregate_stdout.txt 2>&1
$PY scripts/paired_tests.py $R --out results/rq4_equal_compute_seeds --name paired_tests --comparisons \
  ssl_dailyliving:ssl_indomain_long ssl_indomain_long:ssl_indomain ssl_dailyliving_short:ssl_indomain \
  ssl_dailyliving:ssl_dailyliving_short ssl_dailyliving:feat60_fi ssl_indomain_long:feat60_fi \
  > results/rq4_equal_compute_paired_stdout.txt 2>&1
note "ALL DONE (results/rq4_equal_compute_seeds/)"
