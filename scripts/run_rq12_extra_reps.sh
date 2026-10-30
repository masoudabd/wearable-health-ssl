#!/bin/bash
# GPU queue: RQ1/RQ2 with the extra frozen representations (harnet10 with 10 s context, harnet5 with ML/AP swapped or
# AP negated) and tuned probes, seeds 0 to 4; waits for the external-test queue (whssl_ext) to end. Base arms checked
# per seed against the fixed runs; aggregate and paired tests at the end. Progress: results/rq12_extra_reps_STATUS.txt.
#   screen -dmS whssl_extra caffeinate -i /bin/bash scripts/run_rq12_extra_reps.sh
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ST=results/rq12_extra_reps_STATUS.txt
note() { echo "$(date '+%Y-%m-%d %H:%M') $*" >> "$ST"; }
note "queued at commit $(git rev-parse --short HEAD); waiting for whssl_ext"
while screen -ls | grep -q "whssl_ext\b"; do sleep 60; done
note "start at commit $(git rev-parse --short HEAD)"
R=""
for s in 0 1 2 3 4; do
  $PY -u scripts/rq12_extra_reps.py --device mps --threads 4 --seed $s --out rq12_extra_reps_seed$s --extra tuned_probe \
    > results/rq12_extra_reps_seed${s}_stdout.txt 2>&1 || { note "seed $s FAILED"; exit 1; }
  $PY scripts/compare_runs.py results/rq12_fog_transfer_mps_fixed_seed$s results/rq12_extra_reps_seed$s \
    > results/rq12_extra_reps_seed$s/compare_vs_fixed_stdout.txt 2>&1
  R="$R results/rq12_extra_reps_seed$s"
  note "seed $s done"
done
$PY scripts/aggregate_seeds.py $R --out results/rq12_extra_reps_seeds > results/rq12_extra_reps_aggregate_stdout.txt 2>&1
$PY scripts/paired_tests.py $R --out results/rq12_extra_reps_seeds --name paired_tests --comparisons \
  harnet10ctx:harnet5 harnet10ctx:feat60_fi harnet5_swap:harnet5 harnet5_apneg:harnet5 \
  harnet10ctx_tc:harnet5_tc harnet10ctx_tc:feat60_fi_tc > results/rq12_extra_reps_paired_stdout.txt 2>&1
note "ALL DONE (results/rq12_extra_reps_seeds/)"
