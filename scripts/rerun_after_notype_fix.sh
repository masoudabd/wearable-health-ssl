#!/bin/bash
# Reruns on the fixed home labels (notype fix, commit 493fb42), on the Apple GPU:
#   1. RQ1/RQ2, both cohorts, seeds 0 to 4, then the 5-seed aggregate and the 48 paired tests
#   2. RQ3 for each of those runs (CPU, in the background)
#   3. RQ4 (home), seeds 0 to 4, then its aggregate and paired tests
# Progress: results/fixed_reruns_STATUS.txt. About 8.5 h. Launch detached:
#   screen -dmS whssl_fixed caffeinate -i /bin/bash scripts/rerun_after_notype_fix.sh
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ST=results/fixed_reruns_STATUS.txt
note() { echo "$(date '+%Y-%m-%d %H:%M') $*" >> "$ST"; }
note "start at commit $(git rev-parse --short HEAD)"

R12=""
for s in 0 1 2 3 4; do
  $PY -u scripts/rq12_fog_transfer.py --device mps --threads 4 --seed $s --out rq12_fog_transfer_mps_fixed_seed$s \
    > results/rq12_fog_transfer_mps_fixed_seed${s}_stdout.txt 2>&1 || { note "RQ1/RQ2 seed $s FAILED"; exit 1; }
  R12="$R12 results/rq12_fog_transfer_mps_fixed_seed$s"
  note "RQ1/RQ2 seed $s done"
done
$PY scripts/aggregate_seeds.py $R12 --out results/rq12_seeds_mps_fixed > results/rq12_seeds_mps_fixed_aggregate_stdout.txt 2>&1
$PY scripts/paired_tests.py $R12 --out results/rq12_seeds_mps_fixed > results/rq12_seeds_mps_fixed_paired_stdout.txt 2>&1
note "RQ1/RQ2 aggregate and paired tests done (results/rq12_seeds_mps_fixed/)"

for s in 0 1 2 3 4; do  # RQ3 on the CPU while RQ4 uses the GPU
  ( $PY -u scripts/rq3_severity_fog.py --run results/rq12_fog_transfer_mps_fixed_seed$s \
      > results/rq3_fixed_seed${s}_stdout.txt 2>&1 && note "RQ3 seed $s done" || note "RQ3 seed $s FAILED" ) &
done

R4=""
for s in 0 1 2 3 4; do
  $PY -u scripts/rq4_daily_living.py --device mps --threads 4 --seed $s --out rq4_daily_living_fixed_seed$s \
    > results/rq4_daily_living_fixed_seed${s}_stdout.txt 2>&1 || { note "RQ4 seed $s FAILED"; wait; exit 1; }
  R4="$R4 results/rq4_daily_living_fixed_seed$s"
  note "RQ4 seed $s done"
done
wait
$PY scripts/aggregate_seeds.py $R4 --out results/rq4_seeds_mps_fixed > results/rq4_seeds_mps_fixed_aggregate_stdout.txt 2>&1
$PY scripts/paired_tests.py $R4 --out results/rq4_seeds_mps_fixed --name rq4_paired_tests --comparisons \
  ssl_dailyliving:ssl_indomain ssl_dailyliving:feat60_fi ssl_dailyliving:harnet5 harnet5_dl:harnet5 \
  harnet5_dl:feat60_fi harnet5_dl:ssl_indomain ssl_indomain:feat60_fi > results/rq4_seeds_mps_fixed_paired_stdout.txt 2>&1
note "ALL DONE (results/rq4_seeds_mps_fixed/)"
