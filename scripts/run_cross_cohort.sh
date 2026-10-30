#!/bin/bash
# Cross-cohort transfer (lab to home, home to lab), seeds 0 to 4, three axis modes (none; AP flipped; AP and ML
# flipped), CPU. Paired tests merge each seed with its fixed-label RQ1/RQ2 run (within-cohort arms). Progress:
# results/cross_cohort_STATUS.txt. Launch detached:
#   screen -dmS whssl_cross caffeinate -i /bin/bash scripts/run_cross_cohort.sh
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ST=results/cross_cohort_STATUS.txt
note() { echo "$(date '+%Y-%m-%d %H:%M') $*" >> "$ST"; }
note "start at commit $(git rev-parse --short HEAD)"
for mode in none ap apml; do
  M=""
  for s in 0 1 2 3 4; do
    $PY -u scripts/cross_cohort.py --seed $s --flip $mode > results/cross_cohort_${mode}_seed${s}_stdout.txt 2>&1 \
      || { note "$mode seed $s FAILED"; exit 1; }
    M="$M results/rq12_fog_transfer_mps_fixed_seed$s+results/cross_cohort_${mode}_seed$s"
    note "$mode seed $s done"
  done
  case $mode in none) x=_x;; ap) x=_xap;; apml) x=_xrot;; esac
  $PY scripts/paired_tests.py $M --out results/cross_cohort_seeds --name paired_tests_$mode --comparisons \
    harnet5$x:feat60_fi$x harnet5_tc$x:feat60_fi_tc$x harnet5:harnet5$x feat60_fi:feat60_fi$x \
    > results/cross_cohort_paired_${mode}_stdout.txt 2>&1 || note "$mode paired tests FAILED"
  note "$mode paired tests done"
done
note "ALL DONE (results/cross_cohort_seeds/)"
