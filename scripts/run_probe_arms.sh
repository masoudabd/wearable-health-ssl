#!/bin/bash
# CPU queue: FORGE MC as a frozen arm via the RQ1/RQ2 probe replay (scripts/probe_arms.py), seeds 0 to 4; waits for
# the cross-cohort run (whssl_cross). Checks per seed that the replayed harnet5 equals the fixed run's harnet5, then
# paired tests merging the fixed run, the equal-effort run (tuned probes) and the probe run of each seed.
# Progress: results/probe_arms_STATUS.txt.
#   screen -dmS whssl_probe caffeinate -i /bin/bash scripts/run_probe_arms.sh
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ST=results/probe_arms_STATUS.txt
note() { echo "$(date '+%Y-%m-%d %H:%M') $*" >> "$ST"; }
note "queued at commit $(git rev-parse --short HEAD); waiting for whssl_cross"
while screen -ls | grep -q "whssl_cross\b"; do sleep 60; done
note "start at commit $(git rev-parse --short HEAD)"
M=""
for s in 0 1 2 3 4; do
  $PY -u scripts/probe_arms.py --seed $s --reps forge_mc > results/probe_arms_seed${s}_stdout.txt 2>&1 \
    || { note "seed $s FAILED"; exit 1; }
  $PY -c "
import pandas as pd, numpy as np
a = pd.read_parquet('results/rq12_fog_transfer_mps_fixed_seed$s/out_of_fold_scores.parquet')
b = pd.read_parquet('results/probe_arms_seed$s/out_of_fold_scores.parquet')
a, b = a[a.arm == 'harnet5'].reset_index(drop=True), b[b.arm == 'harnet5_replay'].reset_index(drop=True)
same = len(a) == len(b) and (a.subject.values == b.subject.values).all() and np.abs(a.score.values - b.score.values).max() < 1e-9
print('replay identical:', same)" > results/probe_arms_seed$s/replay_check.txt 2>&1
  note "seed $s done ($(cat results/probe_arms_seed$s/replay_check.txt))"
  M="$M results/rq12_fog_transfer_mps_fixed_seed$s+results/rq12_tuned_ft_seed$s+results/probe_arms_seed$s"
done
$PY scripts/paired_tests.py $M --out results/probe_arms_seeds --name paired_tests --comparisons \
  forge_mc:harnet5 forge_mc:feat60_fi forge_mc_tc:harnet5_tc forge_mc_tc:feat60_fi_tc \
  > results/probe_arms_paired_stdout.txt 2>&1
note "ALL DONE (results/probe_arms_seeds/)"
