#!/bin/bash
# Queue: gravity-aligned harnet5 (orientation alignment, decision 23) via the RQ1/RQ2 probe replay
# (scripts/probe_arms.py --reps harnet5_galign), seeds 0 to 4; waits for the PADS run (whssl_pads). Checks per seed that
# the replayed harnet5 equals the fixed run's harnet5; then the aggregate and paired tests merging the fixed run, the
# equal-effort run (tuned probes) and the replay of each seed. Progress: results/galign_STATUS.txt.
#   screen -dmS whssl_galign caffeinate -i /bin/bash scripts/run_galign.sh
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ST=results/galign_STATUS.txt
note() { echo "$(date '+%Y-%m-%d %H:%M') $*" >> "$ST"; }
note "queued at commit $(git rev-parse --short HEAD); waiting for whssl_pads"
while screen -ls | grep -q "whssl_pads\b"; do sleep 30; done
note "start at commit $(git rev-parse --short HEAD)"
M=""
R=""
for s in 0 1 2 3 4; do
  $PY -u scripts/probe_arms.py --seed $s --reps harnet5_galign --device mps --out galign_seed$s \
    > results/galign_seed${s}_stdout.txt 2>&1 || { note "seed $s FAILED (exit $?)"; exit 1; }
  $PY -c "
import pandas as pd, numpy as np
a = pd.read_parquet('results/rq12_fog_transfer_mps_fixed_seed$s/out_of_fold_scores.parquet')
b = pd.read_parquet('results/galign_seed$s/out_of_fold_scores.parquet')
a, b = a[a.arm == 'harnet5'].reset_index(drop=True), b[b.arm == 'harnet5_replay'].reset_index(drop=True)
same = len(a) == len(b) and (a.subject.values == b.subject.values).all() and np.abs(a.score.values - b.score.values).max() < 1e-9
print('replay identical:', same)" > results/galign_seed$s/replay_check.txt 2>&1
  note "seed $s done ($(cat results/galign_seed$s/replay_check.txt))"
  M="$M results/rq12_fog_transfer_mps_fixed_seed$s+results/rq12_tuned_ft_seed$s+results/galign_seed$s"
  R="$R results/galign_seed$s"
done
$PY scripts/aggregate_seeds.py $R --ref harnet5_replay --out results/galign_seeds > results/galign_aggregate_stdout.txt 2>&1
rc=$?
$PY scripts/paired_tests.py $M --out results/galign_seeds --name paired_tests --comparisons \
  harnet5_galign:harnet5 harnet5_galign:feat60_fi harnet5_galign_tc:harnet5_tc harnet5_galign_tc:feat60_fi_tc \
  > results/galign_paired_stdout.txt 2>&1
note "ALL DONE (results/galign_seeds/; aggregate exit $rc, paired exit $?)"
