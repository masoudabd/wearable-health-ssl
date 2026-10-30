#!/bin/bash
# Second fine-tuning recipe (LP-FT: the tuned harnet5 probe as warm-started head, then all layers at lr 1e-5), both
# cohorts, seeds 0 to 4 on the Apple GPU, with the tuned probes it starts from. Base arms are checked per seed against
# the fixed runs; paired tests merge each seed with its equal-effort run (results/rq12_tuned_ft_seed<s>) for the
# comparison with the first recipe. Progress: results/rq12_lpft_STATUS.txt. Launch detached:
#   screen -dmS whssl_lpft caffeinate -i /bin/bash scripts/run_rq12_lpft.sh
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ST=results/rq12_lpft_STATUS.txt
note() { echo "$(date '+%Y-%m-%d %H:%M') $*" >> "$ST"; }
note "start at commit $(git rev-parse --short HEAD)"

R=""
M=""
for s in 0 1 2 3 4; do
  $PY -u scripts/rq12_fog_transfer.py --device mps --threads 4 --seed $s --out rq12_lpft_seed$s \
    --extra tuned_probe harnet5_lpft > results/rq12_lpft_seed${s}_stdout.txt 2>&1 || { note "seed $s FAILED"; exit 1; }
  $PY scripts/compare_runs.py results/rq12_fog_transfer_mps_fixed_seed$s results/rq12_lpft_seed$s \
    > results/rq12_lpft_seed$s/compare_vs_fixed_stdout.txt 2>&1
  R="$R results/rq12_lpft_seed$s"
  M="$M results/rq12_tuned_ft_seed$s+results/rq12_lpft_seed$s"
  note "seed $s done"
done
$PY scripts/aggregate_seeds.py $R --out results/rq12_lpft_seeds > results/rq12_lpft_aggregate_stdout.txt 2>&1
$PY scripts/paired_tests.py $M --out results/rq12_lpft_seeds --name paired_tests --comparisons \
  harnet5_lpft:harnet5_tc harnet5_lpft:harnet5_ft harnet5_lpft:harnet5 harnet5_lpft:feat60_fi \
  harnet5_lpft:feat60_fi_tc > results/rq12_lpft_paired_stdout.txt 2>&1
note "ALL DONE (results/rq12_lpft_seeds/)"
