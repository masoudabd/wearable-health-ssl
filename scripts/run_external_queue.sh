#!/bin/bash
# GPU queue after the LP-FT run: external tests on FogAtHome (FORGE release). Waits for the screen session
# whssl_lpft to end, then runs each source and axis mode. Progress: results/external_STATUS.txt. Launch detached:
#   screen -dmS whssl_ext caffeinate -i /bin/bash scripts/run_external_queue.sh
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ST=results/external_STATUS.txt
note() { echo "$(date '+%Y-%m-%d %H:%M') $*" >> "$ST"; }
note "queued at commit $(git rev-parse --short HEAD); waiting for whssl_lpft"
while screen -ls | grep -q whssl_lpft; do sleep 60; done
note "start"
for spec in "defog Vflip" "defog VMLflip" "tdcsfog VAPflip" "tdcsfog VAPMLflip"; do
  set -- $spec
  $PY -u scripts/external_fogathome.py --source $1 --axes $2 --device mps \
    > results/external_fogathome_$1_$2_stdout.txt 2>&1 && note "$1 $2 done" || note "$1 $2 FAILED"
done
note "ALL DONE"
