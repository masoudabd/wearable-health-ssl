#!/bin/bash
# Recomputes every patient-bootstrap interval and Holm family at a different resample count, from stored scores.
# Nothing is retrained: the paired tests read each run's out-of-fold scores, and the dataset evaluations refit only
# the cheap CPU probes from cached representations, so the point estimates must come out unchanged.
#
#   NB=20000 SUF=_b20k bash scripts/rebootstrap.sh     the rebuild for submission (internal review 1, decision a)
#   NB=2000  SUF=_check bash scripts/rebootstrap.sh     no-op check: every output must equal the registered one
#
# Writes results/<family>$SUF/ and results/paper_derived$SUF/, leaving the registered outputs untouched. Rerunning
# the same command resumes: steps the status file already reports as done are skipped (delete its line to redo one).
# Progress: results/rebootstrap<SUF>_STATUS.txt, per-step logs in results/rebootstrap_logs<SUF>/. Launch detached:
#   screen -dmS whssl_boot caffeinate -i /bin/bash -c 'NB=20000 SUF=_b20k bash scripts/rebootstrap.sh'
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
NB=${NB:-20000}
SUF=${SUF:-_b20k}
DEV=${DEV:-mps}
ST=results/rebootstrap${SUF}_STATUS.txt
DERIVED=results/paper_derived$SUF
note() { echo "$(date '+%Y-%m-%d %H:%M') $*" >> "$ST"; }
run() {  # run <label> <logfile> <command...>; a step the status file already reports as done is skipped, so the
  local label=$1 log=$2; shift 2   # same command resumes an interrupted rebuild without repeating finished work
  if [ -f "$ST" ] && grep -qF "$label done" "$ST"; then echo "skip (done): $label"; return 0; fi
  "$@" > "$log" 2>&1; local rc=$?
  [ $rc -eq 0 ] && note "$label done" || note "$label FAILED (exit $rc, see $log)"
}
note "start at commit $(git rev-parse --short HEAD), n_boot $NB, suffix $SUF"
mkdir -p "$DERIVED" results/rebootstrap_logs$SUF
L=results/rebootstrap_logs$SUF

# Seed run lists, as in the runners that produced the registered tables
R12=""; R4=""; TF=""; LP=""; XR=""; GA=""; PA=""; EQ=""; GT=""; LA=""
for s in 0 1 2 3 4; do
  R12="$R12 results/rq12_fog_transfer_mps_fixed_seed$s"
  R4="$R4 results/rq4_daily_living_fixed_seed$s"
  TF="$TF results/rq12_tuned_ft_seed$s"
  LP="$LP results/rq12_tuned_ft_seed$s+results/rq12_lpft_seed$s"
  XR="$XR results/rq12_extra_reps_seed$s"
  GA="$GA results/rq12_fog_transfer_mps_fixed_seed$s+results/rq12_tuned_ft_seed$s+results/galign_seed$s"
  PA="$PA results/rq12_fog_transfer_mps_fixed_seed$s+results/rq12_tuned_ft_seed$s+results/probe_arms_seed$s"
  EQ="$EQ results/rq4_equal_compute_seed$s"
  GT="$GT results/rq4_daily_living_fixed_seed$s+results/rq4_gait_seed$s"
done
for s in 0 1 2; do LA="$LA results/rq4_daily_living_fixed_seed$s+results/rq4_long_adapt_seed$s"; done

# 1. Paired-test families (stored scores only)
run "RQ1/RQ2 48 tests" $L/rq12.log $PY scripts/paired_tests.py $R12 --n-boot $NB --out results/rq12_seeds_mps_fixed$SUF
run "equal effort 72 tests" $L/tuned_ft.log $PY scripts/paired_tests.py $TF --n-boot $NB \
  --out results/rq12_tuned_ft_seeds$SUF --name paired_tests --comparisons \
  harnet5_ft:feat60_fi harnet5_ft:harnet5 harnet5_ft:supervised_es \
  harnet5_tc:feat60_fi_tc harnet5_tc:harnet5 feat60_fi_tc:feat60_fi ssl_indomain_tc:feat60_fi_tc \
  supervised_es:supervised supervised_es:feat60_fi
run "RQ4 daily living" $L/rq4.log $PY scripts/paired_tests.py $R4 --n-boot $NB \
  --out results/rq4_seeds_mps_fixed$SUF --name rq4_paired_tests --comparisons \
  ssl_dailyliving:ssl_indomain ssl_dailyliving:feat60_fi ssl_dailyliving:harnet5 harnet5_dl:harnet5 \
  harnet5_dl:feat60_fi harnet5_dl:ssl_indomain ssl_indomain:feat60_fi
run "RQ4 equal compute" $L/rq4_eqc.log $PY scripts/paired_tests.py $EQ --n-boot $NB \
  --out results/rq4_equal_compute_seeds$SUF --name paired_tests --comparisons \
  ssl_dailyliving:ssl_indomain_long ssl_indomain_long:ssl_indomain ssl_dailyliving_short:ssl_indomain \
  ssl_dailyliving:ssl_dailyliving_short ssl_dailyliving:feat60_fi ssl_indomain_long:feat60_fi
run "RQ4 walking-like" $L/rq4_gait.log $PY scripts/paired_tests.py $GT --n-boot $NB \
  --out results/rq4_gait_seeds$SUF --name paired_tests --comparisons \
  ssl_dailygait:ssl_dailyliving harnet5_dlgait:harnet5_dl harnet5_dlgait:harnet5 \
  ssl_dailygait:feat60_fi harnet5_dlgait:feat60_fi ssl_dailygait:ssl_indomain
run "RQ4 long adaptation" $L/rq4_long.log $PY scripts/paired_tests.py $LA --n-boot $NB \
  --out results/rq4_long_adapt_seeds$SUF --name paired_tests --comparisons \
  harnet5_dl_e10:harnet5 harnet5_dl_e10:harnet5_dl harnet5_dl_e10:feat60_fi harnet5_dl_e6:harnet5 \
  harnet5_dl_e4:harnet5
run "LP-FT" $L/lpft.log $PY scripts/paired_tests.py $LP --n-boot $NB \
  --out results/rq12_lpft_seeds$SUF --name paired_tests --comparisons \
  harnet5_lpft:harnet5_tc harnet5_lpft:harnet5_ft harnet5_lpft:harnet5 harnet5_lpft:feat60_fi \
  harnet5_lpft:feat60_fi_tc
run "extra representations" $L/extra_reps.log $PY scripts/paired_tests.py $XR --n-boot $NB \
  --out results/rq12_extra_reps_seeds$SUF --name paired_tests --comparisons \
  harnet10ctx:harnet5 harnet10ctx:feat60_fi harnet5_swap:harnet5 harnet5_apneg:harnet5 \
  harnet10ctx_tc:harnet5_tc harnet10ctx_tc:feat60_fi_tc
run "gravity alignment" $L/galign.log $PY scripts/paired_tests.py $GA --n-boot $NB \
  --out results/galign_seeds$SUF --name paired_tests --comparisons \
  harnet5_galign:harnet5 harnet5_galign:feat60_fi harnet5_galign_tc:harnet5_tc harnet5_galign_tc:feat60_fi_tc
run "FORGE" $L/probe_arms.log $PY scripts/paired_tests.py $PA --n-boot $NB \
  --out results/probe_arms_seeds$SUF --name paired_tests --comparisons \
  forge_mc:harnet5 forge_mc:feat60_fi forge_mc_tc:harnet5_tc forge_mc_tc:feat60_fi_tc
for mode in none ap apml; do
  M=""
  for s in 0 1 2 3 4; do M="$M results/rq12_fog_transfer_mps_fixed_seed$s+results/cross_cohort_${mode}_seed$s"; done
  case $mode in none) x=_x;; ap) x=_xap;; apml) x=_xrot;; esac
  run "cross cohort $mode" $L/cross_$mode.log $PY scripts/paired_tests.py $M --n-boot $NB \
    --out results/cross_cohort_seeds$SUF --name paired_tests_$mode --comparisons \
    harnet5$x:feat60_fi$x harnet5_tc$x:feat60_fi_tc$x harnet5:harnet5$x feat60_fi:feat60_fi$x
done

# 2. Other datasets and the external test (probes refitted on the CPU from cached representations)
for spec in "defog Vflip" "defog VMLflip" "tdcsfog VAPflip" "tdcsfog VAPMLflip"; do
  set -- $spec
  run "external $1 $2" $L/external_$1_$2.log $PY -u scripts/external_fogathome.py --source $1 --axes $2 \
    --device $DEV --n-boot $NB --out external_fogathome_$1_$2$SUF
done

# 3. Post hoc analyses that resample themselves, into the parallel derived tree
ln -sfn "$PWD/results/paper_derived/annotations" "$DERIVED/annotations"   # labels, no resampling
export WHSSL_DERIVED="$PWD/$DERIVED"
for a in intervals ptf_agreement ptf_time severity calibration episodes episode_strict; do
  run "post hoc $a" $L/paper_$a.log $PY -u scripts/paper_$a.py --n-boot $NB
done
run "post hoc fog_types" $L/paper_fog_types.log $PY -u scripts/paper_fog_types.py --n-boot $NB
run "post hoc val_operating" $L/paper_val_operating.log $PY -u scripts/paper_val_operating.py --n-boot $NB
run "post hoc latency" $L/paper_latency.log $PY -u scripts/paper_latency.py --n-boot $NB   # after episodes, val_operating
run "post hoc forge_burden" $L/paper_forge_burden.log $PY -u scripts/paper_forge_burden.py --n-boot $NB
run "post hoc validation_audit" $L/paper_validation_audit.log $PY -u scripts/paper_validation_audit.py

# 4. Analyses that only read the tables above
run "post hoc rq4_family" $L/paper_rq4_family.log $PY scripts/paper_rq4_family.py --suffix $SUF
run "post hoc external" $L/paper_external.log $PY scripts/paper_external.py --runs-suffix $SUF
# the per-participant free-living rows the paper reads come from the two registered specs, combined
run "external per participant" $L/external_pp.log $PY scripts/combine_external_participants.py --suffix "$SUF"
run "post hoc cross_cohort" $L/paper_cross_cohort.log $PY scripts/paper_cross_cohort.py
note "ALL DONE (results/*$SUF, $DERIVED)"
