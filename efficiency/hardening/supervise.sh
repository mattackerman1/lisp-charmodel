#!/bin/bash
# supervise.sh RUNDIR -- start or resume one gpt-eff.lisp run; idempotent (safe to call any time, e.g. after a box restart).
# RUNDIR/run.conf (bash) must set: TAG=<name> ARGS="<gpt-eff args, no tag=/resume=/ackpt=>"; optional DYN=4000 ACKPT=600 KEEP=2 NICE=10
# Behaviour: DONE/STUCK marker -> exit; already running (flock) -> exit; preflight fails -> exit 2 (logged);
# else newest RUNDIR/TAG-state-*.meta -> resume=<it> (exact: weights+Adam/Muon+step+time+RNG), none -> fresh start.
# Two consecutive failures resuming from the SAME checkpoint -> STUCK (team rule: stop retrying).
set -u
R=$(cd "${1:?usage: supervise.sh RUNDIR}" && pwd) || exit 1
H=/home/box/lisp-charmodel/efficiency/hardening; . $H/env.sh
log() { echo "$(date '+%F %T %Z') [supervise $(basename $R)] $*" | tee -a "$R/supervisor.log"; }
[ -f "$R/run.conf" ] || { log "no run.conf"; exit 1; }
DYN=4000; ACKPT=600; KEEP=2; NICE=10; . "$R/run.conf"
[ -f "$R/DONE" ] && { echo "done already"; exit 0; }
[ -f "$R/STUCK" ] && { log "STUCK marker present ($(cat $R/STUCK)); not retrying"; exit 3; }
exec 9>"$R/.supervise.lock"
flock -n 9 || { echo "already supervised (lock held)"; exit 0; }
pgrep -f -- "tag=$R/$TAG( |\$)" >/dev/null && { log "a trainer for this tag is already running (unsupervised); exiting"; exit 0; }
$H/preflight.sh >> "$R/supervisor.log" 2>&1 || { log "PREFLIGHT FAILED -- toolchain broken, see supervisor.log; not starting"; exit 2; }
while true; do
  M=$(ls "$R/$TAG"-state-*.meta 2>/dev/null | sort | tail -1)
  if [ -n "$M" ]; then RES="resume=$M"; log "resuming from $(basename $M)"; else RES=""; log "no state checkpoint: fresh start"; fi
  echo "===== segment $(date '+%F %T %Z') boot=$(cat /proc/sys/kernel/random/boot_id) ${RES:-fresh} =====" >> "$R/$TAG.log"
  cd $EFF
  nice -n $NICE "$SBCL" --dynamic-space-size $DYN --script gpt-eff.lisp --run $ARGS ackpt=$ACKPT ackpt-keep=$KEEP $RES tag=$R/$TAG >> "$R/$TAG.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ] && grep -q "=== sample" "$R/$TAG.log"; then log "finished rc=0 $(grep -h 'FINAL' $R/$TAG.log | tail -1)"; date > "$R/DONE"; rm -f "$R/.fails"; exit 0; fi
  # killed by signal 9/15 from outside (e.g. user stop) -> do not loop
  if [ $rc -eq 137 ] || [ $rc -eq 143 ]; then log "trainer killed (rc=$rc); not restarting in this invocation"; exit $rc; fi
  key="${M:-fresh}"; prev=$(cat "$R/.fails" 2>/dev/null)
  if [ "$prev" = "$key" ]; then log "FAILED twice from $key (rc=$rc) -> STUCK"; echo "rc=$rc twice from $key $(date)" > "$R/STUCK"; exit 4; fi
  echo "$key" > "$R/.fails"; log "trainer failed rc=$rc; retrying once"; sleep 5
done
