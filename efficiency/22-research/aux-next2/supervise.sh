#!/bin/bash
# Isolated supervisor for the horizon-2 head. Same contract as hardening/supervise.sh.
# Invokes 22-research/aux-next2/gpt-eff.lisp. Does not call or modify the live
# gpt-eff.lisp, q22.sh, or hardening/supervise.sh.
# RUNDIR/run.conf (bash): TAG, ARGS. Optional DYN=4000 ACKPT=540 KEEP=2 NICE=10.
set -u
R=$(cd "${1:?usage: aux-next2/supervise.sh RUNDIR}" && pwd) || exit 1
HERE=$(cd "$(dirname "$0")" && pwd)
EFF=$(cd "$HERE/../.." && pwd)
# Box toolchain when present. A checkout without it must export SBCL and H.
if [ -z "${H:-}" ]; then
  if [ -f /home/box/lisp-charmodel/efficiency/hardening/env.sh ]; then H=/home/box/lisp-charmodel/efficiency/hardening
  else H=$EFF/hardening; fi
fi
if [ -f "$H/env.sh" ]; then . "$H/env.sh"; fi
# env.sh on the box pins EFF to the live tree. Re-pin to the tree that holds this script
# only when that tree is the one we were launched from; the box copy lives inside the live EFF.
EFF=${EFF_OVERRIDE:-$EFF}
SBCL=${SBCL:-/home/box/opt/sbcl/bin/sbcl}
LISP=$HERE/gpt-eff.lisp
log() { echo "$(date '+%F %T %Z') [n2-supervise $(basename $R)] $*" | tee -a "$R/supervisor.log"; }
[ -f "$R/run.conf" ] || { log "no run.conf"; exit 1; }
[ -f "$LISP" ] || { log "missing isolated lisp $LISP"; exit 1; }
DYN=4000; ACKPT=540; KEEP=2; NICE=10; . "$R/run.conf"
[ -f "$R/DONE" ] && { echo "done already"; exit 0; }
[ -f "$R/STUCK" ] && { log "STUCK marker present ($(cat $R/STUCK)); not retrying"; exit 3; }
exec 9>"$R/.supervise.lock"
flock -n 9 || { echo "already supervised (lock held)"; exit 0; }
pgrep -f -- "tag=$R/$TAG( |\$)" >/dev/null && { log "a trainer for this tag is already running; exiting"; exit 0; }
if [ -x "$H/preflight.sh" ]; then
  "$H/preflight.sh" >> "$R/supervisor.log" 2>&1 || { log "PREFLIGHT FAILED -- not starting"; exit 2; }
else
  log "no preflight at $H/preflight.sh; continuing"
fi
BASE_SHA=8be0dbaa692053ad6f7a125ff5880d3462100cdffb47c8a3132e54261627b5cb
got=$(sha256sum "$EFF/gpt-eff.lisp" | awk '{print $1}')
if [ "$got" != "$BASE_SHA" ]; then
  log "REFUSING: live $EFF/gpt-eff.lisp sha256 $got != baseline $BASE_SHA. Do not overwrite that file."
  exit 5
fi
log "live baseline sha256 ok; isolated lisp $(sha256sum "$LISP" | awk '{print $1}')"
while true; do
  M=$(ls "$R/$TAG"-state-*.meta 2>/dev/null | sort | tail -1)
  if [ -n "$M" ]; then RES="resume=$M"; log "resuming from $(basename $M)"; else RES=""; log "no state checkpoint: fresh start"; fi
  echo "===== segment $(date '+%F %T %Z') boot=$(cat /proc/sys/kernel/random/boot_id) ${RES:-fresh} lisp=$LISP =====" >> "$R/$TAG.log"
  cd "$EFF"
  nice -n $NICE "$SBCL" --dynamic-space-size $DYN --script "$LISP" --run $ARGS ackpt=$ACKPT ackpt-keep=$KEEP $RES tag=$R/$TAG >> "$R/$TAG.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ] && grep -q "=== sample" "$R/$TAG.log"; then log "finished rc=0 $(grep -h 'FINAL' $R/$TAG.log | tail -1)"; date > "$R/DONE"; rm -f "$R/.fails"; exit 0; fi
  if [ $rc -eq 137 ] || [ $rc -eq 143 ]; then log "trainer killed (rc=$rc); not restarting in this invocation"; exit $rc; fi
  key="${M:-fresh}"; prev=$(cat "$R/.fails" 2>/dev/null || true)
  if [ "$prev" = "$key" ]; then log "FAILED twice from $key (rc=$rc) -> STUCK"; echo "rc=$rc twice from $key $(date)" > "$R/STUCK"; exit 4; fi
  echo "$key" > "$R/.fails"; log "trainer failed rc=$rc; retrying once"; sleep 5
done
