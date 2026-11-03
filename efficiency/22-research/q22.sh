#!/bin/bash
# q22.sh LISTFILE -- #22 sequential queue (one 8-thread run at a time). Lines: TAG|DYN|KEEP-HINT|NOTE|ARGS-EXTRA
# Each run: runs/TAG/run.conf (base #19 flags + ARGS-EXTRA), supervise.sh (resumable state ckpts, 540 s), then append-row.sh.
# Idempotent: rerun after a box restart; DONE runs are skipped. Waits for the teacher dump to finish (no CPU overlap).
E=/home/box/lisp-charmodel/efficiency; Q=$E/22-research; H=$E/hardening; L=${1:?}
exec 8>$Q/.q22.$(basename $L).lock; flock -n 8 || { echo "queue $L already running"; exit 0; }
BASE="tokens=08-vocab-up/tokens16384.u16 vocab=08-vocab-up/vocab16384.hex vstart=23505859 fullval-targets=238949 d=256 layers=6 heads=8 ctx=64 batch=64 threads=8 dropout=0 lr=1e-3 optim=muon mlr=0.01 ew=f32 sched=wsd warm=0.01 decay=0.2 lrmin=0 ssm=1024 ssm-alpha=0.75 gemm=lisp512 fullval-every=900 savefinal=1"
while [ ! -f $Q/teacher/kd16.u16.DONE ]; do sleep 60; done
grep -v '^#' $L | while IFS='|' read -r TAG DYN KEEP NOTE EXTRA; do
  [ -z "$TAG" ] && continue; R=$Q/runs/$TAG; mkdir -p $R
  [ -f $R/run.conf ] || printf 'TAG=%s\nARGS="%s %s"\nDYN=%s; ACKPT=540; KEEP=2; NICE=10\n' "$TAG" "$BASE" "$EXTRA" "$DYN" > $R/run.conf
  echo "$NOTE" > $R/note; [ -f $R/keep ] || echo "${KEEP:-pending}" > $R/keep
  if [ ! -f $R/DONE ]; then
    # be polite to other box users (Experimenter): wait while someone else is using >2 cores
    while [ "$(ps -eo pcpu,args --no-headers | grep -v -e "$R" -e grep | awk '/sbcl|python|gpt/{s+=$1} END{print int(s)}')" -gt 200 ]; do echo "$(date '+%T') waiting: other heavy procs"; sleep 120; done
    echo "$(date '+%F %T %Z') START $TAG"; $H/supervise.sh $R; echo "$(date '+%F %T %Z') END $TAG rc=$? $(grep -h FINAL $R/*.log | tail -1)"
  fi
  [ -f $R/DONE ] && $Q/append-row.sh $R
done
echo "Q22_DONE $L"
