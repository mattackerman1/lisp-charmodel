#!/bin/bash
# #3 short-context queue (resumable: jobs whose log has "=== sample" are skipped). Waits for queue2/extra2 first.
cd /home/box/lisp-charmodel/efficiency
log() { echo "$(date '+%F %T %Z') $*" >> queue.log; }
while pgrep -f "extra2.sh|queue2.sh" >/dev/null || pgrep -f "gpt-eff.lisp --run" >/dev/null; do
  pgrep -f "gpt-eff.lisp --run.*03-ctx" >/dev/null && { log "queue3: a 03-ctx job is already running, exiting"; exit 0; }
  sleep 30; done
run() { local name=$1 lg=$2; shift 2
  if grep -q "=== sample" "$lg" 2>/dev/null; then log "skip $name (done)"; return; fi
  log "start $name load=$(cut -d' ' -f1-3 /proc/loadavg)"
  nice -n 10 sbcl --dynamic-space-size 3000 --script gpt-eff.lisp --run "$@" > "$lg" 2>&1
  log "end $name rc=$? $(grep -h 'trained' $lg | cut -c1-120)"; python3 make_report.py >/dev/null 2>&1; }
# Muon LR: best of the #2 sweep (incl. the extra 0.005 point)
MLR=$(python3 -c "
import glob,re
r=[(float(open(f).read().strip().split('\n')[-1].split(',')[2].replace('d0','')),re.search(r'muon-(.*)-val1m',f)[1]) for f in glob.glob('02-muon/sweep/muon-*-val1m.csv')]
print(min(r)[1])")
echo "$MLR" > 03-ctx/mlr.txt; log "queue3: Muon lr carried forward = $MLR"
M="data=../huge/corpus.txt d=256 layers=6 heads=8 ctx=128 batch=32 threads=8 dropout=0 seed=1 fullval-targets=1000000 lr=1e-3 sched=wsd warm=0.01 decay=0.2 lrmin=0 optim=muon mlr=$MLR"
S="secs=900 fullval-every=0 final=0"
run ctx-control128 03-ctx/sweep/ctx128-control.log $M $S tag=03-ctx/sweep/ctx128-control
run ctx-sw0.75 03-ctx/sweep/ctx64-sw0.75.log $M $S ctx1=64 switch=0.75 tag=03-ctx/sweep/ctx64-sw0.75
run ctx-sw0.75-posoff 03-ctx/sweep/ctx64-sw0.75-posoff.log $M $S ctx1=64 switch=0.75 posoff=1 tag=03-ctx/sweep/ctx64-sw0.75-posoff
run ctx-sw0.6-posoff 03-ctx/sweep/ctx64-sw0.6-posoff.log $M $S ctx1=64 switch=0.6 posoff=1 tag=03-ctx/sweep/ctx64-sw0.6-posoff
run ctx-sw0.85-posoff 03-ctx/sweep/ctx64-sw0.85-posoff.log $M $S ctx1=64 switch=0.85 posoff=1 tag=03-ctx/sweep/ctx64-sw0.85-posoff
BEST=$(python3 -c "
import glob,re
r=[(float(open(f).read().strip().split('\n')[-1].split(',')[2].replace('d0','')),re.search(r'ctx64-(.*)-val1m',f)[1]) for f in glob.glob('03-ctx/sweep/ctx64-*-val1m.csv')]
print(min(r)[1])")
log "queue3: best short-ctx arm = $BEST"; echo "$BEST" > 03-ctx/best.txt
SW=$(echo $BEST | sed 's/sw\([0-9.]*\).*/\1/'); PO=0; [[ $BEST == *posoff* ]] && PO=1
run speed-probe3 03-ctx/speed-probe-ctx128.log $M secs=180 fullval-every=0 final=0 tag=03-ctx/speed-probe-ctx128
run ctx-full 03-ctx/ctx.log $M secs=3600 fullval-every=900 ctx1=64 switch=$SW posoff=$PO tag=03-ctx/ctx
log "QUEUE3 DONE (#3 short context)"
