#!/bin/bash
# #2 Muon queue (resumable: jobs whose log has "=== sample" are skipped). Waits for the extra 5e-4 sweep point first.
cd /home/box/lisp-charmodel/efficiency
log() { echo "$(date '+%F %T %Z') $*" >> queue.log; }
if pgrep -f "queue2.sh" | grep -v $$ | grep -qv "^$PPID$"; then :; fi
while pgrep -f "extra.sh|queue.sh" >/dev/null || pgrep -f "gpt-eff.lisp --run" >/dev/null; do
  pgrep -f "gpt-eff.lisp --run.*02-muon" >/dev/null && { log "queue2: a 02-muon job is already running, exiting"; exit 0; }
  sleep 30; done
run() { local name=$1 lg=$2; shift 2
  if grep -q "=== sample" "$lg" 2>/dev/null; then log "skip $name (done)"; return; fi
  log "start $name load=$(cut -d' ' -f1-3 /proc/loadavg)"
  nice -n 10 sbcl --dynamic-space-size 3000 --script gpt-eff.lisp --run "$@" > "$lg" 2>&1
  log "end $name rc=$? $(grep -h 'trained' $lg | cut -c1-120)"; python3 make_report.py >/dev/null 2>&1; }
M="data=../huge/corpus.txt d=256 layers=6 heads=8 ctx=128 batch=32 threads=8 dropout=0 seed=1 fullval-targets=1000000 lr=1e-3 sched=wsd warm=0.01 decay=0.2 lrmin=0"
mkdir -p 02-muon/sweep
# box-speed control: same 15-min AdamW+WSD run as 01-lr/sweep/wsd-1e-3 (also re-measures its val and the run-to-run noise)
run control-wsd-1e-3 02-muon/sweep/control-adamw-wsd-1e-3.log $M secs=900 fullval-every=0 final=0 tag=02-muon/sweep/control-adamw-wsd-1e-3
for mlr in 0.01 0.02 0.04; do
  run muon-sweep-$mlr 02-muon/sweep/muon-$mlr.log $M secs=900 fullval-every=0 final=0 optim=muon mlr=$mlr tag=02-muon/sweep/muon-$mlr
done
BEST=$(python3 -c "
import glob,re
r=[(float(open(f).read().strip().split('\n')[-1].split(',')[2].replace('d0','')),re.search(r'muon-(.*)-val1m',f)[1]) for f in glob.glob('02-muon/sweep/muon-*-val1m.csv')]
print(min(r)[1])")
log "best Muon lr = $BEST"; echo "$BEST" > 02-muon/best_mlr.txt
# 3-min AdamW speed probe right before the 1-h run (box-speed reference for the confound)
run speed-probe 02-muon/speed-probe-adamw.log $M secs=180 fullval-every=0 final=0 tag=02-muon/speed-probe-adamw
run muon-full 02-muon/muon.log $M secs=3600 fullval-every=900 optim=muon mlr=$BEST tag=02-muon/muon
log "QUEUE2 DONE (#2 Muon)"
