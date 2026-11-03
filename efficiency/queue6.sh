#!/bin/bash
# #6 model-size sweep (resumable: jobs whose log has "=== sample" are skipped). Base = #5: BPE-2048, Muon 0.01, WSD, ew=f32.
# Sizes around #5 (d256 L6, 5.3M): d192 L6 (~3.1M), d256 L8 (~6.9M), d384 L6 (~11.5M); head dim kept at 32. Same LRs (Muon LR is width-scaled).
cd /home/box/lisp-charmodel/efficiency
log() { echo "$(date '+%F %T %Z') $*" >> queue.log; }
while pgrep -f "queue5.sh" >/dev/null || pgrep -f "gpt-eff.lisp --run" >/dev/null; do
  pgrep -f "gpt-eff.lisp --run.*06-size" >/dev/null && { log "queue6: a 06-size job is already running, exiting"; exit 0; }
  sleep 30; done
run() { local name=$1 lg=$2; shift 2
  if grep -q "=== sample" "$lg" 2>/dev/null; then log "skip $name (done)"; return; fi
  log "start $name load=$(cut -d' ' -f1-3 /proc/loadavg)"
  nice -n 10 sbcl --dynamic-space-size 4000 --script gpt-eff.lisp --run "$@" > "$lg" 2>&1
  log "end $name rc=$? $(grep -h 'trained' $lg | cut -c1-120)"; python3 make_report.py >/dev/null 2>&1; }
B="tokens=04-bpe/tokens2048.u16 vocab=04-bpe/vocab2048.hex vstart=30476268 fullval-targets=307043 ctx=64 batch=64 threads=8 dropout=0 seed=1 lr=1e-3 sched=wsd warm=0.01 decay=0.2 lrmin=0 optim=muon mlr=0.01 ew=f32"
for sz in "d192-L6 d=192 layers=6 heads=6" "d256-L8 d=256 layers=8 heads=8" "d384-L6 d=384 layers=6 heads=12"; do
  set -- $sz; name=$1; shift
  run probe-$name 06-size/probe-before-$name.log $B d=256 layers=6 heads=8 secs=180 fullval-every=0 final=0 tag=06-size/probe-before-$name
  run size-$name 06-size/$name.log $B "$@" secs=3600 fullval-every=900 tag=06-size/$name
done
log "QUEUE6 DONE (#6 size sweep)"
