#!/bin/bash
# #12: same 720 s linear decay as #11 (init= + sched=lin), from earlier #10 checkpoints. Usage: queue12.sh 2h|5h (one run per launch).
# 2h -> 10-long/v16384-const-ckpt-020410.f32 (2.00 train h, #10 full val 1.0815); 5h -> ckpt-050960 (5.00 h, 1.0679). Final weights saved (savefinal=1).
cd /home/box/lisp-charmodel/efficiency
log() { echo "$(date '+%F %T %Z') $*" >> queue.log; }
case "$1" in 2h) ck=020410;; 5h) ck=050960;; *) echo "usage: $0 2h|5h"; exit 2;; esac
tag=12-decay-from-earlier/v16384-decay-$1; lg=$tag.log
pgrep -f "gpt-eff.lisp --run.*12-decay-from-earlier" >/dev/null && { log "queue12: a 12-decay job is already running, exiting"; exit 0; }
while pgrep -f "gpt-eff.lisp --run" >/dev/null; do sleep 30; done
grep -q "=== sample" $lg 2>/dev/null && { log "skip decay-$1 (done)"; exit 0; }
t=0; while [ $(awk '/MemAvail/{print int($2/1024)}' /proc/meminfo) -lt 3000 ]; do [ $t -eq 0 ] && log "queue12: waiting for 3000MB free"; sleep 60; t=$((t+60)); [ $t -ge 10800 ] && { log "queue12: stop, not enough RAM"; exit 1; }; done
log "start decay-$1 (720 s linear decay from #10 ckpt $ck) load=$(cut -d' ' -f1-3 /proc/loadavg) memavail=$(awk '/MemAvail/{print int($2/1024)}' /proc/meminfo)MB"
nice -n 10 sbcl --dynamic-space-size 4000 --script gpt-eff.lisp --run tokens=08-vocab-up/tokens16384.u16 vocab=08-vocab-up/vocab16384.hex vstart=23505859 fullval-targets=238949 \
  d=256 layers=6 heads=8 ctx=64 batch=64 threads=8 dropout=0 seed=11 lr=1e-3 sched=lin warm=0 optim=muon mlr=0.01 ew=f32 \
  init=10-long/v16384-const-ckpt-$ck.f32 secs=720 fullval-every=360 periodic-full=1 ckpt=1 savefinal=1 tag=$tag > $lg 2>&1
log "end decay-$1 rc=$? $(grep -h 'trained\|FINAL' $lg | cut -c1-140 | tr '\n' ' ')"
log "QUEUE12 $1 DONE"
