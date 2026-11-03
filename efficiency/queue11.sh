#!/bin/bash
# #11 decay-from-checkpoint: resume #10 best weights (10-long/v16384-const-ckpt-097608.f32, weights only, no optimizer state),
# linear LR decay peak (Muon 0.01 / AdamW 1e-3) -> 0 over 720 s of training time = 0.2 x 3600 s, the WSD decay fraction of the 1 h #8 run.
# Midpoint (360 s) val-1M + full val + checkpoint, then end-of-training val-1M + FINAL full val.
cd /home/box/lisp-charmodel/efficiency
log() { echo "$(date '+%F %T %Z') $*" >> queue.log; }
pgrep -f "gpt-eff.lisp --run.*11-decay" >/dev/null && { log "queue11: an 11-decay job is already running, exiting"; exit 0; }
while pgrep -f "gpt-eff.lisp --run" >/dev/null; do sleep 30; done
lg=11-decay/v16384-decay.log
grep -q "=== sample" $lg 2>/dev/null && { log "skip v16384-decay (done)"; exit 0; }
t=0; while [ $(awk '/MemAvail/{print int($2/1024)}' /proc/meminfo) -lt 3000 ]; do [ $t -eq 0 ] && log "queue11: waiting for 3000MB free"; sleep 60; t=$((t+60)); [ $t -ge 10800 ] && { log "queue11: stop, not enough RAM"; exit 1; }; done
log "start v16384-decay (720 s linear decay from #10 ckpt 97608) load=$(cut -d' ' -f1-3 /proc/loadavg) memavail=$(awk '/MemAvail/{print int($2/1024)}' /proc/meminfo)MB"
nice -n 10 sbcl --dynamic-space-size 4000 --script gpt-eff.lisp --run tokens=08-vocab-up/tokens16384.u16 vocab=08-vocab-up/vocab16384.hex vstart=23505859 fullval-targets=238949 \
  d=256 layers=6 heads=8 ctx=64 batch=64 threads=8 dropout=0 seed=11 lr=1e-3 sched=lin warm=0 optim=muon mlr=0.01 ew=f32 \
  init=10-long/v16384-const-ckpt-097608.f32 secs=720 fullval-every=360 periodic-full=1 ckpt=1 tag=11-decay/v16384-decay > $lg 2>&1
log "end v16384-decay rc=$? $(grep -h 'trained\|FINAL' $lg | cut -c1-140 | tr '\n' ' ')"
log "QUEUE11 DONE (#11 decay)"
