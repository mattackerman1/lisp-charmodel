#!/bin/bash
# #10 long run: best model (v16384, d256-L6, ctx 64, Muon 0.01, f32 GELU, dropout 0), constant LR after a 36 s warmup (same warmup as the 1 h runs),
# val-1M + FULL val + recent train loss every 30 min of training time, weight checkpoint at each eval, 12 h wall cap,
# early stop when the last 3 full-val evals are all > best + 0.003 nats/char.
cd /home/box/lisp-charmodel/efficiency
log() { echo "$(date '+%F %T %Z') $*" >> queue.log; }
pgrep -f "gpt-eff.lisp --run.*10-long" >/dev/null && { log "queue10: a 10-long job is already running, exiting"; exit 0; }
while pgrep -f "gpt-eff.lisp --run" >/dev/null; do sleep 30; done
lg=10-long/v16384-const.log
grep -q "=== sample" $lg 2>/dev/null && { log "skip v16384-const (done)"; exit 0; }
t=0; while [ $(awk '/MemAvail/{print int($2/1024)}' /proc/meminfo) -lt 3000 ]; do [ $t -eq 0 ] && log "queue10: waiting for 3000MB free"; sleep 60; t=$((t+60)); [ $t -ge 10800 ] && { log "queue10: stop, not enough RAM"; exit 1; }; done
log "start v16384-const (12 h) load=$(cut -d' ' -f1-3 /proc/loadavg) memavail=$(awk '/MemAvail/{print int($2/1024)}' /proc/meminfo)MB"
nice -n 10 sbcl --dynamic-space-size 4000 --script gpt-eff.lisp --run tokens=08-vocab-up/tokens16384.u16 vocab=08-vocab-up/vocab16384.hex vstart=23505859 fullval-targets=238949 \
  d=256 layers=6 heads=8 ctx=64 batch=64 threads=8 dropout=0 seed=1 lr=1e-3 sched=const warm=0.000833 optim=muon mlr=0.01 ew=f32 \
  secs=43200 wallmax=43200 fullval-every=1800 periodic-full=1 ckpt=1 earlystop=3 estol=0.003 tag=10-long/v16384-const > $lg 2>&1
log "end v16384-const rc=$? $(grep -h 'trained\|EARLY' $lg | cut -c1-140 | tr '\n' ' ')"; python3 make_report.py >/dev/null 2>&1
log "QUEUE10 DONE (#10 long run)"
