#!/bin/bash
# #13 cyclic warm restarts from scratch: v16384 d256-L6 ctx 64, Muon 0.01 / AdamW 1e-3, seed 11, f32 GELU, dropout 0.
# 5 cycles x (7200 s constant peak + 720 s linear decay to 0), jump back to peak; 36 s warmup at the very start only.
# Optimizer state (Adam moments, Muon momentum) is carried CONTINUOUSLY across restarts (not reset).
# Full val + val-1M + weight ckpt every 1800 s of training AND at each decay end (CYCLE-END); final weights saved.
# Crash resume (manual): init=<latest ckpt> toff=<its train_seconds> secs=<39600-toff> warm=0 (weights only; optimizer restarts).
cd /home/box/lisp-charmodel/efficiency
. hardening/env.sh   # HARDENING: sbcl/OpenBLAS/matplotlib under /home/box/opt (survive box restarts)
hardening/preflight.sh >> queue.log 2>&1 || { echo "$(date '+%F %T %Z') queue13.sh: PREFLIGHT FAILED, not starting (see queue.log)" >> queue.log; exit 2; }
log() { echo "$(date '+%F %T %Z') $*" >> queue.log; }
pgrep -f "gpt-eff.lisp --run.*13-cyclic" >/dev/null && { log "queue13: a 13-cyclic job is already running, exiting"; exit 0; }
while pgrep -f "gpt-eff.lisp --run" >/dev/null; do sleep 30; done
lg=13-cyclic/v16384-cyc.log
grep -q "=== sample" $lg 2>/dev/null && { log "skip v16384-cyc (done)"; exit 0; }
t=0; while [ $(awk '/MemAvail/{print int($2/1024)}' /proc/meminfo) -lt 3000 ]; do [ $t -eq 0 ] && log "queue13: waiting for 3000MB free"; sleep 60; t=$((t+60)); [ $t -ge 10800 ] && { log "queue13: stop, not enough RAM"; exit 1; }; done
log "start v16384-cyc (5 x 2h+720s cycles, 11 h training) load=$(cut -d' ' -f1-3 /proc/loadavg) memavail=$(awk '/MemAvail/{print int($2/1024)}' /proc/meminfo)MB"
nice -n 10 "$SBCL" --dynamic-space-size 4000 --script gpt-eff.lisp --run tokens=08-vocab-up/tokens16384.u16 vocab=08-vocab-up/vocab16384.hex vstart=23505859 fullval-targets=238949 \
  d=256 layers=6 heads=8 ctx=64 batch=64 threads=8 dropout=0 seed=11 lr=1e-3 sched=cyc cycle=7200 cdecay=720 warm=0.000909 optim=muon mlr=0.01 ew=f32 \
  secs=39600 fullval-every=1800 periodic-full=1 ckpt=1 savefinal=1 tag=13-cyclic/v16384-cyc > $lg 2>&1
log "end v16384-cyc rc=$? $(grep -h 'trained\|FINAL' $lg | cut -c1-140 | tr '\n' ' ')"
log "QUEUE13 DONE (#13 cyclic)"
