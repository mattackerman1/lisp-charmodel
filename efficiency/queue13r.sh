#!/bin/bash
# #13 resume 1 (after the VM restart ~08:11-08:18 CDT Oct 21): continue the cyclic run from 13-cyclic/v16384-cyc-ckpt-056223.f32 (t=21600.2 s, cycle 3 constant phase).
# Per queue13.sh's resume recipe: init=<ckpt> toff=<its train_seconds> secs=39600-toff warm=0. Weights only: Adam moments + Muon momentum RESET here.
# seed=12 so the resumed segment does not replay the batch order of the first segment. New tag (-r1) keeps the original logs intact.
cd /home/box/lisp-charmodel/efficiency
. hardening/env.sh   # HARDENING: sbcl/OpenBLAS/matplotlib under /home/box/opt (survive box restarts)
hardening/preflight.sh >> queue.log 2>&1 || { echo "$(date '+%F %T %Z') queue13r.sh: PREFLIGHT FAILED, not starting (see queue.log)" >> queue.log; exit 2; }
log() { echo "$(date '+%F %T %Z') $*" >> queue.log; }
pgrep -f "gpt-eff.lisp --run.*13-cyclic" >/dev/null && { log "queue13r: a 13-cyclic job is already running, exiting"; exit 0; }
while pgrep -f "gpt-eff.lisp --run" >/dev/null; do sleep 30; done
lg=13-cyclic/v16384-cyc-r1.log
grep -q "=== sample" $lg 2>/dev/null && { log "skip v16384-cyc-r1 (done)"; exit 0; }
t=0; while [ $(awk '/MemAvail/{print int($2/1024)}' /proc/meminfo) -lt 3000 ]; do [ $t -eq 0 ] && log "queue13r: waiting for 3000MB free"; sleep 60; t=$((t+60)); [ $t -ge 10800 ] && { log "queue13r: stop, not enough RAM"; exit 1; }; done
log "start v16384-cyc-r1 (resume #13 from ckpt 056223, toff=21600.2, 18000 s left) load=$(cut -d' ' -f1-3 /proc/loadavg) memavail=$(awk '/MemAvail/{print int($2/1024)}' /proc/meminfo)MB"
nice -n 10 "$SBCL" --dynamic-space-size 4000 --script gpt-eff.lisp --run tokens=08-vocab-up/tokens16384.u16 vocab=08-vocab-up/vocab16384.hex vstart=23505859 fullval-targets=238949 \
  d=256 layers=6 heads=8 ctx=64 batch=64 threads=8 dropout=0 seed=12 lr=1e-3 sched=cyc cycle=7200 cdecay=720 warm=0 optim=muon mlr=0.01 ew=f32 \
  init=13-cyclic/v16384-cyc-ckpt-056223.f32 toff=21600.2 secs=18000 fullval-every=1800 periodic-full=1 ckpt=1 savefinal=1 tag=13-cyclic/v16384-cyc-r1 > $lg 2>&1
rc=$?
log "end v16384-cyc-r1 rc=$rc $(grep -h 'trained\|FINAL' $lg | cut -c1-140 | tr '\n' ' ')"
if [ $rc -eq 0 ]; then log "QUEUE13 DONE (#13 cyclic, resumed)"; else log "QUEUE13 FAILED rc=$rc (#13 cyclic resume)"; exit $rc; fi
