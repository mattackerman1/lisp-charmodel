#!/bin/bash
# #16: decay-length sweep. Same args as queue12.sh 2h; only secs= changes (fullval-every stays 360). Sequential.
cd /home/box/lisp-charmodel/efficiency
source hardening/env.sh
log() { echo "$(date '+%F %T %Z') $*" >> queue.log; }
hardening/preflight.sh > 16-decay-length/preflight.txt 2>&1 || { log "queue16: preflight FAIL"; exit 2; }
for s in 360 720 1440 2880; do
  tag=16-decay-length/v16384-decay-2h-$s; lg=$tag.log
  grep -q "=== sample" $lg 2>/dev/null && continue
  for try in 1 2; do
    while pgrep -f "gpt-eff.lisp --run" >/dev/null; do sleep 30; done
    log "start #16 decay $s s try $try load=$(cut -d' ' -f1-3 /proc/loadavg)"
    nice -n 10 $SBCL --dynamic-space-size 4000 --script gpt-eff.lisp --run tokens=08-vocab-up/tokens16384.u16 vocab=08-vocab-up/vocab16384.hex vstart=23505859 fullval-targets=238949 \
      d=256 layers=6 heads=8 ctx=64 batch=64 threads=8 dropout=0 seed=11 lr=1e-3 sched=lin warm=0 optim=muon mlr=0.01 ew=f32 \
      init=10-long/v16384-const-ckpt-020410.f32 secs=$s fullval-every=360 periodic-full=1 ckpt=1 savefinal=1 tag=$tag > $lg 2>&1
    rc=$?; log "end #16 decay $s rc=$rc $(grep -h 'trained\|FINAL' $lg | cut -c1-140 | tr '\n' ' ')"
    grep -q "=== sample" $lg && break
    [ $try -eq 2 ] && { log "queue16: decay $s crashed twice, STOP"; echo STUCK > 16-decay-length/STUCK; exit 1; }
    mv $lg $lg.crash1
  done
done
log "QUEUE16 DONE"
