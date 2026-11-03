#!/bin/bash
# #5 low-precision queue (bf16 AMX rejected: silent corruption on this VM; runs the f32-GELU alternative) (resumable: jobs whose log has "=== sample" are skipped). Waits for the curve validation first.
cd /home/box/lisp-charmodel/efficiency
export LD_LIBRARY_PATH=/home/box/lisp-charmodel/venv/lib MKL_THREADING_LAYER=SEQUENTIAL
log() { echo "$(date '+%F %T %Z') $*" >> queue.log; }
while pgrep -f "curves.sh|queue4.sh" >/dev/null || pgrep -f "gpt-eff.lisp --run" >/dev/null; do
  pgrep -f "gpt-eff.lisp --run.*05-precision/(ew32|speed)" >/dev/null && { log "queue5: a 05 job is already running, exiting"; exit 0; }
  sleep 30; done
run() { local name=$1 lg=$2; shift 2
  if grep -q "=== sample" "$lg" 2>/dev/null; then log "skip $name (done)"; return; fi
  log "start $name load=$(cut -d' ' -f1-3 /proc/loadavg)"
  nice -n 10 sbcl --dynamic-space-size 3000 --script gpt-eff.lisp --run "$@" > "$lg" 2>&1
  log "end $name rc=$? $(grep -h 'trained' $lg | cut -c1-120)"; python3 make_report.py >/dev/null 2>&1; }
B="tokens=04-bpe/tokens2048.u16 vocab=04-bpe/vocab2048.hex vstart=30476268 fullval-targets=307043 d=256 layers=6 heads=8 ctx=64 batch=64 threads=8 dropout=0 seed=1 lr=1e-3 sched=wsd warm=0.01 decay=0.2 lrmin=0 optim=muon mlr=0.01"
run speed-probe5 05-precision/speed-probe-fp32.log $B secs=180 fullval-every=0 final=0 prec=fp32 tag=05-precision/speed-probe-fp32
run ew32-full 05-precision/ew32.log $B secs=3600 fullval-every=900 ew=f32 tag=05-precision/ew32
log "QUEUE5 DONE (#5 f32 GELU)"
