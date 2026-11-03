#!/bin/bash
# Efficiency series queue (resumable: finished jobs -- log contains "=== sample" -- are skipped; just re-run after an interruption).
# One job at a time, 8 threads, nice -n 10. Model: d256 L6 H8 ctx128 dropout 0, batch 32x128, 100 MB corpus (huge/corpus.txt).
cd /home/box/lisp-charmodel/efficiency
log() { echo "$(date '+%F %T %Z') $*" >> queue.log; }
if pgrep -f "gpt-eff.lisp --run" >/dev/null; then log "queue: a job is already running, exiting"; exit 0; fi
run() {  # name, logfile, args...
  local name=$1 lg=$2; shift 2
  if grep -q "=== sample" "$lg" 2>/dev/null; then log "skip $name (done)"; return; fi
  if pgrep -f "science-team|[Ee]xperimenter" >/dev/null; then log "WARNING science/experimenter job running"; fi
  log "start $name load=$(cut -d' ' -f1-3 /proc/loadavg)"
  nice -n 10 sbcl --dynamic-space-size 3000 --script gpt-eff.lisp --run "$@" > "$lg" 2>&1
  log "end $name rc=$? $(grep -h 'trained' $lg | cut -c1-120)"
}
M="data=../huge/corpus.txt d=256 layers=6 heads=8 ctx=128 batch=32 threads=8 dropout=0 seed=1 fullval-targets=1000000"
# 0. baseline: today's config, schedule fit to 1 h
run baseline baseline/baseline.log $M secs=3600 fullval-every=900 lr=1e-3 sched=cosine warm=0.02 lrmin=0.1 tag=baseline/baseline
# 1a. LR sweep, 15 min each, short warmup (1%) + constant + fast linear decay to 0 over the last 20% (WSD); scored on val(first 1M) at the end
for lr in 1e-3 2e-3 3e-3 5e-3; do
  run sweep-wsd-$lr 01-lr/sweep/wsd-$lr.log $M secs=900 fullval-every=0 final=0 lr=$lr sched=wsd warm=0.01 decay=0.2 lrmin=0 tag=01-lr/sweep/wsd-$lr
done
run sweep-cos-1e-3 01-lr/sweep/cos-1e-3.log $M secs=900 fullval-every=0 final=0 lr=1e-3 sched=cosine warm=0.02 lrmin=0.1 tag=01-lr/sweep/cos-1e-3
# 1b. full 1 h run with the best swept WSD peak LR
BEST=$(python3 - <<'PY'
import glob, re
best = None
for f in glob.glob('01-lr/sweep/wsd-*-val1m.csv'):
    v = float(open(f).read().strip().split('\n')[-1].split(',')[2].replace('d0', ''))
    lr = re.search(r'wsd-(.*)-val1m', f)[1]
    if best is None or v < best[0]: best = (v, lr)
print(best[1])
PY
)
log "best sweep LR = $BEST"; echo "$BEST" > 01-lr/best_lr.txt
run lr-full 01-lr/lr-wsd.log $M secs=3600 fullval-every=900 lr=$BEST sched=wsd warm=0.01 decay=0.2 lrmin=0 tag=01-lr/lr-wsd
log "QUEUE DONE (baseline + #1)"
