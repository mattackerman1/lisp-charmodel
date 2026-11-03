#!/bin/bash
# #4 BPE queue (resumable: jobs whose log has "=== sample" are skipped).
cd /home/box/lisp-charmodel/efficiency
log() { echo "$(date '+%F %T %Z') $*" >> queue.log; }
while pgrep -f "queue3.sh" >/dev/null || pgrep -f "gpt-eff.lisp --run" >/dev/null; do
  pgrep -f "gpt-eff.lisp --run.*04-bpe" >/dev/null && { log "queue4: a 04-bpe job is already running, exiting"; exit 0; }
  sleep 30; done
run() { local name=$1 lg=$2; shift 2
  if grep -q "=== sample" "$lg" 2>/dev/null; then log "skip $name (done)"; return; fi
  log "start $name load=$(cut -d' ' -f1-3 /proc/loadavg)"
  nice -n 10 sbcl --dynamic-space-size 3000 --script gpt-eff.lisp --run "$@" > "$lg" 2>&1
  log "end $name rc=$? $(grep -h 'trained' $lg | cut -c1-120)"; python3 make_report.py >/dev/null 2>&1; }
# BPE-2048 tokens (train split only), ctx 64 tokens (~190 chars), 64 seqs x 64 = 4096 tokens/step (same tokens/step as before)
B="tokens=04-bpe/tokens2048.u16 vocab=04-bpe/vocab2048.hex vstart=30476268 fullval-targets=307043 d=256 layers=6 heads=8 ctx=64 batch=64 threads=8 dropout=0 seed=1 lr=1e-3 sched=wsd warm=0.01 decay=0.2 lrmin=0 optim=muon"
for mlr in 0.01 0.02; do
  run bpe-sweep-$mlr 04-bpe/sweep/bpe-muon-$mlr.log $B secs=900 fullval-every=0 final=0 mlr=$mlr tag=04-bpe/sweep/bpe-muon-$mlr
done
BEST=$(python3 -c "
import glob,re
r=[(float(open(f).read().strip().split('\n')[-1].split(',')[2].replace('d0','')),re.search(r'muon-(.*)-val1m',f)[1]) for f in glob.glob('04-bpe/sweep/bpe-muon-*-val1m.csv')]
print(min(r)[1])")
log "queue4: best BPE Muon lr = $BEST"; echo "$BEST" > 04-bpe/best_mlr.txt
# 3-min speed probe of the #2 char config right before the 1-h run (box-speed reference)
run speed-probe4 04-bpe/speed-probe-char.log data=../huge/corpus.txt d=256 layers=6 heads=8 ctx=128 batch=32 threads=8 dropout=0 seed=1 fullval-targets=1000000 lr=1e-3 sched=wsd warm=0.01 decay=0.2 lrmin=0 optim=muon mlr=0.01 secs=180 fullval-every=0 final=0 tag=04-bpe/speed-probe-char
run bpe-full 04-bpe/bpe.log $B secs=3600 fullval-every=900 mlr=$BEST tag=04-bpe/bpe
log "QUEUE4 DONE (#4 BPE)"
