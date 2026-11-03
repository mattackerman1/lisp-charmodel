#!/bin/bash
# #7 BPE vocab sweep (resumable). Base = #5 (BPE-2048, d256-L6, Muon 0.01, WSD, ew=f32, ctx 64 tokens, batch 64).
# Per vocab: tokenizer (skipped if meta exists) -> 3-min speed probe at the exact #5 config -> 1-hour run at vocab V.
cd /home/box/lisp-charmodel/efficiency
log() { echo "$(date '+%F %T %Z') $*" >> queue.log; }
while pgrep -f "queue6.sh" >/dev/null || pgrep -f "gpt-eff.lisp --run" >/dev/null; do
  pgrep -f "gpt-eff.lisp --run.*07-vocab" >/dev/null && { log "queue7: a 07-vocab job is already running, exiting"; exit 0; }
  sleep 30; done
run() { local name=$1 lg=$2; shift 2
  if grep -q "=== sample" "$lg" 2>/dev/null; then log "skip $name (done)"; return; fi
  log "start $name load=$(cut -d' ' -f1-3 /proc/loadavg) memavail=$(awk '/MemAvail/{print int($2/1024)}' /proc/meminfo)MB"
  nice -n 10 sbcl --dynamic-space-size 4000 --script gpt-eff.lisp --run "$@" > "$lg" 2>&1
  log "end $name rc=$? $(grep -h 'trained' $lg | cut -c1-120)"; python3 make_report.py >/dev/null 2>&1; }
M="d=256 layers=6 heads=8 ctx=64 batch=64 threads=8 dropout=0 seed=1 lr=1e-3 sched=wsd warm=0.01 decay=0.2 lrmin=0 optim=muon mlr=0.01 ew=f32"
P5="tokens=04-bpe/tokens2048.u16 vocab=04-bpe/vocab2048.hex vstart=30476268 fullval-targets=307043"
for V in 512 1024 4096 8192; do
  if [ ! -f 07-vocab/meta$V.json ]; then
    log "start tokenizer V=$V"
    (cd 07-vocab && nice -n 10 ../../venv/bin/python train_bpe.py $V > tok$V.log 2>&1); log "end tokenizer V=$V rc=$? $(tail -1 07-vocab/tok$V.log | cut -c1-200)"
  fi
  [ -f 07-vocab/meta$V.json ] || { log "queue7: tokenizer V=$V failed, stopping"; exit 1; }
  read TT V1 <<<"$(python3 -c "import json;m=json.load(open('07-vocab/meta$V.json'));print(m['train_tokens'],m['val1m_tokens'])")"
  run probe-v$V 07-vocab/probe-before-v$V.log $P5 $M secs=180 fullval-every=0 final=0 tag=07-vocab/probe-before-v$V
  run vocab-v$V 07-vocab/v$V.log tokens=07-vocab/tokens$V.u16 vocab=07-vocab/vocab$V.hex vstart=$TT fullval-targets=$V1 $M secs=3600 fullval-every=900 tag=07-vocab/v$V
done
log "QUEUE7 DONE (#7 vocab sweep)"
