#!/bin/bash
# #8 conditional vocab sweep upward (resumable). Base/best-so-far = v8192 (#7, full val 1.0582). Setup identical to #7.
# For V = 16384, 32768, 65536: RAM check -> tokenizer -> RAM check -> 3-min #5-config probe -> 1-hour run.
# Continue to the next doubling only if full val improved on the best so far; stop on no improvement, failure, RAM shortfall, or V > 65536 (uint16 ids).
cd /home/box/lisp-charmodel/efficiency
log() { echo "$(date '+%F %T %Z') $*" >> queue.log; }
while pgrep -f "queue7.sh" >/dev/null || pgrep -f "gpt-eff.lisp --run" >/dev/null; do
  pgrep -f "gpt-eff.lisp --run.*08-vocab-up" >/dev/null && { log "queue8: an 08 job is already running, exiting"; exit 0; }
  sleep 30; done
# waitmem NEED LABEL: wait up to 3 h (poll 60 s) for NEED MB available; return 1 if it never frees up (memory is used by processes outside this queue)
waitmem() { local t=0; while [ $(avail) -lt $1 ]; do [ $t -eq 0 ] && log "queue8: waiting for ${1}MB free for $2 (now $(avail)MB)"; sleep 60; t=$((t+60)); [ $t -ge 10800 ] && return 1; done; return 0; }
avail() { awk '/MemAvail/{print int($2/1024)}' /proc/meminfo; }
fullval() { tail -1 "$1" | cut -d, -f4 | sed 's/d0//'; }
run() { local name=$1 lg=$2 dyn=$3; shift 3
  if grep -q "=== sample" "$lg" 2>/dev/null; then log "skip $name (done)"; return 0; fi
  log "start $name load=$(cut -d' ' -f1-3 /proc/loadavg) memavail=$(avail)MB dyn=${dyn}MB"
  nice -n 10 sbcl --dynamic-space-size $dyn --script gpt-eff.lisp --run "$@" > "$lg" 2>&1; local rc=$?
  log "end $name rc=$rc $(grep -h 'trained' $lg | cut -c1-120)"; python3 make_report.py >/dev/null 2>&1; return $rc; }
M="d=256 layers=6 heads=8 ctx=64 batch=64 threads=8 dropout=0 seed=1 lr=1e-3 sched=wsd warm=0.01 decay=0.2 lrmin=0 optim=muon mlr=0.01 ew=f32"
P5="tokens=04-bpe/tokens2048.u16 vocab=04-bpe/vocab2048.hex vstart=30476268 fullval-targets=307043"
best=$(fullval 07-vocab/v8192-final.csv); bestname=v8192
for V in 16384 32768 65536 131072; do
  if [ $V -gt 65536 ]; then log "queue8: stop, V=$V exceeds uint16 token ids"; break; fi
  # tokenizer RAM: ~corpus strings + BPE pair tables + one 8-piece encode batch; require 2.5 GB free (peak RSS logged in tok$V.log)
  if [ ! -f 08-vocab-up/meta$V.json ]; then
    waitmem 2500 "tokenizer V=$V" || { log "queue8: stop, <2500MB free for 3 h before tokenizer V=$V"; break; }
    log "start tokenizer V=$V memavail=$(avail)MB"
    (cd 08-vocab-up && nice -n 10 ../../venv/bin/python train_bpe.py $V > tok$V.log 2>&1); log "end tokenizer V=$V rc=$? $(tail -1 08-vocab-up/tok$V.log | cut -c1-200)"
    [ -f 08-vocab-up/meta$V.json ] || { log "queue8: stop, tokenizer V=$V failed"; break; }
  fi
  # trainer RAM: logits/probs 4096 rows x V f32 (16 KB per vocab entry) + wte/head params, grads, Adam m/v and 8 per-worker grad copies (~12 x 256 x 4 B per entry) + ~1 GB base
  need=$(( 1000 + V * (16384 + 12288) / 1000000 )); dyn=$(( need + 1500 > 4000 ? need + 1500 : 4000 ))
  waitmem $((need + 1500)) "v$V run (est ${need}MB + 1500 margin)" || { log "queue8: stop, <$((need + 1500))MB free for 3 h before v$V"; break; }
  read TT V1 <<<"$(python3 -c "import json;m=json.load(open('08-vocab-up/meta$V.json'));print(m['train_tokens'],m['val1m_tokens'])")"
  run probe-v$V 08-vocab-up/probe-before-v$V.log 4000 $P5 $M secs=180 fullval-every=0 final=0 tag=08-vocab-up/probe-before-v$V
  run vocab-v$V 08-vocab-up/v$V.log $dyn tokens=08-vocab-up/tokens$V.u16 vocab=08-vocab-up/vocab$V.hex vstart=$TT fullval-targets=$V1 $M secs=3600 fullval-every=900 tag=08-vocab-up/v$V \
    || { log "queue8: stop, v$V run failed"; break; }
  fv=$(fullval 08-vocab-up/v$V-final.csv)
  if python3 -c "import sys; sys.exit(0 if float('$fv') < float('$best') else 1)"; then
    log "queue8: v$V full val $fv improved on $bestname $best -> continue"; best=$fv; bestname=v$V
  else log "queue8: v$V full val $fv did not improve on $bestname $best -> stop"; break; fi
done
log "QUEUE8 DONE (#8 vocab-up sweep; best $bestname $best)"
