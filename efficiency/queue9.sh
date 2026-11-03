#!/bin/bash
# #9 dropout on the best vocab so far (resumable). Waits for the v32768 run (#8) to finish, picks v16384 or v32768 by full val,
# then runs it with dropout=0.1, otherwise identical (d256-L6, ctx 64, Muon 0.01, WSD, ew=f32, 1 h). queue8 was stopped (no 65536).
cd /home/box/lisp-charmodel/efficiency
log() { echo "$(date '+%F %T %Z') $*" >> queue.log; }
pgrep -f "gpt-eff.lisp --run.*09-dropout" >/dev/null && { log "queue9: a 09 job is already running, exiting"; exit 0; }
while pgrep -f "gpt-eff.lisp --run" >/dev/null; do sleep 30; done
[ -f 08-vocab-up/v32768-final.csv ] && ! grep -q "end vocab-v32768" queue.log && log "end vocab-v32768 (finished after queue8 was stopped) $(grep -h 'trained' 08-vocab-up/v32768.log | cut -c1-120)"
python3 make_report.py >/dev/null 2>&1
fv() { [ -f "$1" ] && tail -1 "$1" | cut -d, -f4 | sed 's/d0//' || echo 9; }
a=$(fv 08-vocab-up/v16384-final.csv); b=$(fv 08-vocab-up/v32768-final.csv)
if python3 -c "import sys; sys.exit(0 if float('$b') < float('$a') else 1)"; then V=32768; else V=16384; fi
log "queue9: v16384 $a vs v32768 $b -> dropout run on v$V"
avail() { awk '/MemAvail/{print int($2/1024)}' /proc/meminfo; }
need=$(( 1000 + V * (16384 + 12288) / 1000000 + 1500 )); t=0
while [ $(avail) -lt $need ]; do [ $t -eq 0 ] && log "queue9: waiting for ${need}MB free"; sleep 60; t=$((t+60)); [ $t -ge 10800 ] && { log "queue9: stop, not enough RAM for 3 h"; exit 1; }; done
read TT V1 <<<"$(python3 -c "import json;m=json.load(open('08-vocab-up/meta$V.json'));print(m['train_tokens'],m['val1m_tokens'])")"
M="d=256 layers=6 heads=8 ctx=64 batch=64 threads=8 dropout=0.1 seed=1 lr=1e-3 sched=wsd warm=0.01 decay=0.2 lrmin=0 optim=muon mlr=0.01 ew=f32"
lg=09-dropout/v$V-drop0.1.log
if grep -q "=== sample" $lg 2>/dev/null; then log "skip v$V-drop0.1 (done)"; else
  echo $V > 09-dropout/vocab.txt
  log "start v$V-drop0.1 load=$(cut -d' ' -f1-3 /proc/loadavg) memavail=$(avail)MB"
  nice -n 10 sbcl --dynamic-space-size 4000 --script gpt-eff.lisp --run tokens=08-vocab-up/tokens$V.u16 vocab=08-vocab-up/vocab$V.hex vstart=$TT fullval-targets=$V1 $M secs=3600 fullval-every=900 tag=09-dropout/v$V-drop0.1 > $lg 2>&1
  log "end v$V-drop0.1 rc=$? $(grep -h 'trained' $lg | tail -1 | cut -c1-120)"; python3 make_report.py >/dev/null 2>&1; fi
log "QUEUE9 DONE (#9 dropout)"
