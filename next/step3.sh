#!/bin/bash
# Step 3: after queue.sh finishes, if G512's subset val rose over time, run gru-drop at H=512 else at H=288 (which did overfit).
cd /home/box/lisp-charmodel/next
while ! grep -q "QUEUE DONE" queue.log; do sleep 15; done
H=$(python3 - <<'PY'
import csv
g=lambda x: float(x.replace('d0',''))
r=list(csv.DictReader(open('G512.csv'))); v=[g(x['val_subset']) for x in r]
print(512 if sum(v[-10:])/10 > min(v)+0.02 else 288)
PY
)
T=8; pgrep -f "science-team|[Ee]xperimenter" >/dev/null && T=4
echo "$(date '+%F %T %Z') start GD h=$H dropout=0.25 threads=$T load=$(cut -d' ' -f1-3 /proc/loadavg)" >> queue.log
FT=single MM=blas nice -n 10 sbcl --script gru-drop.lisp blas-threads=1 h=$H secs=1800 sched=cosine dropout=0.25 threads=$T tag=GD > GD.log 2>&1
echo "$(date '+%F %T %Z') end GD" >> queue.log
