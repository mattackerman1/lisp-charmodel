#!/bin/bash
# Sequential 1800 s jobs, one 8-thread job at a time; drop to 4 threads if science-team/Experimenter jobs appear.
cd /home/box/lisp-charmodel/next
thr() { if pgrep -f "science-team|[Ee]xperimenter" >/dev/null; then echo 4; else echo 8; fi; }
run() { tag=$1; shift; T=$(thr); echo "$(date '+%F %T %Z') start $tag threads=$T load=$(cut -d' ' -f1-3 /proc/loadavg)" >> queue.log
        "$@" threads=$T tag=$tag > $tag.log 2>&1; echo "$(date '+%F %T %Z') end $tag" >> queue.log; }
[ -f D1.log ] && grep -q "=== sample" D1.log || run D1 env FT=single nice -n 10 sbcl --script gpt-drop.lisp --run secs=1800 dropout=0.1
[ -f D2.log ] && grep -q "=== sample" D2.log || run D2 env FT=single nice -n 10 sbcl --script gpt-drop.lisp --run secs=1800 dropout=0.2
[ -f G288.log ] && grep -q "=== sample" G288.log || run G288 env FT=single MM=blas nice -n 10 sbcl --script gru-big.lisp blas-threads=1 h=288 secs=1800 sched=cosine
[ -f G512.log ] && grep -q "=== sample" G512.log || run G512 env FT=single MM=blas nice -n 10 sbcl --script gru-big.lisp blas-threads=1 h=512 secs=1800 sched=cosine
echo "$(date '+%F %T %Z') QUEUE DONE" >> queue.log
