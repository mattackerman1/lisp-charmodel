#!/bin/bash
# Extra sweep point below the range (best LR 1e-3 was at the low edge). Runs only after the main queue is done.
cd /home/box/lisp-charmodel/efficiency
until grep -q "QUEUE DONE" queue.log; do sleep 60; done
lg=01-lr/sweep/wsd-5e-4.log
grep -q "=== sample" $lg 2>/dev/null && exit 0
echo "$(date '+%F %T %Z') start extra sweep-wsd-5e-4" >> queue.log
nice -n 10 sbcl --dynamic-space-size 3000 --script gpt-eff.lisp --run data=../huge/corpus.txt d=256 layers=6 heads=8 ctx=128 batch=32 threads=8 dropout=0 seed=1 fullval-targets=1000000 secs=900 fullval-every=0 final=0 lr=5e-4 sched=wsd warm=0.01 decay=0.2 lrmin=0 tag=01-lr/sweep/wsd-5e-4 > $lg 2>&1
echo "$(date '+%F %T %Z') end extra sweep-wsd-5e-4 $(grep trained $lg | cut -c1-100)" >> queue.log
python3 make_report.py > /dev/null 2>&1
