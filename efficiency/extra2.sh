#!/bin/bash
# Extra Muon sweep point below the range (0.01 is at the low edge). Runs after queue2 is done.
cd /home/box/lisp-charmodel/efficiency
until grep -q "QUEUE2 DONE" queue.log; do sleep 60; done
lg=02-muon/sweep/muon-0.005.log; grep -q "=== sample" $lg 2>/dev/null && exit 0
echo "$(date '+%F %T %Z') start extra muon-sweep-0.005" >> queue.log
nice -n 10 sbcl --dynamic-space-size 3000 --script gpt-eff.lisp --run data=../huge/corpus.txt d=256 layers=6 heads=8 ctx=128 batch=32 threads=8 dropout=0 seed=1 fullval-targets=1000000 lr=1e-3 sched=wsd warm=0.01 decay=0.2 lrmin=0 secs=900 fullval-every=0 final=0 optim=muon mlr=0.005 tag=02-muon/sweep/muon-0.005 > $lg 2>&1
echo "$(date '+%F %T %Z') end extra muon-sweep-0.005 $(grep trained $lg | cut -c1-100)" >> queue.log
python3 make_report.py >/dev/null 2>&1
