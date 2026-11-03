#!/bin/bash
# alternating speed benchmark: 4 rounds x (old, new), 60 s training each, nice 10, 8 threads
. /home/box/lisp-charmodel/efficiency/hardening/env.sh
cd /home/box/lisp-charmodel/efficiency/15-macro
A="d=128 layers=4 heads=4 ctx=64 batch=32 threads=8 secs=60 sched=wsd final=0 data=../../input.txt"
for r in 1 2 3 4; do
  for f in gpt-eff gpt-eff-macro; do
    if [ $((r%2)) = 0 ]; then f=$([ $f = gpt-eff ] && echo gpt-eff-macro || echo gpt-eff); fi
    nice -n 10 $SBCL --script ../$f.lisp --run tag=bench-$f-r$r $A > bench-$f-r$r.log 2>&1
    echo "round $r $f $(grep '^trained' bench-$f-r$r.log)"
  done
done
