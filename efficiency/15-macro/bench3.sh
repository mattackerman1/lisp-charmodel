#!/bin/bash
# alternating timed-block benchmark (robust to box stalls: compare min / median 100-step block), nice 10, 8 threads
. /home/box/lisp-charmodel/efficiency/hardening/env.sh; cd /home/box/lisp-charmodel/efficiency/15-macro; mkdir -p bench3
for r in 1 2 3 4 5 6; do for f in gpt-eff gpt-eff-macro; do
  if [ $((r%2)) = 0 ]; then f=$([ $f = gpt-eff ] && echo gpt-eff-macro || echo gpt-eff); fi
  nice -n 10 $SBCL --script timedriver.lisp ../$f.lisp adamw 0 2>/dev/null | grep -v "^;" > bench3/$f-r$r.txt; echo "round $r $f $(grep block bench3/$f-r$r.txt | awk '{printf "%s ",$2}')"
done; done
