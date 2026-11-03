#!/bin/bash
. /home/box/lisp-charmodel/efficiency/hardening/env.sh; cd /home/box/lisp-charmodel/efficiency/15-macro
for o in "adamw 0" "muon 0.1"; do set -- $o; for f in gpt-eff gpt-eff-macro; do nice -n 10 $SBCL --script stepdriver.lisp ../$f.lisp $1 $2 2>/dev/null | grep -v "^;" > steps-$1-$f.txt; done
  cmp steps-$1-gpt-eff.txt steps-$1-gpt-eff-macro.txt && echo "stepdriver $1 dropout=$2: 60 per-step losses + grad norms + weight hash IDENTICAL"; done
A1="d=128 layers=4 ctx=64 batch=32 threads=8 steps=600 secs=100000 warm=0 sched=const final=0 savefinal=1 dropout=0.1 optim=muon"
A2="d=128 layers=4 ctx=64 batch=32 threads=8 steps=600 secs=100000 warm=0 sched=const final=0 savefinal=1 ew=f32 ctx1=32 posoff=1"
for cfg in 1 2; do eval A=\$A$cfg; for f in gpt-eff gpt-eff-macro; do nice -n 10 $SBCL --script ../$f.lisp --run data=../../input.txt tag=main$cfg-$f $A > main$cfg-$f.log 2>&1; done
diff <(grep -E "^step|VAL1M" main$cfg-gpt-eff.log | sed -E 's/(t|wall)= *[0-9.]+s//g') <(grep -E "^step|VAL1M" main$cfg-gpt-eff-macro.log | sed -E 's/(t|wall)= *[0-9.]+s//g') && echo "main cfg$cfg ($A): logged losses identical"
cmp main$cfg-gpt-eff-final-000600.f32 main$cfg-gpt-eff-macro-final-000600.f32 && echo "main cfg$cfg: final weights bitwise identical"
cmp <(sed -n '/=== sample/,$p' main$cfg-gpt-eff.log) <(sed -n '/=== sample/,$p' main$cfg-gpt-eff-macro.log) && echo "main cfg$cfg: sample text identical"; done
./bench.sh
