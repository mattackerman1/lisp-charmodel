#!/bin/bash
# all-lisp step 2b: short step-budgeted runs (run.sh ledger conventions), OpenBLAS vs pure-Lisp GEMM, same seed/args.
cd /home/box/lisp-charmodel/efficiency; source 17-method/common.sh
A="$BASE sched=wsd warm=0.01 decay=0.2 lrmin=0 steps=400 seed=1"
bash 17-method/run.sh alllisp-openblas-400 -- $A gemm=openblas
bash 17-method/run.sh alllisp-lisp-400 -- $A gemm=lisp
echo SHORT_RUNS_DONE
