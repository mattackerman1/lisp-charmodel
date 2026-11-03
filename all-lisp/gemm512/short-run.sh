#!/bin/bash
# AVX-512 Lisp GEMM: same 400-step run as all-lisp/gemm/short-runs.sh (seed=1), gemm=lisp512.
cd /home/box/lisp-charmodel/efficiency; source 17-method/common.sh
A="$BASE sched=wsd warm=0.01 decay=0.2 lrmin=0 steps=400 seed=1"
bash 17-method/run.sh alllisp-lisp512-400 -- $A gemm=lisp512
echo SHORT_RUN_DONE
