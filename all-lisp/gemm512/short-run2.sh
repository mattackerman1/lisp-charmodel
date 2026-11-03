#!/bin/bash
# gap-closing final check: 400-step run (seed=1) with lisp512 defaults (gather packing + masked edges, KC256/MC144/NC1024),
# plus a same-session OpenBLAS rerun for a same-load ratio.
cd /home/box/lisp-charmodel/efficiency; source 17-method/common.sh
A="$BASE sched=wsd warm=0.01 decay=0.2 lrmin=0 steps=400 seed=1"
bash 17-method/run.sh alllisp-lisp512v2-400 -- $A gemm=lisp512
bash 17-method/run.sh alllisp-openblas-400-r2 -- $A gemm=openblas
echo SHORT_RUN_DONE
