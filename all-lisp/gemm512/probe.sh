#!/bin/bash
# per-change throughput probe: 120-step training (same BASE args as the 400-step check, final=0), prints training tok/s.
# usage: probe.sh label gemm [ENV=VAL ...]
cd /home/box/lisp-charmodel/efficiency; source 17-method/common.sh; source hardening/env.sh
label=$1; g=$2; shift 2
mkdir -p ../all-lisp/gemm512/probes
env "$@" nice -n 10 $SBCL --dynamic-space-size 4000 --script gpt-eff.lisp --run $BASE sched=wsd warm=0.01 decay=0.2 lrmin=0 steps=120 seed=1 final=0 gemm=$g tag=../all-lisp/gemm512/probes/$label > ../all-lisp/gemm512/probes/$label.log 2>&1
echo "$label gemm=$g $* : $(grep -o '[0-9.]* tokens/s' ../all-lisp/gemm512/probes/$label.log)"
