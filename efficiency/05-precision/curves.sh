#!/bin/bash
# Numerics validation: 1000 steps, identical seed/data order, constant LR (warm=0, step-based), fp32 vs mklf32 vs bf16.
cd /home/box/lisp-charmodel/efficiency
export LD_LIBRARY_PATH=/home/box/lisp-charmodel/venv/lib MKL_THREADING_LAYER=SEQUENTIAL
for pm in fp32 mklf32 bf16; do
  lg=05-precision/curves/$pm.log; grep -q "=== sample" $lg 2>/dev/null && continue
  nice -n 10 sbcl --dynamic-space-size 3000 --script gpt-eff.lisp --run tokens=04-bpe/tokens2048.u16 vocab=04-bpe/vocab2048.hex vstart=30476268 fullval-targets=307043 d=256 layers=6 heads=8 ctx=64 batch=64 threads=8 dropout=0 seed=1 lr=1e-3 sched=wsd warm=0 lrmin=0 optim=muon mlr=0.01 secs=100000 steps=1000 fullval-every=0 final=0 prec=$pm tag=05-precision/curves/$pm > $lg 2>&1
done
echo CURVES DONE >> 05-precision/curves/done.txt
