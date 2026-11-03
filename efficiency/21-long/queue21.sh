#!/bin/bash
# #21: full-softmax control for #20: identical flags (100,604 steps, warm=0.0016, wsd dsteps=2996 dshape=sqrt, seed 1, gemm=lisp512) minus ssm=1024/ssm-alpha.
# Pause-tolerant: hardening/check-runs.sh -> supervise.sh (state ckpt every 540 s wall, exact resume). Idempotent: rerun after a box restart.
# On DONE: appends results.tsv row (append-row.sh) and prints QUEUE21_DONE.
E=/home/box/lisp-charmodel/efficiency; H=$E/hardening; C=$E/21-long; R=$C/run
ln -sfn $R $H/runs.d/e21-long
while [ ! -f $R/DONE ]; do
  [ -f $R/STUCK ] && { echo "STUCK $R $(cat $R/STUCK)"; exit 4; }
  ionice -c3 $H/check-runs.sh; sleep 120
done
echo "$(date '+%F %T %Z') DONE: $(grep -h FINAL $R/*.log | tail -1)"
$C/append-row.sh $R && echo "results.tsv row appended" || echo "ROW APPEND FAILED rc=$?"
rm -f $H/runs.d/e21-long
echo QUEUE21_DONE
