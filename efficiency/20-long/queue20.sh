#!/bin/bash
# #20: ONE long run (100,604 steps = #17's total) with the #19 code (gpt-eff.lisp sha ed0f085f64273d89), ssm=1024 ssm-alpha=0.75 v16384 gemm=lisp512.
# Single run, step budget: warmup -> constant LR -> last 2,996 steps sqrt-shaped decay to 0 (wsd + dsteps + dshape=sqrt; same schedule code as #17's branch decay).
# Pause-tolerant: hardening/check-runs.sh -> supervise.sh (state ckpt every 540 s wall, exact resume). Idempotent: rerun after a box restart.
# On DONE: appends results.tsv row (append-row.sh) and prints QUEUE20_DONE.
E=/home/box/lisp-charmodel/efficiency; H=$E/hardening; C=$E/20-long; R=$C/run
ln -sfn $R $H/runs.d/e20-long
while [ ! -f $R/DONE ]; do
  [ -f $R/STUCK ] && { echo "STUCK $R $(cat $R/STUCK)"; exit 4; }
  ionice -c3 $H/check-runs.sh; sleep 120
done
echo "$(date '+%F %T %Z') DONE: $(grep -h FINAL $R/*.log | tail -1)"
$C/append-row.sh $R && echo "results.tsv row appended" || echo "ROW APPEND FAILED rc=$?"
rm -f $H/runs.d/e20-long
echo QUEUE20_DONE
