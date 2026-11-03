#!/bin/bash
# #18 1-hour confirm (approved by Matthew 2026-10-26 13:35 CT). Sequential, pause-tolerant (state ckpt every 540 s wall, exact resume
# via hardening/check-runs.sh -> supervise.sh). Step budgets = 1 h of clean training time at measured rates:
#   ssm1024: ~3.6 steps/s (e18-v16-s1024 clean segments) -> 12960; full softmax: 2500/1174.1 s (e18-proxy-base) -> 7666.
# Idempotent: rerun this script after a box restart; finished runs (DONE marker) are skipped.
E=/home/box/lisp-charmodel/efficiency; H=$E/hardening; C=$E/18-confirm
for r in ssm base; do
  ln -sfn $C/$r $H/runs.d/e18c-$r
  while [ ! -f $C/$r/DONE ]; do
    [ -f $C/$r/STUCK ] && { echo "STUCK $r $(cat $C/$r/STUCK)"; exit 4; }
    ionice -c3 $H/check-runs.sh; sleep 120
  done
  rm -f $H/runs.d/e18c-$r; echo "$(date '+%F %T %Z') $r DONE: $(grep FINAL $C/$r/*.log | tail -1)"
done
echo QUEUE18C_DONE
