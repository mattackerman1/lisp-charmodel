#!/bin/bash
# #19 step 2: ONE full-length (1 h-equivalent) run with the #19 step-1 code (gpt-eff.lisp = pre19 + change 1), ssm=1024 ssm-alpha=0.75,
# otherwise the #8 protocol (v16384 d256 L6 ctx64 batch64 seed1, Muon+WSD warm .01 decay .2, val-1M every 900 s train time, final full val).
# Deviation from #8: steps=15700 (step budget, 1 h at the measured 4.36 steps/s = 17,852 tok/s from 12 e19-c1 benches) instead of secs=3600,
# so the run is pause-tolerant (exact resume via hardening/check-runs.sh -> supervise.sh, state ckpt every 540 s wall).
# Idempotent: rerun after a box restart; a DONE marker skips it.
E=/home/box/lisp-charmodel/efficiency; H=$E/hardening; C=$E/19-full; r=ssm
ln -sfn $C/$r $H/runs.d/e19f-$r
while [ ! -f $C/$r/DONE ]; do
  [ -f $C/$r/STUCK ] && { echo "STUCK $r $(cat $C/$r/STUCK)"; exit 4; }
  ionice -c3 $H/check-runs.sh; sleep 120
done
rm -f $H/runs.d/e19f-$r; echo "$(date '+%F %T %Z') $r DONE: $(grep FINAL $C/$r/*.log | tail -1)"
echo QUEUE19F_DONE
