#!/bin/bash
# #17 step 2+3: stack kept decay tricks, then the final matched-compute experiment, then qknorm 2nd seed.
cd /home/box/lisp-charmodel/efficiency; . 17-method/common.sh
run() { id=$1; shift; grep -q " END $id " 17-method/queue.log && { echo "skip $id (END exists)"; return; }; 17-method/run.sh $id -- "$@"; }
STACK="dshape=sqrt elr=3e-4"
D7=${DECAY/mlr=0.01/mlr=0.007}
# (2) stack on the 2 h branch decay (vs decay-base-s11 1.0332, bar 1.0326, best single decay-sqrt 1.0314)
run decay-stack ${D7} seed=11 $STACK
fs=$(awk -F'\t' '$1=="decay-stack"{print $14}' 17-method/results.tsv)
if awk -v s="$fs" 'BEGIN{exit !(s!="" && s<1.0314)}'; then FD="$D7"; FX="$STACK"; pick=stack; else FD="$DECAY"; FX="dshape=sqrt"; pick=sqrt-only; fi
echo "$(date '+%F %T %Z') FINAL pick=$pick (decay-stack full=$fs)" >> 17-method/queue.log
# (3) final: branch decay from #10's 10 h ckpt (step 97,608) for 2,996 steps -> 100,604 total steps = #13's 11 h run (1.0152): same compute
FD=${FD/init=10-long\/v16384-const-ckpt-020410.f32/init=10-long/v16384-const-ckpt-097608.f32}
run final-10h-stack ${FD/steps=1868/steps=2996} seed=11 $FX val1m-steps=1000 savefinal=1
# marginal proxy keep: 2nd seed (vs proxy-base-s2 1.1248)
run proxy-qknorm-s2 $PROXY seed=2 qknorm=1 qkg=2
echo Q6 DONE >> 17-method/queue.log
