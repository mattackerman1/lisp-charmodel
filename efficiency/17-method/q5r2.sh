#!/bin/bash
# Rerun of the Muon-LR decay tricks: decay-mlr0.007/0.014 were no-ops because DECAY already contains mlr=0.01
# and gpt-eff's opt() takes the FIRST occurrence (their curves are identical to decay-base-s11). Fix: substitute in place.
cd /home/box/lisp-charmodel/efficiency; . 17-method/common.sh
until grep -q "Q5R DONE" 17-method/queue.log; do sleep 30; done
run() { id=$1; shift; grep -q " END $id " 17-method/queue.log && { echo "skip $id (END exists)"; return; }; 17-method/run.sh $id -- "$@"; }
run decay-mlr0.007b ${DECAY/mlr=0.01/mlr=0.007} seed=11
run decay-mlr0.014b ${DECAY/mlr=0.01/mlr=0.014} seed=11
echo Q5R2 DONE >> 17-method/queue.log
