#!/bin/bash
cd /home/box/lisp-charmodel/efficiency; . 17-method/common.sh
until grep -q "Q3 DONE" 17-method/queue.log; do sleep 30; done
# 5a/5b dynamics: branch decay test (seed 11, vs decay-base-s11/s12)
17-method/run.sh decay-sqrt -- $DECAY seed=11 dshape=sqrt
17-method/run.sh decay-elr3e-4 -- $DECAY seed=11 elr=3e-4
17-method/run.sh decay-elr3e-3 -- $DECAY seed=11 elr=3e-3
17-method/run.sh decay-mlr0.007 -- $DECAY seed=11 mlr=0.007
17-method/run.sh decay-mlr0.014 -- $DECAY seed=11 mlr=0.014
# 5c-f architecture: 15-min proxy from scratch (seed 1, vs proxy-base-s1..s3)
17-method/run.sh proxy-relu2 -- $PROXY seed=1 relu2=1
17-method/run.sh proxy-qknorm -- $PROXY seed=1 qknorm=1 qkg=2
17-method/run.sh proxy-zinit -- $PROXY seed=1 zinit=1
17-method/run.sh proxy-softcap -- $PROXY seed=1 softcap=15
echo Q5 DONE >> 17-method/queue.log
