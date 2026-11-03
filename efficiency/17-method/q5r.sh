#!/bin/bash
# q5 restart (2026-10-23 after box restart): remaining proxy tricks; skip any id with an END line.
cd /home/box/lisp-charmodel/efficiency; . 17-method/common.sh
hardening/preflight.sh > 17-method/preflight-q5r.txt 2>&1 || exit 2
run() { id=$1; shift; grep -q " END $id " 17-method/queue.log && { echo "skip $id (END exists)"; return; }; 17-method/run.sh $id -- "$@"; }
run proxy-relu2 $PROXY seed=1 relu2=1
run proxy-qknorm $PROXY seed=1 qknorm=1 qkg=2
run proxy-zinit $PROXY seed=1 zinit=1
run proxy-softcap $PROXY seed=1 softcap=15
echo Q5R DONE >> 17-method/queue.log
