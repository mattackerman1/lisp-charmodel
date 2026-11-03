#!/bin/bash
cd /home/box/lisp-charmodel/efficiency; . 17-method/common.sh
hardening/preflight.sh --full > 17-method/preflight-q3.txt 2>&1 || exit 2
for s in 1 2 3; do 17-method/run.sh proxy-base-s$s -- $PROXY seed=$s; done
for s in 11 12; do 17-method/run.sh decay-base-s$s -- $DECAY seed=$s; done
echo Q3 DONE >> 17-method/queue.log
