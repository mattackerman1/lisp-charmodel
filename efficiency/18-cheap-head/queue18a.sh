#!/bin/bash
# #18 stage A: default-path bitwise check (400 steps) + matched-compute 15-min-class proxies.
# Matched compute: candidate steps = 2500 * probe_tok_s(candidate)/probe_tok_s(v16 full) (probes1.tsv), so training seconds ~ equal.
cd /home/box/lisp-charmodel/efficiency; source 17-method/common.sh; source 18-cheap-head/common18.sh
bash 17-method/run.sh e18-default-400 -- $BASE $SCH steps=400 seed=1
bash 17-method/run.sh e18-proxy-base -- $V16 $SCH steps=2500 seed=1
bash 17-method/run.sh e18-v16-s1024 -- $V16 $SCH steps=4341 seed=1 ssm=1024
bash 17-method/run.sh e18-v32-s1024 -- $V32 $SCH steps=3752 seed=1 ssm=1024
echo QUEUE18A_DONE
