#!/bin/bash
# usage: gradcheck.sh CODEFILE OUTFILE   -- same configs as 18-cheap-head/gradcheck.txt (full softmax gcv=40, ssm=3, ssm=12), nice/ionice
cd /home/box/lisp-charmodel/efficiency; source hardening/env.sh
code=$1; out=$2; : > $out
for cfg in "gcv=40" "ssm=3 gcv=40" "ssm=12 gcv=40"; do
  echo "== $cfg" >> $out
  ionice -c3 nice -n 10 $SBCL --dynamic-space-size 4000 --script $code --run mode=gradcheck $cfg 2>&1 | grep -E "gradcheck|max rel" >> $out
done
