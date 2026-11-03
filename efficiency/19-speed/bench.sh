#!/bin/bash
# #19 speed bench. usage: bench.sh ID CODEFILE [extra args]   (run from anywhere; nothing else may run concurrently)
# 400-step run, ssm=1024 v16384 d256 L6 b64 ctx64 threads8 lisp512, seed 1, savefinal (bitwise compare), final=0, nice/ionice.
cd /home/box/lisp-charmodel/efficiency; source hardening/env.sh
id=$1; code=$2; shift 2
L=19-speed/runs; mkdir -p $L
while pgrep -f "lisp --run" >/dev/null; do echo "waiting: another gpt-eff run active"; sleep 20; done
ARGS="tokens=08-vocab-up/tokens16384.u16 vocab=08-vocab-up/vocab16384.hex vstart=23505859 fullval-targets=238949 d=256 layers=6 heads=8 ctx=64 batch=64 threads=8 dropout=0 lr=1e-3 optim=muon mlr=0.01 ew=f32 sched=wsd warm=0.01 decay=0.2 lrmin=0 steps=400 seed=1 ssm=1024 ssm-alpha=0.75 gemm=lisp512 final=0 savefinal=1"
echo "$(date '+%F %T') load=$(cut -d' ' -f1-3 /proc/loadavg) START $id $code" >> $L/bench.log
ionice -c3 nice -n 10 $SBCL --dynamic-space-size 4000 --script $code --run $ARGS "$@" tag=$L/$id > $L/$id.log 2>&1
tr=$(grep '^trained' $L/$id.log); tok=$(echo "$tr" | grep -o '[0-9.]* tokens/s' | cut -d' ' -f1); tok=${tok%.}
# clean rate: steps 200->400 (excludes startup/JIT of first 200)
seg=$(grep -E '^step +(200|400) ' $L/$id.log | sed 's/.*t= *\([0-9.]*\)s.*/\1/' | tr '\n' ' ')
echo "$id" "$tr" | tee -a $L/bench.log; echo "  t200,t400: $seg load_after=$(cut -d' ' -f1-3 /proc/loadavg)" | tee -a $L/bench.log
awk -v t="$tok" -v i="$id" -v s="$seg" 'BEGIN{split(s,a," "); printf "  %s tok/s=%s chars/s=%.0f  seg(200-400) tok/s=%.0f chars/s=%.0f\n", i, t, t*3.82458, 200*4096/(a[2]-a[1]), 200*4096/(a[2]-a[1])*3.82458}' | tee -a $L/bench.log
