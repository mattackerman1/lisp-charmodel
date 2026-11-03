#!/bin/bash
# usage: logrow.sh RUNID CODEFILE KEEP(base|keep|discard|pending) [note]  -- append one results.tsv row from 19-speed/runs/RUNID.log
cd /home/box/lisp-charmodel/efficiency; T=17-method/results.tsv; L=19-speed/runs/$1.log; code=$2; keep=$3; note=$4
grep -q "^$1	" $T && { echo "$1 already in ledger"; exit 0; }
tr=$(grep '^trained' $L); steps=$(echo "$tr" | awk '{print $2}'); trs=$(echo "$tr" | awk '{print $5}' | tr -d s)
wall=$(echo "$tr" | grep -o 'wall [0-9.]*' | cut -d' ' -f2); ev=$(echo "$tr" | grep -o 'eval [0-9.]*' | cut -d' ' -f2)
tok=$(echo "$tr" | grep -o '[0-9.]* tokens/s' | cut -d' ' -f1); tok=${tok%.}
seg=$(grep -E '^step +(200|400) ' $L | sed 's/.*t= *\([0-9.]*\)s.*/\1/' | tr '\n' ' ')
segtok=$(awk -v s="$seg" 'BEGIN{split(s,a," "); printf "%.0f", 200*4096/(a[2]-a[1])}')
v=$(grep -E '^step +400 ' $L | grep -o 'val(subset)=[0-9.]*' | cut -d= -f2)
sha=$(sha256sum $code | cut -c1-16); la=$(grep "START $1 " 19-speed/runs/bench.log | sed 's/.*load=\([0-9.]*\).*/\1/')
cmd="sbcl --dynamic-space-size 4000 --script $code --run tokens=08-vocab-up/tokens16384.u16 vocab=08-vocab-up/vocab16384.hex vstart=23505859 fullval-targets=238949 d=256 layers=6 heads=8 ctx=64 batch=64 threads=8 dropout=0 lr=1e-3 optim=muon mlr=0.01 ew=f32 sched=wsd warm=0.01 decay=0.2 lrmin=0 steps=400 seed=1 ssm=1024 ssm-alpha=0.75 gemm=lisp512 final=0 savefinal=1 [19-speed/bench.sh; tok_s=clean seg steps 200-400 (chars/s=$(awk -v t=$segtok 'BEGIN{printf "%.0f",t*3.82458}')); probe_tok_s=whole-run logged tok/s; val1m col=val(subset) at step 400; $note]"
printf "%s\t%s\t%s\t1\t%s\t%s\t%s\tok\t%s\t%s\t%s\t%s\t%s\tna\t%s\t%s\n" "$1" "$(date '+%F %H:%M')" "$sha" "$steps" "$segtok" "$tok" "${la:-na}" "$trs" "$wall" "$ev" "$v" "$keep" "$cmd" >> $T
tail -1 $T | cut -f1-15
