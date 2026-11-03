#!/bin/bash
# #17 ledger wrapper. Usage: run.sh RUNID [CODE=gpt-eff.lisp] -- <gpt-eff args without tag=>
# Waits until no other gpt-eff training runs, does a 80-step speed probe of the same model args, runs nice -n 10,
# appends one line to results.tsv. keep/discard starts as "pending" (set later with mark.sh).
cd /home/box/lisp-charmodel/efficiency; source hardening/env.sh
id=$1; shift; code=gpt-eff.lisp; [ "$1" != "--" ] && { code=$1; shift; }; shift
L=17-method/runs; T=17-method/results.tsv; mkdir -p $L
[ -f $T ] || printf "run_id\tdate_ct\tcode_sha256\tseed\tsteps\ttok_s\tprobe_tok_s\tslow_flag\tload_avg\ttrain_s\twall_s\teval_s\tval1m\tfull_val\tkeep\tcmd\n" > $T
grep -q "^$id	" $T && { echo "run $id already in ledger, skip"; exit 0; }
while pgrep -f "lisp --run" >/dev/null; do sleep 20; done
cp $code $L/$id.lisp; code=$L/$id.lisp   # frozen copy: a live run never reads an edited file
sha=$(sha256sum $code | cut -c1-16)
# speed probe: same model/optim args, 80 steps, no init/evals
pargs=$(printf '%s\n' "$@" | grep -Ev '^(init|steps|secs|tag|fullval-every|periodic-full|ckpt|savefinal|val1m-steps|dsteps|ackpt|resume)=' | tr '\n' ' ')
nice -n 10 $SBCL --dynamic-space-size 4000 --script $code --run $pargs steps=80 final=0 tag=$L/$id-probe > $L/$id-probe.log 2>&1
probe=$(grep -o '[0-9.]* tokens/s' $L/$id-probe.log | cut -d' ' -f1); probe=${probe%.}
( while true; do cut -d' ' -f1 /proc/loadavg; sleep 30; done ) > $L/$id.load & lp=$!
cmd="$SBCL --dynamic-space-size 4000 --script $code --run $* tag=$L/$id"
echo "$(date '+%F %T %Z') START $id: $cmd" >> 17-method/queue.log
nice -n 10 $cmd > $L/$id.log 2>&1; rc=$?
kill $lp
lg=$L/$id.log; tr=$(grep '^trained' $lg)
steps=$(echo "$tr" | awk '{print $2}'); tok=$(echo "$tr" | grep -o '[0-9.]* tokens/s' | cut -d' ' -f1); tok=${tok%.}
trs=$(echo "$tr" | awk '{print $5}' | tr -d s); ev=$(echo "$tr" | grep -o 'eval [0-9.]*' | cut -d' ' -f2)
ev=$(grep EVALTIME $lg | sed 's/ *(.*//' | grep -o '[0-9.]*s$' | tr -d s | awk -v e="$ev" '{e+=$1} END{printf "%.1f", e}')
wall=$(grep 'EVALTIME final full' $lg | grep -o 'so far [0-9.]*' | cut -d' ' -f3)
v1=$(grep 'end of training' $lg | grep -o ')=[0-9.]*' | tail -1 | cut -c3-); fv=$(grep '^FINAL' $lg | grep -o '=[0-9.]* nats' | cut -c2- | cut -d' ' -f1)
la=$(awk '{s+=$1;n++} END{if(n) printf "%.2f", s/n}' $L/$id.load)
seed=$(printf '%s\n' "$@" | grep '^seed=' | cut -d= -f2)
flag=ok; [ -n "$tok" ] && [ -n "$probe" ] && awk "BEGIN{exit !($tok < 0.9*$probe)}" && flag=SLOW
[ $rc -ne 0 ] && flag="rc=$rc"
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$id" "$(date '+%F %H:%M')" "$sha" "${seed:-1}" "$steps" "$tok" "$probe" "$flag" "$la" "$trs" "$wall" "$ev" "$v1" "$fv" pending "$cmd" >> $T
echo "$(date '+%F %T %Z') END $id rc=$rc steps=$steps tok/s=$tok probe=$probe $flag full=$fv" >> 17-method/queue.log
