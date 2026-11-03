#!/bin/bash
# append-row.sh RUNDIR [RESULTS_TSV]  -- append the #20 ledger row (same columns as 17-method/results.tsv). Idempotent: skips if run_id already present.
E=/home/box/lisp-charmodel/efficiency; . $E/hardening/env.sh; R=${1:?}; T=${2:-$E/17-method/results.tsv}
. $R/run.conf; id=$(echo $TAG | sed 's/^e20-/e20-/'); lg=$R/$TAG.log
grep -q "^$id	" $T 2>/dev/null && { echo "row for $id already in $T"; exit 0; }
fin=$(tail -1 $R/$TAG-final.csv); steps=$(echo $fin | cut -d, -f1); trs=$(echo $fin | cut -d, -f2); tok=$(echo $fin | cut -d, -f3 | tr -d .)
fv=$(echo $fin | cut -d, -f4 | sed 's/d0//' | awk '{printf "%.4f",$1}')
v1=$(tail -1 $R/$TAG-val1m.csv | cut -d, -f3 | sed 's/d0//' | awk '{printf "%.4f",$1}')
ev=$(grep EVALTIME $lg | grep -o '[0-9.]*s$' | tr -d s | awk '{e+=$1} END{printf "%.1f", e}')
st=$(stat -c %Y $R/run.conf); wall=$(( $(date +%s) - st ))
sha=$(sha256sum $E/gpt-eff.lisp | cut -c1-16)
cmd="$SBCL --dynamic-space-size $DYN --script gpt-eff.lisp --run $ARGS ackpt=$ACKPT ackpt-keep=$KEEP tag=$R/$TAG [20-long/queue20.sh via hardening/supervise.sh; #20 long run, #19 code, constant LR to step 97,608 then 2,996-step sqrt decay, total 100,604 = #17; wall_s=since run.conf creation incl. pauses; eval_s=sum EVALTIME]"
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$id" "$(date '+%F %H:%M')" "$sha" 1 "$steps" "$tok" na ok na "$trs" "$wall" "$ev" "$v1" "$fv" pending "$cmd" >> $T
echo "appended $id to $T"
