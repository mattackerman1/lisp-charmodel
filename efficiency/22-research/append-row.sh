#!/bin/bash
# append-row.sh RUNDIR [RESULTS_TSV] -- append the #22 ledger row (17-method/results.tsv columns). Idempotent: skips if run_id present.
# eval_s = in-run evals (sum over step lines of d(wall)-d(t), per supervise segment; gpt-eff t = wall - eval secs) + final EVALTIME lines.
# wall_s = train_s + eval_s (excludes box pauses between segments and startup).
E=/home/box/lisp-charmodel/efficiency; . $E/hardening/env.sh; R=${1:?}; T=${2:-$E/17-method/results.tsv}
. $R/run.conf; id=$TAG; lg=$R/$TAG.log
grep -q "^$id	" $T 2>/dev/null && { echo "row for $id already in $T"; exit 0; }
fin=$(tail -1 $R/$TAG-final.csv); steps=$(echo $fin | cut -d, -f1); trs=$(echo $fin | cut -d, -f2); tok=$(echo $fin | cut -d, -f3 | tr -d .)
fv=$(echo $fin | cut -d, -f4 | sed 's/d0//' | awk '{printf "%.4f",$1}')
v1=$(tail -1 $R/$TAG-val1m.csv | cut -d, -f3 | sed 's/d0//' | awk '{printf "%.4f",$1}')
ev=$(awk '/^===== segment/{fresh=($0 ~ / fresh =====$/); have=0; next}
  /^step /{ for(i=1;i<=NF;i++){ if($i ~ /^t=/){ s=$i; sub("t=","",s); if(s=="") s=$(i+1) } if($i ~ /^wall=/){ w=$i; sub("wall=","",w); if(w=="") w=$(i+1) } }
    t=s+0; w=w+0; if(have) e+=(w-pw)-(t-pt); else if(fresh) e+=w-t; pw=w; pt=t; have=1 }
  /^EVALTIME final/{ for(i=1;i<=NF;i++) if($i ~ /^[0-9.]+s$/){ x=$i; sub("s$","",x); e+=x; break } }
  END{printf "%.1f", e}' $lg)
wall=$(awk -v a="$trs" -v b="$ev" 'BEGIN{printf "%.1f", a+b}')
sha=$(sha256sum $E/gpt-eff.lisp | cut -c1-16)
cmd="$SBCL --dynamic-space-size $DYN --script gpt-eff.lisp --run $ARGS ackpt=$ACKPT ackpt-keep=$KEEP tag=$R/$TAG [22-research/q22.sh via hardening/supervise.sh; $(cat $R/note 2>/dev/null); clean $(python3 $E/22-research/clean.py $lg 2>/dev/null); wall_s=train_s+eval_s; eval_s=in-run evals + final EVALTIME]"
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$id" "$(date '+%F %H:%M')" "$sha" "$(echo " $ARGS" | grep -o " seed=[0-9]*" | tail -1 | cut -d= -f2)" "$steps" "$tok" na ok na "$trs" "$wall" "$ev" "$v1" "$fv" "$(cat $R/keep 2>/dev/null || echo pending)" "$cmd" >> $T
echo "appended $id to $T"
