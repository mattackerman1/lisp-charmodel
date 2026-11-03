#!/bin/bash
# usage: ab.sh "id1:c1|c2" ...  runs sequentially (c2 = candidate lgemm via GPT_LGEMM=$LG)
cd /home/box/lisp-charmodel/efficiency
for x in "$@"; do id=${x%%:*}; v=${x##*:}
  if [ "$v" = "cand" ]; then GPT_LGEMM=$LG 19-speed/bench.sh $id ${CODE:-gpt-eff.lisp} >/dev/null 2>&1; elif [ "$v" = "new" ]; then env -u GPT_LGEMM 19-speed/bench.sh $id ${CODE:-gpt-eff.lisp} >/dev/null 2>&1; else env -u GPT_LGEMM 19-speed/bench.sh $id ${BASECODE:-gpt-eff.lisp} >/dev/null 2>&1; fi
  grep "^  $id " 19-speed/runs/bench.log | tail -1
done; echo AB_DONE
