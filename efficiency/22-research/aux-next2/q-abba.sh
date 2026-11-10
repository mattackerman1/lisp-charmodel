#!/bin/bash
# Two-seed ABBA for the horizon-2 head. Does not call q22.sh (that queue waits on the KD teacher dump).
# Order: e22c-base-s1, e22c-n2-s1, e22c-n2-s2, e22c-base-s2. 3925 steps. No 15700. No qknorm.
# Keep only if mean full-val gain (base minus n2) > 0.0036. Equal steps, not equal FLOPs.
# Working directory for the trainer is EFF so tokens= and gemm paths resolve. Live gpt-eff.lisp is not the script.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
EFF=$(cd "$HERE/../.." && pwd)
SUP=$HERE/supervise.sh
Q=$HERE/runs
mkdir -p "$Q"
exec 8>"$HERE/.q-abba.lock"
flock -n 8 || { echo "q-abba already running"; exit 0; }
BASE_SHA=8be0dbaa692053ad6f7a125ff5880d3462100cdffb47c8a3132e54261627b5cb
got=$(sha256sum "$EFF/gpt-eff.lisp" | awk '{print $1}')
[ "$got" = "$BASE_SHA" ] || { echo "REFUSING: $EFF/gpt-eff.lisp is $got, want $BASE_SHA"; exit 5; }
echo "isolated $(sha256sum "$HERE/gpt-eff.lisp")"
BASE="tokens=08-vocab-up/tokens16384.u16 vocab=08-vocab-up/vocab16384.hex vstart=23505859 fullval-targets=238949 d=256 layers=6 heads=8 ctx=64 batch=64 threads=8 dropout=0 lr=1e-3 optim=muon mlr=0.01 ew=f32 sched=wsd warm=0.01 decay=0.2 lrmin=0 ssm=1024 ssm-alpha=0.75 gemm=lisp512 fullval-every=900 savefinal=1"
# tag|extra args   (ABBA: base s1, aux s1, aux s2, base s2)
LIST="
e22c-base-s1|steps=3925 seed=1
e22c-n2-s1|steps=3925 seed=1 n2=1 n2-alpha=0.1
e22c-n2-s2|steps=3925 seed=2 n2=1 n2-alpha=0.1
e22c-base-s2|steps=3925 seed=2
"
printf '%s\n' "$LIST" | grep -v '^$' | while IFS='|' read -r TAG EXTRA; do
  [ -z "$TAG" ] && continue
  R=$Q/$TAG
  mkdir -p "$R"
  [ -f "$R/run.conf" ] || printf 'TAG=%s\nARGS="%s %s"\nDYN=4000; ACKPT=540; KEEP=2; NICE=10\n' "$TAG" "$BASE" "$EXTRA" > "$R/run.conf"
  if [ ! -f "$R/DONE" ]; then
    echo "$(date '+%F %T %Z') START $TAG"
    "$SUP" "$R"
    echo "$(date '+%F %T %Z') END $TAG rc=$? $(grep -h FINAL "$R"/*.log 2>/dev/null | tail -1)"
  fi
done
echo "Q_ABBA_DONE"
echo "Decision: mean full fixed val, nats/char. Keep iff (mean base - mean n2) > 0.0036."
echo "Report N2 ACCOUNT extra-gemm-flop/tok beside the val. Do not retune steps to match FLOPs. No 15700 run in this queue."
