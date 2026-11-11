#!/bin/bash
# Real-corpus #19-shape check for the parent box. Not an ABBA and not a 15700-step run.
# Proves live baseline vs isolated flag-off vs isolated n2=1 n2-alpha=0 at
# v16384 d256 L6 ctx64 batch64 ssm1024 lisp512 seed 1, 200 steps:
# final weights, optimizer-state bins, logged train/val, and the fixed full val.
# Then label/mask/leakage at that shape, and an enabled (alpha 0.1) resume.
# Writes proof/corpus200/VERIFIED only when every check passes. q-abba.sh requires that marker.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
EFF=$(cd "$HERE/../.." && pwd)
OUT=$HERE/proof/corpus200
mkdir -p "$OUT"
BASE_SHA=8be0dbaa692053ad6f7a125ff5880d3462100cdffb47c8a3132e54261627b5cb
BASELISP=$EFF/gpt-eff.lisp
NEWLISP=$HERE/gpt-eff.lisp
got=$(sha256sum "$BASELISP" | awk '{print $1}')
[ "$got" = "$BASE_SHA" ] || { echo "REFUSING: live baseline is $got"; exit 5; }
ISO=$(sha256sum "$NEWLISP" | awk '{print $1}')
SBCL=${SBCL:-/home/box/opt/sbcl/bin/sbcl}
[ -x "$SBCL" ] || { echo "box SBCL missing at $SBCL"; exit 2; }
TOK=$EFF/08-vocab-up/tokens16384.u16
VOC=$EFF/08-vocab-up/vocab16384.hex
[ -f "$TOK" ] && [ -f "$VOC" ] || { echo "missing $TOK or $VOC"; exit 2; }
echo "SBCL=$($SBCL --version) isolated=$ISO" | tee "$OUT/env.txt"
SHAPE="tokens=$TOK vocab=$VOC vstart=23505859 fullval-targets=238949 d=256 layers=6 heads=8 ctx=64 batch=64 threads=8 dropout=0 lr=1e-3 optim=muon mlr=0.01 ew=f32 sched=wsd warm=0.01 decay=0.2 lrmin=0 ssm=1024 ssm-alpha=0.75 gemm=lisp512 seed=1"
run() {
  local name=$1 lisp=$2; shift 2
  echo "RUN $name"
  "$SBCL" --dynamic-space-size 4000 --script "$lisp" --run "$@" $SHAPE tag=$OUT/$name > "$OUT/$name.log" 2>&1
}
run base200 "$BASELISP" steps=200 savefinal=1 final=1 ackpt-steps=200 ackpt-keep=1
run off200 "$NEWLISP" steps=200 savefinal=1 final=1 ackpt-steps=200 ackpt-keep=1
run a0200 "$NEWLISP" steps=200 savefinal=1 final=1 ackpt-steps=200 ackpt-keep=1 n2=1 n2-alpha=0
cmp "$OUT/base200-final-000200.f32" "$OUT/off200-final-000200.f32"
cmp "$OUT/base200-final-000200.f32" "$OUT/a0200-final-000200.f32"
cmp "$OUT/base200-state-000000200.bin" "$OUT/off200-state-000000200.bin"
cmp "$OUT/base200-state-000000200.bin" "$OUT/a0200-state-000000200.bin"
echo "CMP_OK weights and state bins" | tee "$OUT/cmp.txt"
python3 - << PY
import pathlib, re, sys
root = pathlib.Path("$OUT")
def grab(name):
    text = (root/f"{name}.log").read_text(errors="replace")
    steps = re.findall(r"lr=([0-9.]+)  train\(last batch\)=([0-9.]+)  val\(subset\)=([0-9.]+)", text)
    fin = re.findall(r"FINAL: val\(full fixed val, (\d+) targets scored, (\d+) chars\)=([0-9.]+)", text)
    if not steps or not fin:
        sys.exit(f"missing loss lines in {name}")
    return steps, fin[0]
b, o, a = grab("base200"), grab("off200"), grab("a0200")
open(root/"losses.txt","w").write(f"base {b}\noff {o}\nalpha0 {a}\n")
if not (b == o == a):
    sys.exit(f"LOSS_MISMATCH {b} {o} {a}")
print("LOSSES_OK", b[1])
PY
echo "RUN leak"
"$SBCL" --dynamic-space-size 4000 --script "$NEWLISP" --run mode=n2shape \
  tokens=$TOK vstart=23505859 d=256 layers=6 heads=8 ctx=64 gcv=16384 trials=2000 \
  n2=1 n2-alpha=0.1 ssm=1024 gemm=lisp512 seed=1 > "$OUT/leak.log" 2>&1
grep -q "N2SHAPE PASS" "$OUT/leak.log"
grep N2SHAPE "$OUT/leak.log" | tee "$OUT/leak.txt"
echo "RUN resume"
run cont40 "$NEWLISP" steps=40 savefinal=1 final=0 ackpt-steps=20 ackpt-keep=2 n2=1 n2-alpha=0.1
run resume40 "$NEWLISP" steps=40 savefinal=1 final=0 ackpt-steps=0 n2=1 n2-alpha=0.1 \
  resume=$OUT/cont40-state-000000020.meta
cmp "$OUT/cont40-final-000040.f32" "$OUT/resume40-final-000040.f32"
echo "RESUME_OK" | tee "$OUT/resume.txt"
cat > "$OUT/VERIFIED" << EOF
VERIFIED_CORPUS200
baseline_sha256=$BASE_SHA
isolated_sha256=$ISO
config=v16384 d256 L6 ctx64 batch64 threads8 ssm1024 ssm-alpha=0.75 lisp512 seed1 steps200
weights=cmp_equal
state=cmp_equal
losses=equal
fullval=equal
leakage=pass
resume=cmp_equal
EOF
echo "CORPUS_VERIFY_PASS" | tee "$OUT/CORPUS_VERIFY_PASS"
echo "ABBA may now be started with q-abba.sh. Marker: $OUT/VERIFIED"
