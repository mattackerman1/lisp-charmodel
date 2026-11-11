#!/bin/bash
# Correctness harness for the isolated horizon-2 lisp. Synthetic corpus only.
# Does not read the box corpus and does not launch the 3925-step ABBA.
# On the box, uses /home/box/opt/sbcl and gemm=lisp512. Elsewhere, gemm=lisp via GPT_LGEMM.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
EFF=$(cd "$HERE/../.." && pwd)
ROOT=$(cd "$EFF/.." && pwd)
BASELISP=$EFF/gpt-eff.lisp
NEWLISP=$HERE/gpt-eff.lisp
PROOF=${PROOF:-$HERE/proof}
mkdir -p "$PROOF/syn" "$PROOF/runs"
if [ -x /home/box/opt/sbcl/bin/sbcl ] && [ -f /home/box/lisp-charmodel/all-lisp/lgemm512.lisp ]; then
  SBCL=${SBCL:-/home/box/opt/sbcl/bin/sbcl}
  GEMM=lisp512
elif [ -x /tmp/sbcl-2.4.10-x86-64-linux/run-sbcl.sh ]; then
  SBCL=${SBCL:-/tmp/sbcl-2.4.10-x86-64-linux/run-sbcl.sh}
  GEMM=lisp
  export GPT_LGEMM=${GPT_LGEMM:-$ROOT/all-lisp/lgemm.lisp}
else
  SBCL=${SBCL:-sbcl}
  GEMM=lisp
  export GPT_LGEMM=${GPT_LGEMM:-$ROOT/all-lisp/lgemm.lisp}
fi
echo "SBCL=$SBCL GEMM=$GEMM GPT_LGEMM=${GPT_LGEMM:-unset}" | tee "$PROOF/env.txt"
"$SBCL" --version | tee -a "$PROOF/env.txt"
python3 - << PY
import random
rng = random.Random(0)
alphabet = [chr(c) for c in range(48, 80)]
open("$PROOF/syn/corpus.txt","w").write("".join(rng.choice(alphabet) for _ in range(20000)))
PY
COMMON="data=$PROOF/syn/corpus.txt d=32 layers=2 heads=4 ctx=16 batch=8 dropout=0 lr=1e-3 optim=muon mlr=0.01 ew=f32 sched=wsd warm=0.01 decay=0.2 lrmin=0 ssm=16 ssm-alpha=0.75 gemm=$GEMM savefinal=1 final=1"
run() {
  local name=$1 lisp=$2; shift 2
  echo "RUN $name"
  # Extra args precede COMMON so opt() sees them first (first match wins).
  "$SBCL" --dynamic-space-size 2000 --script "$lisp" --run "$@" $COMMON tag=$PROOF/runs/$name > "$PROOF/$name.log" 2>&1
}
cmpw() {
  local a=$1 b=$2
  cmp "$a" "$b" && echo "CMP_OK $a $b" | tee -a "$PROOF/cmp.txt"
}
: > "$PROOF/cmp.txt"
run base200 "$BASELISP" steps=200 seed=1 threads=1 ackpt-steps=200 ackpt-keep=1
run off200 "$NEWLISP" steps=200 seed=1 threads=1 ackpt-steps=200 ackpt-keep=1
run a0200 "$NEWLISP" steps=200 seed=1 threads=1 ackpt-steps=200 ackpt-keep=1 n2=1 n2-alpha=0
cmpw "$PROOF/runs/base200-final-000200.f32" "$PROOF/runs/off200-final-000200.f32"
cmpw "$PROOF/runs/base200-final-000200.f32" "$PROOF/runs/a0200-final-000200.f32"
cmpw "$PROOF/runs/base200-state-000000200.bin" "$PROOF/runs/off200-state-000000200.bin"
cmpw "$PROOF/runs/base200-state-000000200.bin" "$PROOF/runs/a0200-state-000000200.bin"
python3 - << PY
import pathlib, re, sys
root = pathlib.Path("$PROOF")
def fields(name):
    text = (root/f"{name}.log").read_text(errors="replace")
    steps = re.findall(r"train\(last batch\)=([0-9.]+)  val\(subset\)=([0-9.]+)", text)
    fin = re.findall(r"FINAL: val\(full fixed val, (\d+) targets scored, (\d+) chars\)=([0-9.]+)", text)
    if not steps or not fin:
        sys.exit(f"missing metrics in {name}")
    return steps, fin
b, bo, ba = fields("base200"), fields("off200"), fields("a0200")
if b != bo or b != ba:
    print("METRIC_MISMATCH", b, bo, ba)
    sys.exit(1)
print("METRICS_OK", b[1])
PY
# pool path, still alpha 0
run baseT "$BASELISP" steps=40 seed=1 threads=2 batch=8 ackpt-steps=0
run offT "$NEWLISP" steps=40 seed=1 threads=2 batch=8 n2=1 n2-alpha=0 ackpt-steps=0
cmpw "$PROOF/runs/baseT-final-000040.f32" "$PROOF/runs/offT-final-000040.f32"
# resume of an enabled run: continuous 40 vs resume from its own step-20 state
run cont "$NEWLISP" steps=40 seed=3 threads=1 n2=1 n2-alpha=0.1 ackpt-steps=20 ackpt-keep=2
run resume "$NEWLISP" steps=40 seed=3 threads=1 n2=1 n2-alpha=0.1 ackpt-steps=0 resume=$PROOF/runs/cont-state-000000020.meta
cmpw "$PROOF/runs/cont-final-000040.f32" "$PROOF/runs/resume-final-000040.f32"
grep -E 'N2 ACCOUNT|N2 account' "$PROOF/cont.log" | tee "$PROOF/n2-account.txt"
gcheck() {
  local name=$1 lisp=$2; shift 2
  echo "GRAD $name"
  "$SBCL" --dynamic-space-size 2000 --script "$lisp" --run mode=gradcheck gemm=$GEMM gcv=40 "$@" > "$PROOF/$name.log" 2>&1
  grep -E '^(gradcheck|  max rel)' "$PROOF/$name.log" > "$PROOF/$name.lines" || true
}
gcheck g-base "$BASELISP" ssm=3
gcheck g-off "$NEWLISP" ssm=3
gcheck g-a0 "$NEWLISP" ssm=3 n2=1 n2-alpha=0
diff -u "$PROOF/g-base.lines" "$PROOF/g-off.lines" > "$PROOF/diff-g-off.txt"
diff -u "$PROOF/g-base.lines" "$PROOF/g-a0.lines" > "$PROOF/diff-g-a0.txt"
gcheck g-a01-s3 "$NEWLISP" ssm=3 n2=1 n2-alpha=0.1
gcheck g-a01-s12 "$NEWLISP" ssm=12 n2=1 n2-alpha=0.1
gcheck g-a1-s3 "$NEWLISP" ssm=3 n2=1 n2-alpha=1
gcheck g-a1-s12 "$NEWLISP" ssm=12 n2=1 n2-alpha=1
# Stock eps=1e-2 global error is reported above and is not the pass rule.
# n2eps requires the residual to shrink at least 4x from eps=1e-2 and land under 5e-4
# (3x the measured primary-only floor), plus a directional check on wte/wpe/trunk/U.
mkdir -p "$PROOF/grad-eps"
eps_one() {
  local name=$1; shift
  echo "EPS $name"
  "$SBCL" --dynamic-space-size 2000 --script "$NEWLISP" --run mode=n2eps gemm=$GEMM gcv=40 ssm=3 "$@" > "$PROOF/grad-eps/$name.log" 2>&1
  grep -E 'N2EPS_SUMMARY|N2EPS_PASS|N2EPS_FAIL' "$PROOF/grad-eps/$name.log"
  grep -q N2EPS_PASS "$PROOF/grad-eps/$name.log"
}
eps_one a0
eps_one a01 n2=1 n2-alpha=0.1
eps_one a1 n2=1 n2-alpha=1
"$SBCL" --dynamic-space-size 2000 --script "$NEWLISP" --run mode=n2check n2=1 n2-alpha=0.1 ssm=8 gemm=$GEMM seed=1 > "$PROOF/n2check.log" 2>&1
grep N2CHECK "$PROOF/n2check.log" | tee "$PROOF/n2check.txt"
grep -q "N2CHECK PASS" "$PROOF/n2check.log"
echo "VERIFY_PASS" | tee "$PROOF/VERIFY_PASS"
