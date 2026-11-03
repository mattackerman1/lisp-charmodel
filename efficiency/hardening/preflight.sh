#!/bin/bash
# Preflight for gpt-eff.lisp runs. Exit 0 = ready; nonzero = clear FAIL line. Usage: preflight.sh [--full] (--full adds gradcheck, ~3 s)
. /home/box/lisp-charmodel/efficiency/hardening/env.sh
fail() { echo "PREFLIGHT FAIL: $*" >&2; echo "  fix: see /home/box/lisp-charmodel/efficiency/HARDENING.md (section Toolchain / reinstall)" >&2; exit 1; }
[ -x "$SBCL" ] || fail "sbcl wrapper missing at $SBCL"
v=$("$SBCL" --version 2>&1) || fail "sbcl does not run: $v"; echo "ok  sbcl: $v ($SBCL)"
[ -f "$GPT_OPENBLAS" ] || fail "OpenBLAS missing at $GPT_OPENBLAS"
out=$("$SBCL" --noinform --non-interactive --no-userinit --eval '(require :sb-simd)' \
  --eval "(sb-alien:load-shared-object \"$GPT_OPENBLAS\")" \
  --eval '(let ((a (make-array 4 :element-type (quote single-float) :initial-contents (quote (1f0 2f0 3f0 4f0)))) (c (make-array 4 :element-type (quote single-float) :initial-element 0f0))) (sb-sys:with-pinned-objects (a c) (sb-alien:alien-funcall (sb-alien:extern-alien "cblas_sgemm" (function sb-alien:void sb-alien:int sb-alien:int sb-alien:int sb-alien:int sb-alien:int sb-alien:int sb-alien:single-float sb-sys:system-area-pointer sb-alien:int sb-sys:system-area-pointer sb-alien:int sb-alien:single-float sb-sys:system-area-pointer sb-alien:int)) 101 111 111 2 2 2 1f0 (sb-sys:vector-sap a) 2 (sb-sys:vector-sap a) 2 0f0 (sb-sys:vector-sap c) 2)) (format t "SGEMM ~a~%" c) (format t "CONFIG ~a~%" (sb-alien:alien-funcall (sb-alien:extern-alien "openblas_get_config" (function sb-alien:c-string)))))' 2>&1)
echo "$out" | grep -q "SGEMM #(7.0 10.0 15.0 22.0)" || fail "OpenBLAS did not load / sgemm wrong: $(echo "$out" | tail -3)"
echo "ok  OpenBLAS: $(echo "$out" | grep CONFIG | cut -c8-) ; sgemm 2x2 correct; sb-simd ok"
m=$("$PY" -c 'import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot, numpy; print(matplotlib.__version__)' 2>&1) || fail "matplotlib venv broken ($PY): $m"
echo "ok  matplotlib $m ($PY)"
[ -f /home/box/lisp-charmodel/input.txt ] && echo "ok  data dir present" || echo "warn ../input.txt missing (only needed for char runs)"
if [ "$1" = "--full" ]; then
  g=$(cd $EFF && nice -n 10 "$SBCL" --script ${GPT_FILE:-gpt-eff.lisp} --run mode=gradcheck 2>&1 | grep "global") || fail "gradcheck did not run"
  echo "$g" | awk '{x=$NF; gsub("d","e",x); e=x+0; if (e>2e-3) bad=1} END{exit bad}' || fail "gradcheck error too large: $g"
  echo "ok  gradcheck (3 cases) global rel err: $(echo "$g" | awk '{printf "%s ", $NF}')"
fi
echo "PREFLIGHT OK"
