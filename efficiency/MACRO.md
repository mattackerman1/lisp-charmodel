# DEFOP/TAPE macro rewrite of gpt-eff.lisp (2026-10-21, Grok Bot)

New file: `gpt-eff-macro.lisp` (gpt-eff.lisp is unchanged). Same command line, same options, same outputs.
Proof files: `15-macro/`.

## What changed
- `(defop NAME (params) :grads (...) :decl :gdecl :wrap :fwd :bwd)` defines both directions of an op in one place. It produces
  `NAME` (forward) and `NAME-BACK` (backward; it receives every forward argument plus the grad buffers). Type declarations are written
  once. `:wrap` holds scaffolding that both directions share (the per-row loop of layernorm/embed, the per-sequence/per-head loop and
  offsets of attention); `%body` marks the slot, and `(%fwd x)` / `(%bwd x)` mark parts used in one direction only.
- A grad declared as `(gW W)` is derived from the call-site argument for `W`: `(P +wqkv+)` becomes `(GR +wqkv+)`, and
  `(m-wte m)` becomes `(m-wte g)`. The call site only names activation-gradient buffers (`:dY tmp :dX dx`).
- `(tape backprop bwd-bindings pgrad steps...)` writes the model **once** as a list of op calls. Forward = the calls in order. Backward
  = the same calls reversed, `(:loop (l nl) ...)` turned into a `downto` loop, `(:fwd ...)` steps (residual adds, mask fill) dropped.
  `forward-backward` went from 77 lines (hand-written forward and backward) to 26.
- Ops: embed (+ embedding dropout), layernorm, linear (+ optional output dropout mask), gelu (dispatches to the unchanged f64/f32 SIMD
  kernels), attention, head (tied logits + softmax + CE). The elementwise kernels (gelu64/32, softmax-row, vexp/vtanh) are unchanged.
- Small tooling change: `hardening/preflight.sh` accepts `GPT_FILE=gpt-eff-macro.lisp` (the default is still gpt-eff.lisp).

## Results
1. **Gradient check**: output is identical to gpt-eff.lisp, character for character (`15-macro/gradcheck-*.txt`). Global rel err is
   6.501e-5, 7.697e-5, 7.170e-4. `GPT_FILE=gpt-eff-macro.lisp hardening/preflight.sh --full` gives PREFLIGHT OK (`15-macro/preflight-macro.txt`).
2. **Loss equality (bit-identical)**:
   - `15-macro/stepdriver.lisp`, d128 L4 ctx64, 8 workers, 60 steps, seed 1, all printed at %.12f:
     - AdamW, no dropout: per-step loss, grad norm and final weight hash are identical (`steps-adamw-*.txt`).
     - Muon, dropout 0.1: same, identical (`steps-muon-*.txt`).
     - Muon steps 1..5 loss: 4.203187617152, 3.878009376079, 3.667434035655, 3.560017409868, 3.499034604165.
       Step 10: 3.168320911243. Step 30: 2.665318991618. Step 60: 2.494267902255, in both files.
   - Full `main`, 600 steps, sched=const warm=0:
     - cfg1 (Muon, dropout 0.1): step 200/400/600 train 2.1930/1.9359/1.8760, val 2.1333/1.9484/1.8587, VAL1M 1.8748.
     - cfg2 (AdamW, ew=f32, short-ctx 32 with posoff): train 2.3105/2.1117/1.9949, VAL1M 2.0426.
     - Both cfgs, both files: logs identical, **final weights bitwise identical** (cmp of savefinal .f32), and the 300-char samples are
       identical (`main*.log`, `equal.txt`).
3. **Speed** (nice 10, 8 threads, alternating old/new). The box is noisy (single blocks vary 1.6–3.2 s; two 60-s runs stalled at ~15 steps/s).
   - Timed blocks (`bench3.sh`, 6 alternating runs x 12 blocks of 100 AdamW steps, d128 L4 ctx64 b32):
     - old: min 1.612 s, p25 1.722 s, median 1.792 s.
     - new: min 1.596 s, p25 1.741 s, median 1.796 s.
     - **new/old time: min 0.990, p25 1.011, median 1.002**, i.e. the same within noise (`bench3-stats.txt`).
   - 8 alternating 60-s training runs each (`bench.sh`, `bench2.sh`):
     - median steps/s: old 48.7, new 51.7 (noise-dominated, includes one stall per file; `bench12-stats.txt`).
4. **Code size** (`15-macro/count.lisp`, `count2.lisp`: Lisp reader over all top-level forms; size = conses + non-nil atoms):

| | old | new | change |
|---|---|---|---|
| whole file, non-blank non-comment lines | 896 | 896 | 0 |
| whole file, tree size (conses) | 22749 (13570) | 22514 (13440) | -1.0% |
| model/ops section, lines (incl. unchanged kernels) | 263 | 263 | 0 |
| model/ops section, tree size | 7110 | 6875 | -3.3% |
| model/ops without shared kernels, tree size | 5123 | 4888 (incl. 556 macro machinery) | -4.6% |
| op definitions + forward-backward only (no machinery) | 5096 | 4304 | -15.5% |
| forward-backward alone, lines / tree | 77 / 2498 | 26 / 1019 | -66% / -59% |

   So the code gets only slightly shorter overall. The backward duplication was mostly the plumbing in forward-backward, which shrinks
   by about 60%. Kernel math differs between forward and backward, so it cannot be shared. The macro machinery (defop + tape,
   36 lines / 556 tree) uses up most of the savings. What you gain is a single point of truth: a new op or a layer change is written once,
   and the backward order and buffer routing are derived instead of hand-mirrored.

## Before/after: linear
Before (two functions, signatures and declarations written twice; calls hand-mirrored in forward-backward, with dropout spliced into the
backward call site):
```lisp
(defun linear (X R din W bias Y dout)
  (declare (fvec X W bias Y) (dim R din dout))
  (dotimes (r R) (replace Y bias :start1 (* r dout)))
  (gemm nil nil R dout din (f 1) X 0 din W 0 dout (f 1) Y 0 dout))
(defun linear-back (X R din W dY dout gW gb dX)
  (declare (fvec X W dY gW gb) (dim R din dout))
  (gemm t nil din dout R (f 1) X 0 din dY 0 dout (f 1) gW 0 dout)
  (let ((d8 (logand dout (lognot 7))))       ; bias grad: per column, rows summed in the original order
    (declare (dim d8))
    (dotimes (r R)
      (let ((o (* r dout))) (declare (idx o))
        (loop for j of-type idx from 0 below d8 by 8 do (setf (v:f32.8-aref gb j) (v:f32.8+ (v:f32.8-aref gb j) (v:f32.8-aref dY (+ o j)))))
        (sb-simd-avx:vzeroupper)
        (loop for j from d8 below dout do (incf (aref gb j) (aref dY (+ o j)))))))
  (when dX (gemm nil t R din dout (f 1) dY 0 dout W 0 dout (f 0) dX 0 din)))
;; in forward-backward, forward:  (linear (svref (w-y w) l) R d (svref ly +wproj+) (svref ly +bproj+) xmid d)
;;                                (when (w-drop w) (vmul xmid (svref (w-m1 w) l) (* R d)))
;; and, separately, backward:     (linear-back (svref (w-y w) l) R d (svref ly +wproj+) (if (w-drop w) (progn (vmul-into (w-dm w) dx (svref (w-m1 w) l) (* R d)) (w-dm w)) dx) d (svref gl +wproj+) (svref gl +bproj+) dy)
```
After (one definition; one line in the tape gives both calls):
```lisp
(defop linear (X R din W bias Y dout mask)   ; Y = X W + bias, then (optional) dropout MASK on the output
  :grads (dY (gW W) (gb bias) dX dm) :decl ((fvec X W bias Y) (dim R din dout)) :gdecl ((fvec gW gb))
  :fwd ((dotimes (r R) (replace Y bias :start1 (* r dout)))
        (gemm nil nil R dout din (f 1) X 0 din W 0 dout (f 1) Y 0 dout)
        (when mask (vmul Y mask (* R dout))))
  :bwd ((let ((dY (if mask (progn (vmul-into dm dY mask (* R dout)) dm) dY)))
          (declare (fvec dY))
          (gemm t nil din dout R (f 1) X 0 din dY 0 dout (f 1) gW 0 dout)
          (let ((d8 (logand dout (lognot 7))))       ; bias grad: per column, rows summed in the original order
            (declare (dim d8))
            (dotimes (r R)
              (let ((o (* r dout))) (declare (idx o))
                (loop for j of-type idx from 0 below d8 by 8 do (setf (v:f32.8-aref gb j) (v:f32.8+ (v:f32.8-aref gb j) (v:f32.8-aref dY (+ o j)))))
                (sb-simd-avx:vzeroupper)
                (loop for j from d8 below dout do (incf (aref gb j) (aref dY (+ o j)))))))
          (when dX (gemm nil t R din dout (f 1) dY 0 dout W 0 dout (f 0) dX 0 din)))))
;; in the tape (the only mention):
;;   (linear (L w-y) R d (P +wproj+) (P +bproj+) xmid d (and drop (L w-m1)) :dY dx :dX (w-dy w) :dm dm)
```
