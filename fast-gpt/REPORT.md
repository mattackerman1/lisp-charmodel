# fast-gpt: faster SBCL char transformer, same maths (bitwise-identical results)

`gpt-fast.lisp` is a drop-in replacement for `../big/gpt-big.lisp` (copied here as `gpt-old.lisp`). It takes the same CLI options and adds `steps=N`. It uses the same model, init, RNG streams, dropout masks, LR schedule and AdamW. Each step's gradients, and therefore the loss curve, are **bit-for-bit identical** to the old code at the same seed and thread count. It is single-float only; use gpt-big.lisp for the FT=double gradcheck.
Box: 8-core Xeon (Sapphire Rapids, AVX-512), SBCL 2.5.2, OpenBLAS (pthread build, set to 1 thread), all runs under `nice -n 10` with 8 threads. Nothing needed reinstalling. No science-team jobs were seen (queue.log).

## Results
| | gpt-big.lisp (old) | gpt-fast.lisp (new) | speedup |
|---|---|---|---|
| bench harness, 200 random-token steps (`prof.lisp`) | 32.3k tok/s (63.4 ms/step) | 93.8k tok/s (21.8 ms/step) | 2.9x |
| real script, 300 s on the corpus, same config back to back (`bench-*.log`) | 33,006 tok/s (16.1 steps/s) | 86,909 tok/s (42.4 steps/s) | 2.63x |
| 1 h run (3600 s training time) | 27,276 tok/s, 47,947 steps (Oct 17 morning run) | **93,195 tok/s**, 163,820 steps (`fast.log`) | 3.4x |
| full val @ 600/1200/1800/2400/3000 s | 1.4408 / 1.3680 / 1.3323 / 1.3045 / 1.2753 | 1.3750 / 1.3324 / 1.3114 / 1.2820 / 1.2538 | |
| **final full val (3600 s)** | 1.2645 (PyTorch: 1.2750 at 23.0k tok/s) | **1.2423** | -0.022 vs Lisp, -0.033 vs PyTorch |
| eval time inside the 1 h run (excluded from the budget) | 300 s | 305 s (more subset evals, each one faster) | |

The new code runs 3.4x as many steps in the same hour. Because the per-step maths is identical, the whole val-loss gain comes from the extra steps. Val still falls at every checkpoint. Tokens/s varies about ±10% with other load on the shared box (a node process uses about one core). The fairest old-vs-new number is the back-to-back 300 s benchmark (2.63x). The two 1 h runs ran at different times, so that 3.4x includes box-load differences: the old code measured 33k tok/s in this session against 27.3k in its own 1 h run. Chart: `speed.png` (left: tokens/s; right: val loss vs training time for old Lisp, PyTorch and new).

Dropout-off option: the existing `dropout=0` flag still works. It gives 105.5k tok/s in the harness, but it is a different model, so none of the numbers above use it.

## Profile before -> after (8 threads; harness `prof.lisp`; worker split from sb-sprof on one worker)
Step phases:
| phase | before | after |
|---|---|---|
| forward+backward (8 workers in parallel) | 47.2 ms (74%) | 20.7 ms (95%) |
| gradient reduction across workers | 4.3 ms (7%), serial | \ 1.1 ms (5%), parallel, fused |
| AdamW + clip | 11.8 ms (19%), serial | / |
| thread spawn/sync | make-thread x7 per step | persistent pool, 2 barriers/step |
| consing / GC | 1.5 MB/step, GC ~0 | 175 KB/step, GC 0 |

Inside one worker's forward+backward (share of samples):
| function | before | after |
|---|---|---|
| GELU backward | 29.6% | 1.9% |
| GELU forward | 24.5% | 15.0% |
| sgemm (kernel + pack copies + small kernels) | ~24.5% | ~50% (now the main cost, near peak) |
| attention softmax/mask (scalar parts) | 7.7% | 8.1% fwd + 2.0% bwd |
| dropout mask generation (xorshift) | 4.1% | 6.2% (at its serial dependency-chain limit) |
| layernorm + LN backward | 4.0% | 10.7% (unchanged code; larger share) |
| libm floor() call inside exp | 1.0% | 0 |
The attention matmuls already went through sgemm per (sequence, head), so there was no scalar matmul left to batch. Per-call kernel times (`kbench.lisp`, one worker, R=256): GELU 2.27 -> 0.55 ms (4.1x), GELU-back 3.00 -> 0.24 ms (12.5x), attention forward 0.59 -> 0.36 ms (1.6x), attention-back 0.19 -> 0.15 ms, fill-mask 0.168 -> 0.116 ms (1.45x).

## Changes and their speedups (cumulative ablation, `ablate.lisp`, `ablate.log`, 8 threads)
`ablate.lisp` loads gpt-fast.lisp and puts back the gpt-big.lisp code for every change that is not enabled. With nothing enabled it measures 33.5k, matching the old code.
| change (in order of payoff) | ms/step | tok/s | step speedup |
|---|---|---|---|
| baseline (all old code) | 61.1 | 33.5k | |
| + persistent thread pool instead of make-thread/join per step | 58.2 | 35.2k | 1.05x |
| + gradient reduction and AdamW in parallel on the workers (whole tensors per thread, LPT-balanced, 8-wide SIMD, unboxed sqrt) | 42.4 | 48.3k | 1.37x |
| + GELU: tanh cached (double) from forward for backward, fwd/bwd in 4x-double SIMD (sb-simd AVX2) with the exact op order and no FMA; exp's floor done in-register | 25.3 | 80.9k | 1.67x |
| + attention: SIMD exp/normalize in softmax (sum kept sequential in double), SIMD dropout multiply and softmax backward | 21.7 | 94.6k | 1.17x |
| + element-wise: SIMD residual adds, dropout multiplies, bias grads, logits scaling; branch-free mask select (same xorshift stream); LN 1/std unboxed | 20.2 | 101.5k | 1.07x |
| **total** | | | **3.0x** in this ablation run (2.6–3.4x across the measurements above) |
Tried and dropped:
- Vectorizing LayerNorm's normalize step in 4x-double made it 14x slower: SBCL emitted legacy-SSE moves inside the AVX loop. It stays scalar.
- Unrolling GELU 2x gave no gain.
- A VZEROUPPER inside the generic 8-wide loop macro was a real bug: it clobbered a hoisted broadcast register. The equivalence test caught it and it was removed. VZEROUPPER is now issued only where no YMM value is live.
Not done:
- Finer-grained or dynamic work splitting across threads. It would change the per-worker dropout RNG streams and gradient summation order, so results would no longer be bit-identical.
- A separate fused softmax+mask+dropout kernel. The softmax is already SIMD, and the mask multiply is one SIMD pass.

## Correctness
- **Gradient check, old vs new analytic gradients** (`equiv.lisp`, `equiv.log`): tiny config (V=5 d=8 L=2 H=2 T=4, 2 workers x 2 seqs) at dropout 0 / 0.2 / 0.5, and the real config (V=86 d=128 L=4 H=4 T=64, 8 workers x 4 seqs) at dropout 0 / 0.1. **All 1,832 and 812,544 gradients are bitwise equal (0 differ)**, and so are the losses.
- **Numerical gradcheck** (`gradcheck.log`): new FT=single, eps 1e-2, gives the same numbers as old FT=single (global rel err 9.1e-5 / 3.1e-4 / 9.1e-4). Old FT=double (eps 1e-5, rel err ≤ 9e-10) validates the maths the new code reproduces bit for bit.
- **Loss curves** (`curves-old-vs-new.csv`): 500 training steps on the corpus with seed 1, same windows, dropout 0.1, 8 threads and a step-based LR. **0 of 500 steps differ** (e.g. step 500: 2.24707648 in both). Wall time was 34.9 s old vs 11.3 s new.

## Files (all in /home/box/lisp-charmodel/fast-gpt/)
- Code: `gpt-fast.lisp` (new) and `gpt-old.lisp` (copy of gpt-big.lisp).
- Results: `REPORT.md`, `speed.png` (from `plot_speed.py`).
- 1 h run: `fast.log`, `fast.csv`, `fast-full.csv`.
- 300 s benchmark: `bench-old.*`, `bench-new.*`; `queue.sh`, `queue.log`.
- Correctness: `equiv.lisp`, `equiv.log`, `curves-old-vs-new.csv`, `gradcheck.log`.
- Profiling: `prof.lisp`, `prof-after.log`, `kbench.lisp`, `ablate.lisp`, `ablate.log`.
