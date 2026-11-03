# all-lisp audit: non-Lisp pieces of the gpt-eff.lisp pipeline (2026-10-25 CT)

Pipeline: Gutenberg download -> corpus.txt -> BPE (16384) -> tokens16384.u16 -> gpt-eff.lisp training/eval -> charts/reports.
Model code, optimizers (Muon/AdamW), eval (val1m / full val), checkpoints (save/resume, `.f32` files, `init=`/`resume=`),
sampling and gradcheck were already Lisp (efficiency/gpt-eff.lisp). Everything else is listed here.

| # | Piece | File(s) | Language | Status |
|---|-------|---------|----------|--------|
| 1 | GEMM (all matmuls: linear fwd/bwd, attention, tied head, Muon Newton-Schulz) | `/home/box/opt/openblas/lib/libopenblas.so.0` via FFI in `efficiency/gpt-eff.lisp` (`cblas_sgemm`, `openblas_set_num_threads`) | C/asm (OpenBLAS) | **PORTED**: `all-lisp/lgemm.lisp` (sb-simd AVX2 FMA), `gemm=lisp`. OpenBLAS stays the default (`gemm=openblas`). |
| 2 | Low-precision GEMM experiment (#5, prec=bf16/mklf32) | `efficiency/05-precision/gemmlp.c`, `libgemmlp.so`, MKL `venv/lib/libmkl_rt.so.3`; benches `gemmbench.c`, `racetest*.c` | C + Intel MKL | Not ported (dead experiment, off by default; `gemm=lisp` skips loading it). |
| 3 | Corpus build (clean Gutenberg -> ASCII corpus.txt) | `huge/build_corpus.py` | Python | **PORTED**: `all-lisp/corpus.lisp` (byte-identical corpus.txt) |
| 4 | Corpus download | `huge/fetch.sh` (curl, 1 req/s) | bash + curl | Kept as shell (network fetch, already done; raw/ present). |
| 5 | BPE training + encoding (HF `tokenizers` Rust lib + numpy) | `efficiency/08-vocab-up/train_bpe.py` (also 04-bpe/, 07-vocab/ copies) | Python + Rust | **PORTED**: `all-lisp/bpe.lisp` (trainer + encoder; byte-identical vocab/merges/tokens) |
| 6 | Launch / queue / ledger wrappers | `efficiency/17-method/run.sh`, `common.sh`, `mark.sh`, `q*.sh`, `efficiency/queue*.sh`, `extra*.sh`, `finalize.sh`, `hardening/{env,preflight,supervise,check-runs}.sh`, `15-macro/*.sh`, `05-precision/curves.sh`, `huge/{queue,finalize}.sh` | bash (+grep/awk/sed) | Kept as shell (allowed). Note: `preflight.sh` checks OpenBLAS + matplotlib; harmless for `gemm=lisp`. |
| 7 | Charts | `efficiency/17-method/charts.py`, `efficiency/make_report.py`, `10-long/report10.py`, `11-decay/report11.py`, `12-decay-from-earlier/report12.py`, `13-cyclic/report13.py`, `huge/plot_compare.py`, `huge/make_report.py` | Python + matplotlib (PNG) | Not ported: needs a plotting stack (PNG raster + font text); no Lisp plotting lib on the box. Not cheap. They only read the CSV/TSV logs the Lisp code writes. |
| 8 | Ranking check (Kendall tau) | `efficiency/17-method/rank_check.py` | Python | **PORTED**: `all-lisp/eval/rank-check.lisp` (identical output) |
| 9 | Muon numpy reference check | `efficiency/02-muon/muon_check.py` | Python + numpy | Not ported: one-off cross-check against an independent implementation (porting it to Lisp would defeat its purpose). |
| 10 | PyTorch comparison baselines | `pytorch/`, `huge/gpt_torch.py`, `big/`, `venv/` | Python + PyTorch | Out of scope (comparison targets, not part of the Lisp pipeline). |
| 11 | SBCL itself | `/home/box/opt/sbcl` | C runtime + Lisp | The runtime is C, as for any Lisp. |

Eval and checkpoint tools: already Lisp inside gpt-eff.lisp (val1m, full val, `ckpt=`, `ackpt=`, `resume=`, `init=`, `savefinal=`). No separate non-Lisp checkpoint tool exists.

## Port results (2026-10-25 CT)
- Corpus: `corpus.lisp` -> `tok/corpus.txt`, cmp-identical to `huge/corpus.txt` (121 books, 99,888,937 chars), 12 s.
- BPE: `bpe.lisp train 16384` -> `tok/vocab16384.hex`, `tok/merges16384.txt`, `tok/tokens16384.u16`: vocab, all 16,289 merges and tokens identical (sha256 0967cae9…fa9) to the Python/HF output, 7.4 s.
- GEMM: `lgemm.lisp`, `gpt-eff.lisp gemm=lisp` (backup of pre-change file: `efficiency/gpt-eff-pre-all-lisp.lisp`). Random test vs OpenBLAS max rel err 1.1e-6 (`gemm/test.out`). Gradcheck `gemm/gradcheck-{openblas,lisp}.txt`: 6.525e-5 / 7.690e-5 / 7.161e-4 (Lisp) vs 6.501e-5 / 7.697e-5 / 7.170e-4 (OpenBLAS, same as history). A process running with gemm=lisp maps no BLAS/MKL .so.
- 400-step WSD run (ledger `alllisp-openblas-400` vs `alllisp-lisp-400`): same losses to 4 decimals at step 200 (1.4207/1.2644), step 400 (1.2908/1.1614), val1m 1.2055, full val 1.3236.
- Speed: 3,087 vs 12,515 tok/s = 0.25x (probe 3,318 vs 11,992 = 0.28x). GEMM-only 0.13-0.27x per call. Why: sb-simd has no AVX-512 (OpenBLAS uses AVX-512 on this Xeon), and there is no VEX scalar load, so A is stored pre-broadcast.

## AVX-512 Lisp GEMM (gemm=lisp512), 2026-10-25
- CPU has avx512f (/proc/cpuinfo). SBCL 2.5.2 has no EVEX encoder and no 512-bit storage class, so `zmm-vops.lisp` defines a custom VOP `sb-vm::%zkern` whose generator hand-encodes EVEX instructions with `(inst byte ..)`: VPXORD, VMOVUPS zmm load/store, VADDPS zmm,m512 (beta=1), VFMADD231PS zmm, zmm, m32{1to16} (embedded broadcast from packed A). Uses only zmm16-31, which SBCL never allocates, and ends with VZEROUPPER. Disassembly proof: `gemm512/zkern-objdump.txt`.
- `lgemm512.lisp`: same API as lgemm.lisp, 6x32 tile (12 zmm accumulators), KC=256 MC=144 NC=1024. The VOP writes full tiles straight into C (alpha=1, beta 0/1); other cases go through a tmp tile. Kernel alone: 223 GF/s on 1 core.
- `gpt-eff.lisp gemm=lisp512` (default stays openblas; gemm=lisp is still AVX2; backup `efficiency/gpt-eff-pre-lisp512.lisp`).
- Random test (`gemm512/test.out`): 400 GEMMs, max rel err 1.108e-6 PASS; per-call 0.40-1.00x OpenBLAS (AVX2 was 0.13-0.27x).
- Gradcheck (`gemm512/gradcheck-lisp512.txt`): 6.525e-5 / 7.690e-5 / 7.161e-4 (same as gemm=lisp).
- 400-step run (`gemm512/short-run.sh`, ledger `alllisp-lisp512-400`, log `efficiency/17-method/runs/alllisp-lisp512-400.log`): step 200 1.4207/1.2644, step 400 1.2908/1.1614, val1m 1.2055, full val 1.3236 (identical to baseline). 8,629 tok/s = 0.69x OpenBLAS (12,515), 2.80x AVX2 Lisp (3,087).
