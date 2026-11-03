# lisp-charmodel

A small GPT-style language model written from scratch in Common Lisp (SBCL), trained on a ~100 MB Project Gutenberg corpus, and a long series of experiments on how to get the lowest validation loss per unit of compute on one CPU box. Everything that matters at training time (model, autograd, optimizers, matrix multiply, tokenizer) is Lisp. Plotting and a few checks are Python.

The main file is `efficiency/gpt-eff.lisp`. The full experiment log is `efficiency/REPORT.md`, and every run since #17 is one row in `efficiency/17-method/results.tsv`.

## The model

- Decoder-only transformer: d_model 256, 6 layers, 8 heads, context 64 tokens, batch 64 sequences (4,096 tokens per step). About 5.3M parameters with the 16k vocab.
- Tokenizer: byte-pair encoding, vocab 16,384 (about 3.82 training characters per token).
- Optimizer: Muon (Newton-Schulz orthogonalized momentum, lr 0.01) on the block weight matrices, AdamW (lr 1e-3) on embeddings, norms and biases.
- Schedule: WSD (warmup, long constant LR, then decay to 0). The best recipe uses a sqrt-shaped decay over the last 2,996 steps (`dshape=sqrt`).
- Matrix multiply: an AVX-512 GEMM written in Lisp with custom SBCL VOPs (`all-lisp/lgemm512.lisp`), the default since 2026-10-25 (`gemm=lisp512`). It runs at about 0.84x OpenBLAS with identical losses. OpenBLAS is still available with `gemm=openblas`.
- Output head: full softmax by default. `ssm=1024 ssm-alpha=0.75` turns on a sampled softmax (1,024 negatives from unigram^0.75, with logQ correction) for training only; evaluation is always the exact full softmax.
- Runs are deterministic given seed and step count. Full-state checkpoints and exact resume are described in `efficiency/HARDENING.md`.

## Target metric

The goal is the best validation loss for the compute spent, which in practice means training more characters per second without hurting loss per step. Two numbers are tracked:

- **Full val**: nats per character over the last 10% of the corpus (9,988,883 characters). This is the headline number. Lower is better.
- **Val 1M**: the first 1M val characters, used for cheap in-run checks.

Throughput is reported as clean characters (or tokens) trained per second, with stalls of the VM removed. A change is kept only if it beats the baseline by more than 2x the measured seed noise (about 0.0018 nats/char at the 15-minute proxy). Val loss is the guardrail: a speedup that makes loss per step worse is only worth it if it still wins at a fixed time budget.

## Requirements

- SBCL 2.5.x with `sb-simd` (the scripts expect it at `/home/box/opt/sbcl/bin/sbcl`; any `sbcl` on PATH works).
- A CPU with AVX-512 for `gemm=lisp512`; otherwise OpenBLAS 0.3.x for `gemm=openblas` (path via `GPT_OPENBLAS`).
- Python 3 with numpy and matplotlib, only for charts and some checks.
- `efficiency/hardening/env.sh` sets PATH, `SBCL`, `GPT_OPENBLAS` and `PY` for the box layout; `efficiency/hardening/preflight.sh` checks the toolchain.

## Build the data and tokenize

The corpus and token files are not in the repo (too large). To rebuild:

1. Corpus: `huge/fetch.sh` downloads the Gutenberg books listed in `huge/ids.txt` into `huge/raw/`, and `huge/build_corpus.py` builds `huge/corpus.txt` (the last 9,988,894 characters are the val split).
2. Tokens, in Lisp (#23, single-threaded):

   ```
   cd efficiency/23-lisp-bpe
   sbcl --dynamic-space-size 6000 --script bpe.lisp 16384 out
   ```

   This writes `out/merges16384.hex`, `out/vocab16384.hex` and `out/tokens16384.u16` and does a decode round-trip check. The output is byte-identical to the original Python/HF-tokenizers version (`efficiency/08-vocab-up/train_bpe.py`), and encoding is about 3x faster (3.08 s vs 9.36 s). Proof is in `efficiency/23-lisp-bpe/proof/`.

   Training reads `08-vocab-up/tokens16384.u16` and `08-vocab-up/vocab16384.hex` (the vocab hex is in the repo; copy or point the token file there).

## Train

Run from `efficiency/`. The #21 run (best model), 100,604 steps, full softmax:

```
sbcl --dynamic-space-size 4000 --script gpt-eff.lisp --run \
  tokens=08-vocab-up/tokens16384.u16 vocab=08-vocab-up/vocab16384.hex vstart=23505859 fullval-targets=238949 \
  d=256 layers=6 heads=8 ctx=64 batch=64 threads=8 dropout=0 lr=1e-3 optim=muon mlr=0.01 ew=f32 \
  sched=wsd warm=0.0016 lrmin=0 steps=100604 dsteps=2996 dshape=sqrt seed=1 gemm=lisp512 \
  fullval-every=900 savefinal=1 tag=21-long/run/e21-v16-full-100k
```

The 1-hour run with the sampled head (#19 full, best 1-hour model):

```
sbcl --dynamic-space-size 4000 --script gpt-eff.lisp --run \
  tokens=08-vocab-up/tokens16384.u16 vocab=08-vocab-up/vocab16384.hex vstart=23505859 fullval-targets=238949 \
  d=256 layers=6 heads=8 ctx=64 batch=64 threads=8 dropout=0 lr=1e-3 optim=muon mlr=0.01 ew=f32 \
  sched=wsd warm=0.01 decay=0.2 lrmin=0 steps=15700 seed=1 ssm=1024 ssm-alpha=0.75 \
  fullval-every=900 savefinal=1 ackpt=540 ackpt-keep=2 tag=19-full/ssm/e19f-v16-s1024-1h
```

For long runs on a box that can restart, put the args in a `run.conf` (see `efficiency/21-long/run/run.conf` and `efficiency/hardening/run.conf.example`) and launch with `efficiency/hardening/supervise.sh RUNDIR`, which takes full-state checkpoints (`ackpt=`) and resumes exactly. `efficiency/17-method/run.sh` runs one experiment through the ledger (frozen copy of the code, speed probe, one row in `results.tsv`).

## Key results

All numbers are full val in nats/char, from `efficiency/REPORT.md` and `efficiency/17-method/results.tsv`. One seed each unless noted.

| run | what | steps | full val | clean chars/s |
|---|---|---|---|---|
| baseline | AdamW, cosine, characters, ctx 128, 1 h | 18,899 | 1.1680 | |
| #2 | Muon + WSD, characters, 1 h | 17,263 | 1.1419 | |
| #4 | + BPE 2,048 tokens, ctx 64, 1 h | 16,447 | 1.0775 | |
| #8 | BPE 16,384, full softmax, OpenBLAS, 1 h | 10,194 | 1.0480 | 44,354 |
| #18 confirm | sampled softmax ssm=1024, 1 h | 12,960 | 1.0441 | 62,881 |
| #19 full | #18 + faster gradient reduce (+15% speed), 1 h | 15,700 | **1.0388** (best 1 h) | 69,462 |
| #13 | cyclic restarts, 11 h | 100,604 | 1.0152 | |
| #17 final | sqrt decay from a 10 h constant-LR checkpoint | 100,604 | 1.0080 | |
| #20 | #19 code, ssm=1024, one WSD run | 100,604 | 1.0138 | 71,220 |
| **#21** | **#20 with the full softmax head** | **100,604** | **1.0060** (best) | 40,894 |

What the long runs say: at a fixed step count the full softmax is better (#21 beats #20 by 0.0078); at a fixed time budget the sampled head wins because it is 1.74x faster (#19 at 1 h).

## Tried and dropped

- **#22a knowledge distillation** from #21 (16 sampled teacher ids per position, kd-alpha 0.5). It won at the 15-minute proxy (mean -0.0054 over two seeds) but lost the 1-hour confirm: 1.0462 vs 1.0388 without KD, about 6% slower, plus a one-off teacher dump of about 4.6 CPU-hours. Lesson: a short proxy with a short decay can misrank changes that act early in training.
- **#22b modded-nanogpt tricks**, each alone at two seeds on the 15-minute proxy (baseline mean 1.09835, keep bar -0.0036). None kept: QK-norm -0.0027 (closest), value embeddings -0.00045, zero-init +0.00055, softcap 30 +0.00045 (and about 5x slower eval), U-Net skips +0.00095, untied head +0.0025, ReLU^2 +0.0059.
- **#17 tricks**: decay-side tweaks (sqrt shape, lower AdamW or Muon lr) each helped by 0.0015 to 0.0020 but did not stack; ReLU^2, zero-init and softcap 15 were discarded.
- **#19 speed candidates**: a GEMM packing change (+0.8%) and a symmetric Newton-Schulz Gram (+0.6%) were inside box noise and rejected. bf16 AMX GEMM was rejected in #5. Dropout 0.1 (#9) was not kept.

## Open next steps

- **Cheaper Muon optimizer.** The optimizer update (Newton-Schulz GEMMs plus AdamW) is about 34 of ~235 ms per step in the #19 profile, one of the remaining big costs.
- **QK-norm 1-hour confirm.** QK-norm kept in #17 and was the closest miss in #22b (-0.0027, both seeds agree). A 1-hour run against #19 full (1.0388) would settle it.

## Layout

```
efficiency/            the main experiment series
  gpt-eff.lisp         the model and trainer (current code)
  gpt-eff-pre-*.lisp   frozen earlier versions, one per change
  REPORT.md            full experiment log, #1 to #23
  HARDENING.md         checkpoints, exact resume, supervisor, toolchain notes
  MACRO.md             notes on the macro experiment (#15)
  01-lr ... 23-lisp-bpe/  one folder per experiment: scripts, CSV curves, charts, notes
  17-method/results.tsv  the run ledger (every run since #17)
  17-method/run.sh     ledger runner; 17-method/charts.py plots
  19-speed/            profiling, benches and the #19 diff
  22-research/         KD and tricks code, gradchecks, notes
  23-lisp-bpe/bpe.lisp Lisp BPE trainer/encoder, with proof/
  hardening/           supervise.sh, check-runs.sh, preflight.sh, env.sh
all-lisp/              pure-Lisp AVX-512 GEMM (lgemm512.lisp) and its audit
huge/, big/            corpus fetch/build scripts and the earlier Lisp vs PyTorch comparison
charmodel.lisp, rnn/, fast-gru/, transformer/, fast-gpt/, ablation/, next/, pytorch/
                       earlier character-level models (RNN, GRU, first transformer) and comparisons
```

Not in the repo: model weights and checkpoints (`*.f32`, `*.bin`), token files (`*.u16`), the corpus and raw books, logs, the Python venv, and the SBCL/OpenBLAS toolchain.
