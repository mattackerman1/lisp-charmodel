# big/: char transformer on a 9.58 MB Gutenberg corpus, Lisp vs PyTorch (phase A)

## Corpus (`corpus.txt`, built by `build_corpus.py`; sources and offsets in `corpus_meta.json`)
- 14 public-domain Project Gutenberg ebooks, downloaded 2026-10-17 from `https://www.gutenberg.org/cache/epub/<id>/pg<id>.txt`. They are concatenated in this order:
  - Pride and Prejudice (1342), Emma (158), Sense and Sensibility (161)
  - Great Expectations (1400), A Tale of Two Cities (98)
  - Jane Eyre (1260), Wuthering Heights (768), Dracula (345)
  - Notes from the Underground (600), Huckleberry Finn (76), Moby Dick (2701)
  - The Picture of Dorian Gray (174), Frankenstein (84), Alice in Wonderland (11)
- Cleaning:
  - The Gutenberg header and license are stripped (only text between the `*** START OF` and `*** END OF` markers is kept).
  - Text is normalized to ASCII: curly quotes become straight, em/en dashes become `--`/`-`, and accents are removed via NFKD. Any other non-ASCII is dropped.
  - CRLF line endings become LF, and runs of blank lines are capped.
- **9,579,366 chars, vocab 86.**
- Fixed split: val = last 10% (chars from 8,621,429 on; 957,937 val chars, 957,920 scored). The val set is most of Dorian Gray plus all of Frankenstein and Alice.
- Removed before finalizing, to stay within 5–10 MB and avoid an all-Melville val set: Les Miserables, David Copperfield, Middlemarch and Oliver Twist (downloaded but unused).

## Setup, identical in both implementations
- Model: pre-LN GPT, d=128, 4 layers, 4 heads, ctx 64, GELU (tanh approximation), LM head tied to wte. **812,544 params.**
- Dropout 0.1 on embeddings, attention probs, the attention-proj output and the MLP output.
- Init:
  - N(0, 0.02) for wte, wpe, qkv and fc.
  - N(0, 0.02/sqrt(2L)) for proj and fc2.
  - Biases 0; LayerNorm 1/0.
- Training:
  - AdamW: lr 1e-3, betas 0.9/0.99, eps 1e-8, wd 0.1 on matrices and embeddings.
  - Global grad-norm clip 1.0.
  - Schedule: 2% warmup, then cosine to lr/10.
  - Batch: 32 random windows (2048 tokens), float32.
- The budget is 3600 s of training time; eval time is excluded, and the LR schedule follows training time too. 8 threads, nice -n 10, run one after the other (`queue.sh`, `queue.log`).
- Eval: windows of 64 with stride 32, scoring the second half (32–63 chars of context).
  - Subset val (first 32k val targets) every 200 steps.
  - Full val every 600 s of training time, plus at the end.
- Code: `gpt-big.lisp` (gpt-drop.lisp plus a data= option, training-time budget and periodic full val) and `gpt_torch.py` (uses `F.scaled_dot_product_attention` with dropout_p on the attention probs).
- The two implementations use their own RNGs, so they draw the same distributions but not the same numbers.

## Results (3600 s of training time each, 8 threads; Lisp ran 11:25–12:31 CDT, PyTorch 12:31–13:33 CDT)
| | Lisp (gpt-big.lisp) | PyTorch 2.14 CPU (gpt_torch.py) |
|---|---|---|
| params | 812,544 | 812,544 |
| steps | 47,947 | 40,363 |
| steps/s (training time) | 13.32 | 11.21 |
| tokens/s (training time) | **27,276** | 22,962 |
| eval time (6 full + subset evals) | 300 s | 103 s |
| full val @600/1200/1800/2400/3000 s | 1.4408 / 1.3680 / 1.3323 / 1.3045 / 1.2753 | 1.4682 / 1.3888 / 1.3449 / 1.3184 / 1.2866 |
| **final full val (3600 s) = best** | **1.2645** | 1.2750 |
| val rises late? | no, falls at every checkpoint | no, falls at every checkpoint |

- **Per step, the two implementations agree.** Lisp at step 40,299 has full val 1.2753; PyTorch at step 40,363 (its final) has 1.2750. The right panel of `transformer-compare.png` shows the curves on top of each other by step. Lisp's 0.0105 lead at 3600 s comes entirely from running 19% more steps in the same time.
- Neither run overfits on 8.6M training chars. About 98M (Lisp) or 83M (PyTorch) tokens is roughly 11 or 10 passes, and val still falls at the end, so a longer run or a bigger model would likely help.
- **Why PyTorch is slower here** (`bench_torch.py`, `bench_torch.log`: 150-step probes on the idle box after both runs):
  - Dropout is the main cost. 8 threads with no dropout gives 21.54 steps/s. Dropout only on embeddings and residuals gives 14.75. Adding attention dropout (the actual config) gives 11.37.
    - With `dropout_p>0`, `scaled_dot_product_attention` on CPU cannot use its fused kernel. It falls back to building the full probability matrix plus a separate Bernoulli mask.
    - Every `F.dropout` also generates and stores a random mask through separate kernels.
    - The Lisp code fuses mask generation (xorshift) and the multiply into its own loops, so dropout costs it only about 4% (13.78 vs 14.39 steps/s on Tiny Shakespeare).
  - Threads don't scale for this tiny model: 1/2/4/8 threads give 5.28/8.07/11.23/11.37 steps/s. With 32x64 tokens and d=128, the GEMMs are too small, so per-op dispatch and threading overhead dominate. The Lisp code instead splits the batch into 8 independent per-thread workers, each running single-threaded sgemm.
  - Writing the attention by hand instead of SDPA is slightly slower (10.71 steps/s), so SDPA itself is not the problem.
  - Without dropout PyTorch would be about 1.5x faster than the Lisp run's 13.3 steps/s. I didn't measure the Lisp code without dropout on this corpus.
- Lisp evaluation is slower: 300 s vs 103 s total eval time, excluded from both budgets. A single full-val pass took about 27 s in the Lisp smoke test. I didn't time a PyTorch pass separately.

## Samples (300 chars, T=0.8, from a newline)
Lisp:
```
the curse of him in the private letters, in the same sure. But they
said, "Can you say here are standing in a pence as a gentleman, do
you make you, and I shall not accommunicate the morning. If you were of
it, for I have no expression of the remembrance of God's being able
to find companion to the
```
PyTorch:
```
"He is the way to live in the last of his reason," said she, "if you
will be considered a sad of looking at him her eye again, and _do_
not precise her time about for a reply and vary complainted to care
to change. You wouldn't suppose that sort of lawyer circumstance
endure to him. And then they c
```

Files: corpus.txt, corpus_meta.json, build_corpus.py, raw/, gpt-big.lisp, gpt_torch.py, bench_torch.py, queue.sh, queue.log, {lisp,torch}.log/.csv/-full.csv, torch.pt, transformer-compare.png, plot_compare.py, backup/.
GRUs (phase B) were not run.

# Phase B: GRU H=288 with dropout 0.25 on the same corpus, Lisp vs PyTorch
- Config: the GD288 setup. GRU H=288, gate order z, r, n.
  - Non-recurrent dropout 0.25 on the input projection Ein[x] (3H per position) and on h before the output layer. The recurrence uses the undropped h.
  - Adam (betas 0.9/0.999, eps 1e-8, no weight decay), peak lr 2e-3, 2% warmup then cosine to lr/10, global clip 1.0.
  - TBPTT T=32 over 32 contiguous train streams with state carried. **349,718 params** (vocab 86).
  - Init: Ein N(0, 0.1); Wh and Why N(0, 1/sqrt(H)); biases 0.
- Budget and checkpoints: 3600 s of training time (eval excluded), full val every 600 s, 8 threads, nice -n 10, run one after the other (`queue_b.sh`, `queue_b.log`).
- Code:
  - `gru-big.lisp` is gru-drop.lisp plus a data= option, a training-time budget and periodic full val. It uses float32, OpenBLAS sgemm and 8 workers of 4 streams each.
  - `gru_torch.py` is a hand-written GRU loop with the same equations, init, dropout placement and data streams.
- Eval differs from the transformer's. GRU full val runs 32 streams of 29,935 val chars each, with state carried, so context is effectively unbounded. The transformer's full val gives each scored char only 32–63 chars of context. Both score about 958k val targets.

## Phase B results (3600 s of training time each, 8 threads; Lisp 14:04–15:09 CDT, PyTorch 15:09–16:11 CDT)
| | GRU Lisp | GRU PyTorch | Transformer Lisp (phase A) | Transformer PyTorch (phase A) |
|---|---|---|---|---|
| params | 349,718 | 349,718 | 812,544 | 812,544 |
| steps | 198,546 | 72,709 | 47,947 | 40,363 |
| tokens/s (training time) | **56,475** | 20,680 | 27,276 | 22,962 |
| full val @600 s | 1.3990 | 1.4591 | 1.4408 | 1.4682 |
| @1200 s | 1.3747 | 1.4145 | 1.3680 | 1.3888 |
| @1800 s | 1.3537 | 1.3852 | 1.3323 | 1.3449 |
| @2400 s | 1.3360 | 1.3659 | 1.3045 | 1.3184 |
| @3000 s | 1.3247 | 1.3522 | 1.2753 | 1.2866 |
| **final = best (3600 s)** | **1.3171** | 1.3475 | **1.2645** | 1.2750 |
| train subset 32x3000 (dropout off) | 1.1524 | 1.1832 | – | – |
| val rises late? | no | no | no | no |

- **The transformer wins on this corpus despite the eval handicap.** The best GRU (Lisp, 1.3171) is 0.053 behind the Lisp transformer (1.2645) and 0.042 behind the PyTorch transformer (1.2750). That's with the GRU's full val carrying state across each 29,935-char stream, while the transformer sees only 32–63 chars of context per scored char. On Tiny Shakespeare at 1800 s the gap was similar: GD288 1.4974 vs D1 1.4379.
  - The GRU leads only early: 1.3990 vs 1.4408 at 600 s.
  - The transformer pulls ahead between 1200 and 1800 s and keeps widening the gap.
- **Lisp vs PyTorch GRU.** The Lisp GRU is 2.7x faster (55.15 vs 20.19 steps/s) and ends 0.030 better (1.3171 vs 1.3475).
  - By step count PyTorch is not behind. It reached 1.3475 at step 72,709, while Lisp had 1.3747 at step 66,069.
  - But both runs schedule LR by wall-clock time. So at a given step, PyTorch had already annealed further (it ended at lr/10, while Lisp was still near 0.8x peak LR at step 66k). Per-step quality can't be cleanly compared here. The difference is throughput.
- **Why the PyTorch GRU is slow.** `gru_torch.py` runs the recurrence as a Python loop: 32 timesteps per chunk, each a 32x288 @ 288x864 matmul plus about 10 small elementwise ops, all recorded by autograd.
  - Per-op dispatch and autograd overhead dominate these tiny ops, and extra threads don't help. Short 25 s probes (`bench_gru_torch.log`) gave 11.89 steps/s on 1 thread and 23.78 on 4. The full run on 8 threads averaged 20.19.
  - The Lisp version gives each of 8 workers 4 streams with fused per-element gate loops and single-threaded sgemm, and does the output layer and dWh as one big GEMM per chunk.
  - `torch.nn.GRU` (fused kernel) would be faster, but it can't apply dropout to the 3H input projection without changing the architecture, so I kept the exact hand-written match.
- Neither GRU overfits on 8.6M training chars. Val falls at every checkpoint, and the train-subset vs val gap (about 0.16) is the same as on the 1M-char corpus.

## Phase B samples (300 chars, T=0.8, from a newline)
GRU Lisp:
```
loss merely in the eled, and as I don't know, that as I could have too wonder to reason to have
really not objected a ball to a gangame, the mind broke before the hand through
leaving my in the shor, and he that engaged her young man is, and respect before, in the
signs of weeks, and gazed at the gi
```
GRU PyTorch:
```
have discovered every of a lady of consideration, and which all the
next disappointment, and the fallow done the seen with the deach, and satisfailing
the nemold of cold bits, hid by the reproach of the difference, and now but
was the flowing minutes and had when I was like the caprice's hair resent
```
Phase B files: gru-big.lisp, gru_torch.py, queue_b.sh, queue_b.log, gru-{lisp,torch}.log/.csv/-full.csv, gru-torch.pt, bench_gru_torch.log, gru-compare.png, plot_gru.py.
