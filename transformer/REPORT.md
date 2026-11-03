# Char-level decoder-only transformer in SBCL (hand-written backprop)

Code: `gpt.lisp`. It is a pre-LN GPT with token embeddings, learned position embeddings, causal multi-head self-attention, LayerNorm, a GELU MLP (4d), residuals, and an LM head tied to wte.
- Training: AdamW (betas 0.9/0.99, wd 0.1 on matrices and embeddings, grad clip 1.0), 2% warmup, then cosine decay to lr/10 over the wall-clock budget. Peak lr 1e-3.
- Speed: float32, OpenBLAS sgemm through sb-alien, 8 Lisp worker threads with 4 sequences each, and OpenBLAS at 1 thread.
- Size: d=128, 4 layers, 4 heads, ctx 64, batch 32 seqs (2048 tokens per step). **809,856 params.**
- Data: same full-data fixed split as the ablation (train 1,003,854 chars, val = last 10%, 111,540 chars). Training windows are sampled at random positions.
- Every run used `nice -n 10`. Before the 8-thread runs I checked `ps`; there were no science-team or Experimenter jobs.

Gradient check (`gradcheck.log`), using a tiny GPT (V=5, d=8, L=2, heads=2, T=4), 2 workers x 2 seqs, 1832 params:
- float64: max rel err 3.9e-6, max abs err 9.0e-11, global |g-num|/|num| 3.3e-10. **PASS.**
- float32 (eps 1e-2, for reference only): global rel 1.1e-4, max abs 7.0e-5. Max rel err is 1.0, on entries that are essentially zero.

Val protocol: the val set is cut into windows of 64 with stride 32, and only the second half of each window is scored. So every scored char has 32–63 chars of context, and 111,520 of the 111,540 targets are scored. The GRU's full val instead streams 32 carried-state streams over the val set, which gives it unbounded context. The periodic "subset" curves cover the first 32k val chars, which are easier than the full set for both models.

| Model | Budget | Params | steps | tokens/s | full val | best subset val |
|---|---|---|---|---|---|---|
| GRU-F (H=144) | 480 s | 100,577 | 85,800 | 183,019 | 1.6447 | 1.5154 |
| GRU-C (best GRU at 480 s, double+BLAS 1 thr) | 480 s | 100,577 | 20,592 | 43,929 | 1.6135 | 1.5166 |
| **Transformer T480** | 480 s | 809,856 | 7,185 | 30,654 | **1.5048** | 1.4151 |
| GRU-G (H=144) | 1800 s | 100,577 | 311,877 | 177,423 | 1.6391 | 1.5054 |
| Transformer T1800 | 1800 s | 809,856 | 25,904 | 29,472 | 1.6271 | 1.4100 (t=875 s) |

(tokens/s includes time spent on periodic validation.)

Findings
- At 480 s the transformer beats every GRU run by about 0.11–0.14 nats/char on the full val (1.5048 vs 1.6135–1.6447). It processes 6x fewer tokens per second than GRU-F, and it scores with only 32–63 chars of context.
- At 30 min the transformer overfits. Its subset val bottoms out at 1.4100 around 875 s (lr still about 6e-4), then climbs to 1.5254 by the end. Meanwhile train(last batch) falls to about 1.1 (25,904 steps x 2048 tokens is about 53 epochs). Its final full val, 1.6271, is *worse* than T480's 1.5048 and only slightly better than GRU-G's 1.6391. With 810k params on 1M chars, a longer run needs dropout, more weight decay, or early stopping. None of these were tried.
- Plot: `transformer-vs-gru.png`, val subset vs wall-clock for GRU-F/G and T480/T1800.

Samples (300 chars, T=0.8): `s480.txt`, `s1800.txt`; the full samples are also in `T480.log` and `T1800.log`.
