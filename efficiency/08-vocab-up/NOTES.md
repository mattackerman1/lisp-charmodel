### #8 interpretation and recommendation

- **v16384 improved on v8192 (1.0480 vs 1.0582). v32768 did not (1.0564),** so the doubling stops at 16384 and 65536 was never trained. The queue8 controller was stopped by request after v32768 started; the v32768 run itself finished normally. v32768 trails v16384 on val-1M at every checkpoint after 15 min (0.9322 vs 0.9257 at 60 min).
- **Why v32768 loses:** throughput collapses at a fixed model width. The output layer dominates: 7.7k tok/s and 30.8k chars/s, against 11.6k and 44.4k for v16384. It got only 6,745 steps (1.2 passes over the data). Its gap of +0.070 (v16384 +0.060, v8192 +0.034) shows that the 13.1M-param model, about two thirds of it embeddings, also memorises more per pass. The window gain is small: 4.01 chars/token against 3.82.
- **Speed:** the 3-min #5-config check before v32768 read 20.3k tok/s (normal), so the slowdown is the vocab, not the box. Full-val eval also takes longer (260 s vs 170 s), but that is outside the training time.
- **Memory:** tokenizer peak RSS was 1.9 GB (16k) and 1.8 GB (32k). The trainer fit in a 4 GB dynamic space at both sizes.
- **Recommendation:** use **vocab 16384** as the base (−0.025 vs #5's 2048, −0.010 vs v8192). Watch its overfitting gap: regularisation (dropout, #9) is the natural next test.
