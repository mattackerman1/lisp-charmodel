# Efficiency series: 4.8M-param Lisp transformer, 1-hour budget per run

Model: d256, 6 layers, 8 heads, ctx 128, batch 32x128, dropout 0, 100 MB Gutenberg corpus (`../huge/corpus.txt`). 8 threads, `nice -n 10`, one job at a time (`queue.sh`, resumable).
Every run is scored on the same fixed measures, which are kept as separate series and never mixed:
* **val 1M**: the first 1,000,000 val characters, every 15 min of training time and at the end.
* **full val**: all 9,988,894 val characters (the last 10% of the corpus), once at the end. This slice is harder than the 1M slice, so the two numbers differ.
Train-batch loss vs the 32k-char val subset (both every 200 steps) is the overfitting check, shown as the last-10% average train vs val.

| run | val 1M at 15/30/45/60 min + end | full val (end) | tok/s | steps | overfit: train vs val subset | status |
|---|---|---|---|---|---|---|
| baseline: AdamW lr 1e-3, cosine (2% warmup, to lr/10) | 1.2509 / 1.1675 / 1.1018 / 1.0721 | 1.1680 | 21,501 | 18899 | 1.157 vs 1.026 | done |
| #1: WSD lr 1e-3 (1% warmup, constant, linear decay to 0 over last 20%) | 1.2296 / 1.1692 / 1.1518 / 1.0506 | 1.1477 | 24,912 | 21896 | 1.160 vs 1.026 | done |
| #2: Muon (mlr 0.01) on block matrices + AdamW rest, WSD | 1.1999 / 1.1587 / 1.1336 / 1.0483 | 1.1419 | 19,641 | 17263 | 1.135 vs 1.015 | done |
| #3: #2 (mlr 0.01) + train ctx 64 (64 seqs) then 128 (sw0.75-posoff) | 1.3854 / 1.3502 / 1.3767 / 1.0575 | 1.1490 | 21,385 | 18796 | 1.168 vs 1.028 | done |
| #4: #2 + BPE-2048 tokens, ctx 64 tok, 64 seqs (mlr 0.01); tok/s = TOKENS/s | 1.0832 / 1.0447 / 1.0253 / 0.9612 | 1.0775 | 18,712 | 16447 | 1.042 vs 0.935 | done |
| #5: #4 + single-precision GELU (bf16 AMX GEMM rejected, see notes); tok/s = TOKENS/s | 1.0736 / 1.0409 / 1.0248 / 0.9595 | 1.0729 | 20,186 | 17742 | 1.044 vs 0.932 | done |

All losses are nats per character (#4: total val nats / val characters scored), so all rows compare directly. #4 chars/s = tokens/s x 2.950 chars/token (train) = 55,197 chars/s.

## #1 LR sweep (15 min each, schedule fit to 15 min, scored on val 1M at the end)

| run | val 1M at end | tok/s | steps |
|---|---|---|---|
| cos-1e-3 | 1.1725 | 25,430 | 5588 |
| wsd-1e-3 | 1.1708 | 21,978 | 4830 |
| wsd-2e-3 | 1.1789 | 22,170 | 4872 |
| wsd-3e-3 | 1.1926 | 22,681 | 4984 |
| wsd-5e-3 | 1.1951 | 23,615 | 5189 |
| wsd-5e-4 | 1.1802 | 24,694 | 5426 |

Best WSD peak LR: **1e-3**

Charts: `10-long/chart.png` (#10), `09-dropout/chart.png` (#9), `08-vocab-up/chart.png` (#8 vs v8192), `07-vocab/chart.png` (#7 vocab vs #5), `06-size/chart.png` (#6 sizes vs #5), `05-precision/chart.png` (#5 vs #4/#2/baseline), `04-bpe/chart.png` (#4 vs #3/#2/#1/baseline), `04-bpe/overfit.png`, `03-ctx/chart.png` (#3 vs #2 vs #1 vs baseline), `03-ctx/overfit.png`, `02-muon/chart.png` (#2 vs #1 vs baseline), `02-muon/overfit.png`, `01-lr/chart.png` (#1 vs baseline: val-1M series + full-val end markers), `01-lr/overfit.png` (train vs val subset), `01-lr/sweep.png`, `baseline/chart.png`.

## #2 Muon sweep (15 min each, WSD, AdamW lr 1e-3 for embeddings/norms/biases; val 1M at end)

| run | val 1M at end | tok/s | steps |
|---|---|---|---|
| control-adamw-wsd-1e-3 | 1.1636 | 24,018 | 5278 |
| muon-0.005 | 1.1337 | 19,402 | 4264 |
| muon-0.01 | 1.1317 | 21,360 | 4694 |
| muon-0.02 | 1.1855 | 18,563 | 4080 |
| muon-0.04 | 2.3544 | 10,187 | 2239 |

AdamW speed probe (3 min, right before the 1-h Muon run): 23,904 tok/s

Muon optimizer check (tiny matrices, 2 steps, vs numpy float64 reference of the Keller Jordan algorithm): see `02-muon/muon_check.log`.

## #3 short-context sweep (15 min each, Muon + WSD; eval always at ctx 128; val 1M at end)

| run | val 1M at end | tok/s | steps |
|---|---|---|---|
| ctx128-control | 1.1335 | 19,375 | 4258 |
| ctx64-sw0.6-posoff | 1.1506 | 20,845 | 4581 |
| ctx64-sw0.75-posoff | 1.1444 | 22,417 | 4926 |
| ctx64-sw0.75 | 1.1515 | 20,551 | 4516 |
| ctx64-sw0.85-posoff | 1.1680 | 20,933 | 4601 |

ctx-128 Muon speed probe (3 min, right before the 1-h #3 run): 18,832 tok/s

Short-context gradient check (single float, finite differences): `03-ctx/gradcheck.log`.

## #4 BPE sweep (15 min each; val 1M in nats/char at end)

| run | val 1M at end (nats/char) | tokens/s | steps |
|---|---|---|---|
| bpe-muon-0.01 | 1.0289 | 18,683 | 4106 |
| bpe-muon-0.02 | 1.0437 | 19,442 | 4272 |

#2 char-config speed probe (3 min, right before the 1-h #4 run): 19,743 tok(=char)/s

Tokenizer survey, gradient checks: `04-bpe/survey.json`, `04-bpe/gradcheck.log`.

#4 fp32 speed probe (3 min, right before the 1-h #5 run): 19,874 tokens/s

## #6 model-size sweep (1 h each, base #5; nats/char)

| model | params | val 1M at 15/30/45/60 min | full val | tok/s | steps | passes over train | train(last 10%) - full val | 3-min #5-config probe before |
|---|---|---|---|---|---|---|---|---|
| #5 d256-L6 | 5.28M | 1.0736 / 1.0409 / 1.0248 / 0.9595 | 1.0729 | 20,186 | 17742 | 2.38 | -0.029 | 19,874 tok/s |
| d192-L6 | 3.08M | 1.0629 / 1.0413 / 1.0307 / 0.9675 | 1.0825 | 32,171 | 28276 | 3.80 | -0.024 | 20,825 tok/s |
| d256-L8 | 6.86M | 1.0788 / 1.0360 / 1.0246 / 0.9574 | 1.0729 | 16,392 | 14407 | 1.94 | -0.027 | 21,847 tok/s |
| d384-L6 | 11.46M | 1.0982 / 1.0553 / 1.0289 / 0.9642 | 1.0803 | 9,605 | 8442 | 1.13 | -0.021 | 21,363 tok/s |

### #6 interpretation

- **No size beats #5.** d256-L8 (6.9M) ties #5 (d256-L6, 5.3M) on full val (1.0729 vs 1.0729). On val-1M at 60 min it is slightly ahead (0.9574 vs 0.9595), but that gap is within the run-to-run noise we've seen. The smaller d192 (+0.010) and the larger d384 (+0.007) are both worse on full val.
- **Compute is the limit, not overfitting.** Moving from d192 to d384 cuts the passes over the data from 3.8 to 1.1, and the train-vs-full-val gap only shrinks from 0.024 to 0.021 (0.029 for #5). With data repetition this mild, the larger models don't gain from seeing fresh data, and they lose too many steps (d384: 8.4k steps vs 17.7k).
- **Speed confound:** the probes at the #5 config before each run measured 20.8k–21.8k tok/s, versus 19.9k before #5's run. The box was 5–10% faster during #6, which favours the #6 runs slightly. Even with that edge, none of them won.
- **Recommendation:** keep #5 (d256-L6) as the base. It matches d256-L8 with 23% fewer params and about 23% more tok/s. At this budget, the curve is flat around 5–7M params.
- Note: 1M-val vs full-val rankings differ (d192 wins val-1M at 15 min, loses full val); always decide on full val.


## #7 BPE vocab sweep (d256-L6, ctx 64 tokens, 1 h each, base #5; nats/char)

| vocab | chars/token train / val | chars per 64-token window | params | val 1M at 15/30/45/60 min | full val | tok/s | chars/s | steps | full val - train(last 10%) | #5-config probe before |
|---|---|---|---|---|---|---|---|---|---|---|
| 2048 (#5) | 2.95 / 3.00 | 189 | 5.28M | 1.0736 / 1.0409 / 1.0248 / 0.9595 | 1.0729 | 20,186 | 59,545 | 17742 | +0.029 | 19,874 tok/s |
| 512 | 2.25 / 2.29 | 144 | 4.89M | 1.1030 / 1.0687 / 1.0492 / 0.9802 | 1.0938 | 21,235 | 47,869 | 18664 | +0.015 | 14,935 tok/s |
| 1024 | 2.61 / 2.65 | 167 | 5.02M | 1.0934 / 1.0579 / 1.0413 / 0.9728 | 1.0871 | 18,666 | 48,659 | 16406 | +0.027 | 20,778 tok/s |
| 4096 | 3.28 / 3.32 | 210 | 5.80M | 1.0561 / 1.0283 / 1.0086 / 0.9493 | 1.0656 | 17,261 | 56,531 | 15172 | +0.038 | 19,385 tok/s |
| 8192 | 3.57 / 3.62 | 229 | 6.85M | 1.0434 / 1.0071 / 0.9995 / 0.9388 | 1.0582 | 13,866 | 49,568 | 12187 | +0.034 | 18,004 tok/s |

### #7 interpretation and recommendation

- **Full val falls with every vocab doubling:** 512 1.0938, 1024 1.0871, 2048 (#5) 1.0729, 4096 1.0656, 8192 **1.0582**. Each doubling from 1024 up gains about 0.007–0.014. The val-1M curve gives the same ranking at every 15-min checkpoint.
- **Why bigger vocab wins:** at a fixed 64-token context, 8192 sees 229 characters per window against 189 for #5, and each step covers more text. That outweighs the throughput cost: tok/s falls from 20.2k to 13.9k, chars/s from 59.5k to 49.6k, and steps from 17.7k to 12.2k. Context length is a confound here: some of the gain is the longer context in characters, not the tokenizer itself. Splitting the two apart would need a matched-characters control.
- **Params confound:** the bigger output layer and embeddings add parameters (6.85M vs 5.28M). Those are embedding parameters and cheap per token, but some of the gain may come from them.
- **Overfitting:** the gap grows with vocab, from +0.015 at 512 to +0.038 at 4096 and +0.034 at 8192. The larger vocabs need fewer passes over the data (~2.0 at 8192 vs 2.4 for #5), so the gap stays bounded. It is not the limiting factor yet.
- **Speed confounds:** the 3-min checks at the #5 config ranged from 14.9k (before v512, an outlier) to 20.8k tok/s, so box speed varied about 5–10% between runs. The v4096 and v8192 checks (19.4k and 18.0k) were below #5's 19.9k, so those runs were if anything slightly handicapped. The ranking is not explained by box speed.
- **Recommendation:** adopt **vocab 8192** as the new base (full val 1.0582, −0.0147 vs #5). Vocab 4096 (1.0656) is the cheaper fallback if speed matters (+24% tok/s over 8192). Larger vocabs (16k) could be tried next, but the gains are shrinking, the output layer is starting to dominate compute, and the overfitting gap is growing. uint16 token ids still fit up to 65,536.


## #8 vocab sweep upward (conditional; d256-L6, ctx 64 tokens, 1 h each, base v8192; nats/char)

| vocab | chars/token train / val | chars per window | params | val 1M at 15/30/45/60 min | full val | tok/s | chars/s | steps | full val - train(last 10%) | #5-config probe before |
|---|---|---|---|---|---|---|---|---|---|---|
| 8192 (#7 best) | 3.57 / 3.62 | 229 | 6.85M | 1.0434 / 1.0071 / 0.9995 / 0.9388 | 1.0582 | 13,866 | 49,568 | 12187 | +0.034 | 18,004 tok/s |
| 16384 | 3.82 / 3.86 | 245 | 8.95M | 1.0380 / 0.9922 / 0.9790 / 0.9257 | 1.0480 | 11,597 | 44,354 | 10194 | +0.060 | 18,942 tok/s |
| 32768 | 4.01 / 4.05 | 256 | 13.14M | 1.0459 / 0.9986 / 0.9835 / 0.9322 | 1.0564 | 7,674 | 30,754 | 6745 | +0.070 | 20,259 tok/s |

### #8 interpretation and recommendation

- **v16384 improved on v8192 (1.0480 vs 1.0582). v32768 did not (1.0564),** so the doubling stops at 16384 and 65536 was never trained. The queue8 controller was stopped by request after v32768 started; the v32768 run itself finished normally. v32768 trails v16384 on val-1M at every checkpoint after 15 min (0.9322 vs 0.9257 at 60 min).
- **Why v32768 loses:** throughput collapses at a fixed model width. The output layer dominates: 7.7k tok/s and 30.8k chars/s, against 11.6k and 44.4k for v16384. It got only 6,745 steps (1.2 passes over the data). Its gap of +0.070 (v16384 +0.060, v8192 +0.034) shows that the 13.1M-param model, about two thirds of it embeddings, also memorises more per pass. The window gain is small: 4.01 chars/token against 3.82.
- **Speed:** the 3-min #5-config check before v32768 read 20.3k tok/s (normal), so the slowdown is the vocab, not the box. Full-val eval also takes longer (260 s vs 170 s), but that is outside the training time.
- **Memory:** tokenizer peak RSS was 1.9 GB (16k) and 1.8 GB (32k). The trainer fit in a 4 GB dynamic space at both sizes.
- **Recommendation:** use **vocab 16384** as the base (−0.025 vs #5's 2048, −0.010 vs v8192). Watch its overfitting gap: regularisation (dropout, #9) is the natural next test.


## #9 dropout 0.1 on v16384 (otherwise identical, 1 h; nats/char)

| run | val 1M at 15/30/45/60 min | full val | tok/s | steps | full val - train(last 10%) |
|---|---|---|---|---|---|
| v16384 dropout 0 | 1.0380 / 0.9922 / 0.9790 / 0.9257 | 1.0480 | 11,597 | 10194 | +0.060 |
| v16384 dropout 0.1 | 1.0580 / 1.0178 / 0.9977 / 0.9520 | 1.0760 | 10,774 | 9470 | +0.019 |

Note: with dropout the train-batch loss is measured with dropout on, which raises it and shrinks the apparent gap; the gaps are not comparable.

### #9 interpretation

- **Dropout 0.1 hurts badly:** full val 1.0760 vs 1.0480 without dropout (+0.028 nats/char). That erases most of the vocab gain: the result is worse than v8192 (1.0582) and about level with #5 (1.0729). It is behind on val-1M at every checkpoint (0.9520 vs 0.9257 at 60 min).
- **Why:** in a 1-hour, ~1.7-pass budget the model is compute-limited, not data-limited. Dropout slows learning per step. It also cost about 7% throughput (mask generation and multiplies): 10.8k vs 11.6k tok/s, 9,470 vs 10,194 steps.
- **Gap caveat:** the train-batch loss is measured with dropout on, so the train-vs-val gap of the dropout run is not comparable as an overfitting measure. The val curves are the fair comparison.
- **Recommendation:** keep dropout=0 at this budget. If overfitting from the large embedding table matters later (longer runs, more passes), try targeted regularisation instead: weight decay on wte, or a lower dropout such as 0.02–0.05.


## #10 long constant-LR run: when does v16384 d256-L6 overfit? (nats/char)

Constant LR 1e-3 / Muon 0.01 after a 36 s warmup, no decay. Val-1M, full 10M val and recent train loss every 30 min of training time. 12 h wall cap; early stop if the last 3 full-val evals are all > best + 0.003. Chart: `10-long/chart.png`.

| eval | train h | wall h | step | passes | val 1M | full val | train (recent) | full - train |
|---|---|---|---|---|---|---|---|---|
|  | 0.50 | 0.58 | 5333 | 0.93 | 1.0089 | 1.1203 | 1.0745 | +0.046 |
|  | 1.00 | 1.17 | 10399 | 1.81 | 0.9664 | 1.0926 | 1.0343 | +0.058 |
|  | 1.50 | 1.75 | 15254 | 2.66 | 0.9633 | 1.0843 | 1.0117 | +0.073 |
|  | 2.00 | 2.33 | 20410 | 3.56 | 0.9573 | 1.0815 | 1.0022 | +0.079 |
|  | 2.50 | 2.92 | 25618 | 4.46 | 0.9568 | 1.0773 | 0.9899 | +0.087 |
|  | 3.00 | 3.50 | 30752 | 5.36 | 0.9549 | 1.0766 | 1.0035 | +0.073 |
|  | 3.50 | 4.09 | 35846 | 6.25 | 0.9466 | 1.0692 | 0.9994 | +0.070 |
|  | 4.00 | 4.67 | 40952 | 7.14 | 0.9463 | 1.0682 | 0.9911 | +0.077 |
|  | 4.50 | 5.25 | 45968 | 8.01 | 0.9518 | 1.0682 | 0.9940 | +0.074 |
|  | 5.00 | 5.84 | 50960 | 8.88 | 0.9474 | 1.0679 | 0.9864 | +0.082 |
|  | 5.50 | 6.43 | 55769 | 9.72 | 0.9447 | 1.0656 | 0.9859 | +0.080 |
|  | 6.00 | 7.01 | 60771 | 10.59 | 0.9474 | 1.0653 | 0.9754 | +0.090 |
|  | 6.50 | 7.60 | 65800 | 11.47 | 0.9446 | 1.0656 | 0.9788 | +0.087 |
|  | 7.00 | 8.19 | 70432 | 12.27 | 0.9416 | 1.0628 | 0.9808 | +0.082 |
|  | 7.50 | 8.77 | 74976 | 13.06 | 0.9438 | 1.0626 | 0.9825 | +0.080 |
|  | 8.00 | 9.36 | 79554 | 13.86 | 0.9400 | 1.0615 | 0.9793 | +0.082 |
|  | 8.50 | 9.95 | 83961 | 14.63 | 0.9445 | 1.0638 | 0.9872 | +0.077 |
|  | 9.00 | 10.54 | 88651 | 15.45 | 0.9431 | 1.0621 | 0.9891 | +0.073 |
|  | 9.50 | 11.13 | 93324 | 16.26 | 0.9416 | 1.0616 | 0.9758 | +0.086 |
| **best** | 10.00 | 11.72 | 97608 | 17.01 | 0.9376 | 1.0605 | 0.9812 | +0.079 |

Best full val 1.0605 at 10.00 training h, step 97608, 17.01 passes (checkpoint 10-long/v16384-const-ckpt-097608.f32).


### #10 interpretation

- **How it ended:** hit the 12 h wall cap (10.27 h of training, 1.73 h of evals, 100,128 steps, ~17.4 passes over the 23.5M train tokens). The early stop never fired, because full val never rose for 3 evals in a row.
- **Best point:** full val **1.0605** at 10.0 training h (step 97,608, 17.0 passes), the second-to-last eval. The very last eval at 10.27 h read 1.0635, which is within the eval-to-eval wobble of a constant-LR run (other bumps: 1.0615 to 1.0638 at 8.5 h).
- **No classic overfitting:** full val kept creeping down the whole run and never turned up. It fell fast for 2 h (1.1203 to 1.0815), then slowly: 1.0682 at 4 h, 1.0653 at 6 h, 1.0615 at 8 h, 1.0605 at 10 h (only about 0.002 per hour after 4 h).
- **Train-vs-val gap:** grew from +0.046 at 0.5 h to about +0.08 by 2 to 2.5 h (3.5 to 4.5 passes), then stayed flat in a +0.07 to +0.09 band for the rest of the run. So the memorisation gap opened during the first few passes and then stopped growing; train loss itself flattened near 0.98 after about 4 h. This looks like a model at its capacity at this LR, not one sliding into overfitting.
- **Versus the 1 h v16384 run (#8, full val 1.0480):** 10 h of constant LR still ends **0.0125 worse** than 1 h with WSD decay. The 1 h run had already beaten this run's 10 h number by the end of its decay phase. That points to the missing LR decay, not the extra data passes, as the main difference.
- **Hunch to test, not a finding:** a short WSD-style linear decay to 0 started from one of the late checkpoints (for example `v16384-const-ckpt-097608.f32`) would likely drop well below 1.0480, since the decay was worth a lot in the 1 h runs. That is the natural next test, one run at a time against the 1.0480 baseline.


## #11 short linear LR decay from the #10 checkpoint (nats/char)

Resumed 10-long/v16384-const-ckpt-097608.f32 (weights only; full val 1.0605) and decayed LR linearly from peak (Muon 0.01 / AdamW 1e-3) to 0 over 720 s of training time (= the 20% WSD decay of the 1 h runs). Chart: `11-decay/chart.png`.

| point | step | train s | val 1M | full val |
|---|---|---|---|---|
| start (#10 ckpt) | 0 | 0 | 0.9376 | 1.0605 |
| midpoint | 1018 | 360 | 0.9206 | 1.0388 |
| end | 1966 | 720 | 0.8982 | **1.0155** |

### #11 interpretation

Setup. This run resumed the #10 constant-LR checkpoint with the best full val (10-long/v16384-const-ckpt-097608.f32, full val 1.0605, step 97,608, 10 training hours) and decayed the learning rate linearly from its peak (Muon 0.01, AdamW 1e-3 for embeddings, norms and biases) to zero. The decay ran for 720 s of training time, with evals excluded. That matches the 1-hour WSD runs in #8, which used decay=0.2 of a 3600 s budget, so the same 12 minutes of decay. The config was otherwise identical: v16384, d256-L6, ctx 64, batch 64, 8 threads, f32 GELU, dropout 0. Data sampling used seed 11, so the decay batches do not replay the start of #10. gpt-eff.lisp gained two default-off options for this run: init= loads a raw f32 weight checkpoint, and sched=lin gives a linear decay from peak to 0 over the budget. Loading was checked first. At step 0 the loaded weights scored val-1M 0.9376, which matches #10's value at that checkpoint exactly (11-decay/smoke-init.log).

Optimizer state. The #10 checkpoints hold weights only. The writer in gpt-eff.lisp says so ("weights only; no optimizer state"), and the file size is exactly 8,949,760 parameters times 4 bytes. So the Adam moments and the Muon momentum restarted at zero. Adam's bias correction covers that cold start. The Muon momentum buffer rebuilds within a few dozen steps. Any cost from this is at most a small part of the 1,966 steps.

Result. Full val fell from 1.0605 to 1.0388 at the midpoint (step 1,018, 360 s, LR about half of peak) and to 1.0155 at the end (step 1,966). That is 0.0450 better than #10's best of 1.0605 and 0.0325 better than the 1-hour WSD baseline of 1.0480. Val-1M went from 0.9376 to 0.9206 at the midpoint and 0.8982 at the end. The gain was steady through the whole decay, with no flattening before LR reached zero.

Train/val gap. Train loss is the noisy last-batch value logged every 200 steps. Averaged over the second half of the decay (steps 1000 to 1800) it is 0.929, which puts full val minus train at about +0.086. At the midpoint, using the average of all logged points, the gap was +0.077. At the #10 checkpoint it was +0.079. So the decay lowered train and val loss by about the same amount, and the gap stayed in the +0.08 band that #10 held after its first few passes. The decay did not buy its gain by memorising more.

Cost. 720 s of training plus 262 s of in-run evals (the midpoint val-1M and full val) gives 982.5 s of wall time by the end of training. The final val-1M and full-val pass came after that. From queue start to queue end took 31.5 minutes (13:45:53 to 14:17:22 CDT).

Takeaway. The 10 hours of constant LR were not wasted. They were only unrealised. Constant LR followed by a 12-minute linear decay gives the best full val of the series so far, 1.0155. This run cannot say how much of the gain comes from the 10 extra hours rather than the decay alone. A matching test would be decays from earlier #10 checkpoints (for example 2 h or 5 h), to see how the decayed full val scales with training time.


## #12 the same 720 s linear decay started from earlier #10 checkpoints (nats/char)

Same settings as #11 (init= + sched=lin, peak to 0 over 720 s of training time, seed 11); final weights saved. Chart: `12-decay-from-earlier/chart.png`.

| decay start | #10 full val there | midpoint full val | final full val | vs 1 h WSD 1.0480 | full - train (end) |
|---|---|---|---|---|---|
| 2 h (step 20,410) | 1.0815 | 1.0578 | 1.0332 | -0.0148 | +0.083 |
| 5 h (step 50,960) | 1.0679 | 1.0440 | 1.0209 | -0.0271 | +0.081 |
| 10 h (step 97,608, #11) | 1.0605 | 1.0388 | **1.0155** | -0.0325 | +0.086 |

### #12 interpretation

Setup. This test repeated #11's 720 s linear LR decay to zero (Muon 0.01 and AdamW 1e-3 down to 0, seed 11, otherwise the #10 config) from two earlier #10 checkpoints. The checkpoints were step 20,410 (2.00 training hours, #10 full val 1.0815) and step 50,960 (5.00 h, #10 full val 1.0679). As in #11, the checkpoints hold weights only, so the optimizer state started from zero. The runs went one at a time through queue12.sh (2h, then 5h). Each wrote a midpoint checkpoint and, through the new savefinal=1 option, its final weights.

Results, in full val nats/char:
- Decay from 2 h: 1.0815 to 1.0578 at the midpoint, then 1.0332 at the end (1,868 steps).
- Decay from 5 h: 1.0679 to 1.0440 at the midpoint, then 1.0209 at the end (1,955 steps).
- Decay from 10 h (#11): 1.0605 to 1.0388 at the midpoint, then 1.0155 at the end (1,966 steps).

All three beat the 1-hour WSD baseline of 1.0480, by 0.015, 0.027 and 0.033. The decay itself was worth about 0.045 to 0.048 nats/char every time, roughly the same size wherever it started. The final result therefore tracks the constant-LR value it started from. Extra constant-LR training keeps paying off once it is decayed, with diminishing returns: 2 h to 5 h gained 0.012, and 5 h to 10 h gained another 0.005. The 2-hour run got about 100 fewer steps than the other two because the box was busier (2.59 vs 2.71 to 2.73 steps/s). That makes its result slightly pessimistic, but the effect is far smaller than the gaps between runs.

Train/val gap. Train loss here is the average of the logged last-batch losses over the second half of each decay (steps 1000 to 1800). Full val minus train came out at +0.083 for the 2 h start, +0.081 for 5 h and +0.086 for 10 h. #10 itself sat at +0.079 to +0.082 at those checkpoints. The decay lowers train and val loss together, and the later starts gain without a larger gap. The extra passes in #10 did not turn into overfitting that the decay exposes.

Takeaway. For this model, the best recipe found is a long constant-LR phase followed by a short linear decay to zero. The decay can be applied to any saved checkpoint, so one long constant run can give a decayed model at any budget for an extra 12 minutes of training plus evals. At the margin, 5 more hours of constant LR before the decay bought only 0.005.


## #16 decay-length sweep: linear decay 360 / 720 / 1440 / 2880 s from #10's 2 h checkpoint (nats/char)

**CAVEAT: unreliable as a decay-length comparison.** Budgets were wall-clock training seconds and the box was busy and froze during the runs, so step counts are unplanned (720 s got 1,262 steps vs 1,868 in #12's identical run). Redone with step budgets as #17. Chart: `16-decay-length/chart.png`. Notes: `16-decay-length/NOTES.md`.

| decay s | steps | steps/s | final full val | val 1M (end) |
|---|---|---|---|---|
| 360 | 916 | 2.54 | 1.0447 | 0.9232 |
| 720 | 1,262 | 1.75 | 1.0498 | 0.9268 |
| 1440 | 3,089 | 2.14 | 1.0298 | 0.9104 |
| 2880 | 8,182 | 2.84 | **1.0176** | 0.8983 |
| #12 720 s (ref) | 1,868 | 2.59 | 1.0332 | 0.9129 |

### #16 interpretation

Setup. queue16.sh, same args as queue12.sh 2h (seed 11, Muon 0.01 and AdamW 1e-3 linearly to zero, init 10-long/v16384-const-ckpt-020410.f32, full val 1.0815) except secs=, one run at a time, 5:44pm to 8:27pm CT Oct 21.

Results. Longer decays help: 1.0447 at 360 s down to 1.0176 at 2880 s, 0.0156 better than #12's 720 s decay and 0.030 better than the 1 h WSD baseline 1.0480. But the 720 s point (1.0498) is worse than 360 s and than #12 (1.0332) purely because it lost about 600 steps: it kept #12's pace to step 1,200, then box freezes at 6:06 to 6:12pm CT (heartbeat gaps of 240 s and 136 s) were counted as training time while the time-based LR ran down. Metric, seed, init and args match #12. The 2880 run exited rc=1 after writing all results (gpt-eff.lisp was edited for #17 mid-run).

Takeaway. Direction is clear (longer decay is better, at least up to 2880 s), magnitudes are not; use #17's step-budget numbers.

## #13 cyclic warm restarts: 5 x (2 h constant + 720 s linear decay) in one 11 h run (nats/char)

Interrupted by a box VM restart at 6.0 training h and resumed once from ckpt 056223 (weights only; optimizer moments reset). Chart: `13-cyclic/chart.png`.

| cycle end | training h | step | full val | branch decay (#11/#12) | vs 1 h WSD 1.0480 | full - train |
|---|---|---|---|---|---|---|
| 1 | 2.2 | 20,907 | 1.0317 | 2 h: 1.0332 | -0.0163 | +0.058 |
| 2 | 4.4 | 41,833 | 1.0227 | 5 h: 1.0209 | -0.0253 | +0.063 |
| 3 (after resume) | 6.6 | 61,500 | 1.0212 | | -0.0268 | +0.056 |
| 4 | 8.8 | 81,059 | 1.0164 | | -0.0316 | +0.067 |
| 5 (final) | 11.0 | 100,604 | **1.0152** | 10 h: 1.0155 | -0.0328 | +0.078 |

### #13 interpretation

Setup. One run of 11 training hours made of 5 cycles, each 2 h at constant LR (Muon 0.01, AdamW 1e-3) followed by a 720 s linear decay to zero, then a jump straight back to peak LR (warm restart). Otherwise the #10 config. It started at 11:32pm CT on Oct 20. A box VM restart killed it around 8:11 to 8:18am CT on Oct 21, at 6.0 training hours (last checkpoint step 56,223, t=21,600 s, in the constant phase of cycle 3). It was resumed once at 8:26am CT from that checkpoint (queue13r.sh, tag v16384-cyc-r1, toff=21,600.2 s, 18,000 s left) and finished at 3:21pm CT with rc=0. A first resume attempt at 8:22am failed instantly with rc=127 because sbcl was missing after the restart; it trained nothing. The checkpoint holds weights only, so Adam and Muon moments restarted from zero at 6.0 h. The original segment ran at 2.60 steps/s, the resumed one at 2.47 steps/s (busier box).

Results, full val at the end of each cycle's decay (training hours include all decays):
- Cycle 1 at 2.2 h: 1.0317, vs #12's decay from 2 h (also 2.2 h total) 1.0332.
- Cycle 2 at 4.4 h: 1.0227, vs #12's decay from 5 h (5.2 h total) 1.0209.
- Cycle 3 at 6.6 h: 1.0212 (first cycle end after the resume).
- Cycle 4 at 8.8 h: 1.0164.
- Cycle 5 at 11.0 h: **1.0152** (final), vs #11's decay from 10 h (10.2 h total) 1.0155.

All five beat the 1-hour WSD baseline of 1.0480, the last by 0.033. The cycle-end points sit on the same curve as the #12 branch decays: about equal at 2.2 h, and at the end 0.0003 better than the 10 h branch while using 0.8 more training hours. So warm restarts neither hurt nor help the final number. Each restart throws the decay's gain away within one eval (val goes from about 1.02 to 1.07 in the next 18 minutes), and the constant-LR stretches between decays track #10's constant run closely (for example 1.0684 at 4 h vs #10's 1.0682). Cycle 3 improved on cycle 2 by only 0.0015, less than the other cycles. That may be a cost of the resume resetting the optimizer moments, or just diminishing returns; this run can't separate the two (hunch, not tested).

Train/val gap. Here train is the logged recent-train loss at each eval, which is noisier than #12's averaged batch losses. Full val minus train at the cycle ends was +0.058, +0.063, +0.056, +0.067 and +0.078, and 0.07 to 0.11 in the constant phases. The gap at the cycle ends drifts up a little in the last two cycles, but val kept improving, so there is still no sign of overfitting at about 17.5 passes over the data (100,604 steps, about the same as #10).

Takeaway. Cyclic warm restarts give a decayed model every 2.2 h during one run at no extra cost, and reach the same quality as the branch-decay recipe (one long constant run, decay any checkpoint for 12 minutes). Branch decays stay slightly more flexible, since any checkpoint can be decayed later, but each costs an extra 12 minutes plus evals. Both land at about 1.015 nats/char after 10 to 11 training hours at this model size.


## #17 method upgrades: step budgets, a run ledger, cheap evals, a noise floor and a 2x rule, then tricks and one matched-compute final (nats/char)

Plan and full log: `17-method/PLAN.md`. Every run is one line in `17-method/results.tsv` (the ledger). Charts: `17-method/charts/decay-tricks.png`, `17-method/charts/proxy-tricks.png`, `17-method/charts/final.png`, `17-method/charts/summary.png` (regenerate with `17-method/charts.py final-10h-stack`).

| run | what | steps | full val | vs baseline mean | verdict |
|---|---|---|---|---|---|
| decay-base-s11 / s12 | branch decay from #10's 2 h ckpt, lin, seeds 11/12 | 1,868 | 1.0332 / 1.0336 | mean 1.0334 | control (bar 1.0326) |
| decay-sqrt | decay shape 1-sqrt instead of linear | 1,868 | 1.0314 | -0.0020 | keep |
| decay-elr3e-4 | AdamW (embeddings, norms, biases) lr 3e-4 instead of 1e-3 | 1,868 | 1.0316 | -0.0018 | keep |
| decay-mlr0.007b | Muon lr 0.007 instead of 0.01 | 1,868 | 1.0319 | -0.0015 | keep |
| decay-elr3e-3 | AdamW lr 3e-3 | 1,868 | 1.0414 | +0.0080 | discard |
| decay-mlr0.014b | Muon lr 0.014 | 1,868 | 1.0354 | +0.0020 | discard |
| decay-stack | sqrt + elr 3e-4 + mlr 0.007 together | 1,868 | 1.0318 | -0.0016 | not better than sqrt alone |
| proxy-base-s1 / s2 / s3 | 15-min proxy: WSD from scratch, seeds 1/2/3 | 2,500 | 1.1254 / 1.1248 / 1.1236 | mean 1.1246 | control (bar 1.1210) |
| proxy-qknorm | QK-norm, gain 2, seed 1 (vs proxy-base-s1 1.1254) | 2,500 | 1.1207 | -0.0039 (-0.0047 vs s1) | keep (2 seeds) |
| proxy-qknorm-s2 | QK-norm, gain 2, seed 2 (vs proxy-base-s2 1.1248) | 2,500 | 1.1202 | -0.0044 (-0.0046 vs s2) | keep (2 seeds) |
| proxy-relu2 | ReLU^2 instead of GELU | 2,500 | 1.1311 | +0.0065 | discard |
| proxy-zinit | zero-init output projections | 2,500 | 1.1255 | +0.0009 | discard |
| proxy-softcap | logit soft-cap 15 | 2,500 | 1.1268 | +0.0022 | discard |
| **final-10h-stack** | **sqrt decay from #10's 10 h ckpt (step 97,608), 100,604 total steps** | **2,996** | **1.0080** | **-0.0072 vs #13 1.0152** | **final** |

### #17 interpretation

What changed in method. #16 showed that wall-clock budgets are unreliable on this box: freezes were counted as training time and a 720 s decay lost a third of its steps. So #17 changed how experiments are run before testing anything new.
- Step budgets. gpt-eff.lisp gained steps=N (the LR schedule position is step/N, no time cap) and dsteps=N (decay over the last N steps), plus dshape=sqrt. Box pauses now only cost wall time, never steps. A 400-step check gave bitwise-equal final weights against the pre-method code, and the gradient check was unchanged (`17-method/bitwise/`, `17-method/gradcheck-step1.txt`). Training seconds for the same 1,868-step decay ranged from 735 s to 1,413 s across the ledger, with no effect on the loss.
- Ledger. `17-method/run.sh` freezes a copy of the code per run (`runs/<id>.lisp`, after #16's 2880 run crashed because I edited gpt-eff.lisp under it), does an 80-step speed probe, samples the load average, and appends one line to results.tsv with the code hash, seed, steps, speed, a SLOW flag, timings, val 1M, full val and a keep column (set later with `mark.sh`). A rerun of proxy-relu2 after a box restart matched the killed run exactly at step 1,400, so runs are deterministic given seed and steps.
- Cheap evals. In-run evals are now an occasional val-1M (val1m-steps=, 19 s each) and one full-val pass at the end only (about 190 to 240 s). The same 2 h branch decay spent 242.6 s in in-run evals in #12 and 31.6 s in #17's decay-base-s11, saving about 210 s (about 20% of the wall time) per decay run.
- Noise floor and the 2x rule. Before testing tricks I measured seed noise for both test beds: three 15-minute proxy runs (WSD from scratch, 2,500 steps) gave 1.1254 / 1.1248 / 1.1236 (spread 0.0018), and two branch decays from #10's 2 h checkpoint (1,868 steps) gave 1.0332 / 1.0336 (spread 0.0004). A trick is kept only if it beats the baseline mean minus 2x the spread: 1.0326 for decay tricks, 1.1210 for proxy tricks. A rank check against earlier 15-minute and 1-hour results (`17-method/rank_check.txt`) found the 15-minute proxy ranks changes much like the 1-hour runs for vocab (#7, Kendall tau +1.0; #8, +0.33 on only 3 pairs) and for training-dynamics changes (#1 to #4, +0.8), but not for model size (#6, tau -0.2), so the proxy is not trusted for capacity changes.

Tricks. Decay-side tricks were tested on the 2 h branch decay, architecture tricks on the proxy (they change the model, so they cannot be decayed from a checkpoint trained without them).
- Kept: sqrt-shaped decay 1.0314, AdamW lr 3e-4 during the decay 1.0316, Muon lr 0.007 1.0319, all under the 1.0326 bar by 0.0007 to 0.0012. QK-norm (gain 2) on the proxy: seed 1 scored 1.1207, only 0.0003 under the 1.1210 bar, so it got a second seed. Seed 2 scored 1.1202 against 1.1248 for the seed-2 baseline. Both seeds clear the bar, and each beats its own-seed baseline by about the same amount (-0.0047 and -0.0046), more than twice the 0.0018 seed spread. Two-seed verdict: keep. Val 1M agrees (0.9934 and 0.9930 vs 0.9969 to 0.9977 for the baselines). It is an architecture change, so it was not part of the final, which decays a checkpoint trained without it. Testing it needs a run from scratch.
- Discarded: AdamW lr 3e-3 (1.0414), Muon lr 0.014 (1.0354), ReLU^2 (1.1311), zero-init output projections (1.1255), logit soft-cap 15 (1.1268).
- Bug caught on the way. The first two Muon-lr runs (decay-mlr0.007/0.014) were no-ops: the base args already held mlr=0.01 and the option parser takes the first occurrence. Their curves were identical to decay-base-s11, which is how it showed. They are marked invalid in the ledger and were rerun as mlr0.007b/0.014b with the value replaced in place.

Stacking. The three kept decay tricks together (decay-stack) scored 1.0318. That clears the bar but does not beat sqrt alone (1.0314); the 0.0004 difference equals the decay seed spread, so it is a tie and the tricks do not add up. Hunch, not tested: all three shrink the effective step late in the decay, so stacking them double-counts. The queue picked sqrt-only for the final automatically (the run id still says "stack" because it was named before the pick; the frozen command in `runs/final-10h-stack.lisp` and queue.log shows dshape=sqrt only, Muon 0.01, AdamW 1e-3).

Final experiment. A sqrt-shaped branch decay from #10's 10 h checkpoint (step 97,608, full val 1.0605) for 2,996 steps, seed 11, ending at 100,604 total steps. That is exactly #13's 11 h run (100,604 steps, 1.0152), so the compute matches. Result: full val **1.0080** (val 1M 0.8914), **0.0072 better than #13's 1.0152 at matched compute**, and 0.0075 better than #11's plain linear decay from the same checkpoint (1.0155, 1,966 steps). It is also 0.040 better than the 1-hour WSD baseline of 1.0480. Val 1M went 0.9153 at step 1,000, 0.9001 at 2,000, 0.8914 at the end, still falling at the last eval. Wall time: 1,169 s of training plus 319 s of evals (1,488 s, about 25 minutes) on top of the existing 10 h checkpoint. #11's decay from the same checkpoint was 720 s of training. Training started 1:27am CT and was done at 1:54am CT Oct 24; the final full-val pass only finished at 5:50am CT because the box was paused from about 1:55 to 5:45am CT, which cost wall time but not steps. Final weights: `17-method/runs/final-10h-stack-final-002996.f32`. In `charts/final.png` the val-subset curves of #17's final, #11 and #13's last cycle decay show the final tracking #13's last cycle and sitting below #11 over the first 1,800 steps, even though its LR is still higher at each of those steps; it then keeps improving for another 1,000 steps after the other two have ended.

Caveats.
- The final is one seed. Its margin (0.0072) is 18 times the decay seed spread (0.0004), so it is very unlikely to be noise, but it was not repeated.
- The final changes two things at once compared with #13 and #11: the decay shape (sqrt instead of linear) and the decay length (2,996 steps instead of about 1,950 for #11 and for #13's last cycle). On the 2 h test bed the shape alone was worth 0.0020, and #16 pointed to longer decays helping, so most of the 0.0072 is probably the longer decay. This run cannot split the two.
- The box paused many times during #17 (roughly 11:06pm CT Oct 21 to 1:50am CT Oct 22, a VM restart around 11:27am CT Oct 23 that killed proxy-relu2 mid-run, and several more multi-hour pauses). Step budgets made these harmless for the losses, but they stretched #17 from the evening of Oct 21 to the morning of Oct 24 and make every wall-time number noisy. The SLOW flags in the ledger mark runs whose speed fell below 90% of their probe.
- The proxy is unreliable for capacity changes (rank check tau -0.2 for model size). QK-norm, ReLU^2 and soft-capping change the model's dynamics rather than its size, but a proxy keep is still only a lead until a full-length run confirms it.
- The decay-trick margins (0.0015 to 0.0020) are small against a two-seed spread; with only two decay seeds the spread itself is a rough estimate.

Takeaway. The method changes worked: step budgets made results immune to box pauses, the ledger caught a no-op bug, and the 2x rule sorted nine tricks into three small decay keeps plus one proxy keep (QK-norm, confirmed on two seeds) and five discards. The kept decay tricks do not stack. The best recipe found is a long constant-LR run followed by a longer, sqrt-shaped decay: 1.0080 nats/char at 100,604 steps, the best result of the series.


## #18 cheaper output head: sampled softmax with logQ correction (ssm=1024), 15-min proxy at matched compute (nats/char)

Method: training uses a sampled softmax over 1,024 negatives drawn from the unigram^0.75 distribution, with the logQ correction (flag `ssm=1024`, `ssm-alpha=0.75`; default `ssm=0` is the old full softmax). Evaluation is always the exact full softmax. Artifacts: `18-cheap-head/` (gradcheck.txt, probes1.tsv, queue18a.sh/.out), notes `18-head/NOTES.md`, ledger lines in `17-method/results.tsv`, logs `17-method/runs/e18-*.log`. Backup of the pre-#18 code: `gpt-eff-pre-18.lisp`.
Checks before the comparison: gradient check passes (ssm 1.2e-3/3.9e-4 vs full 4.4e-4/3.6e-4, fp32 finite-difference noise); a 400-step run with no flag (e18-default-400) gave val 1M 1.2055, full val 1.3236 and identical step losses, so the default path is bitwise unchanged.

Matched compute: each candidate got 2,500 x probe_tok_s(candidate) / probe_tok_s(v16 full) steps, so training seconds come out about equal to the 2,500-step proxy baseline. Throughput is reported as chars/s = tokens/s x train chars/token (v16384 3.82458, v32768 4.00758).

| run | vocab / head | steps | train s | tok/s | chars/s | chars trained | full val | vs proxy mean 1.1246 (bar 1.1210) | verdict |
|---|---|---|---|---|---|---|---|---|---|
| e18-proxy-base | v16384, full softmax | 2,500 | 1,174.1 | 8,722 | 33,358 | 39.2M | 1.1253 | +0.0007 (inside noise) | control |
| e18-v16-s1024 | v16384, ssm=1024 | 4,341 | ~1,251 clean (2,038 logged, ~760 s VM stall) | 14,213 clean (8,725 logged, SLOW) | 54,358 clean | 68.0M | **1.0924** | -0.0322 | keep |
| e18-v32-s1024 | v32768, ssm=1024 | 3,752 | 1,293.0 | 11,886 | 47,634 | 61.6M | **1.0902** | -0.0344 | keep |

How the v16 clean rate was computed. The log shows steps 1400 -> 1600 taking 815.1 s of training time against about 55 s for every other 200-step segment: the box/VM stalled for about 760 s, and the run's own timer counted it. Clean segments: steps 200 -> 1400 (59.9 -> 387.3 s, 327.4 s) plus 1800 -> 4200 (1288.6 -> 1998.7 s, 710.1 s) = 3,600 steps in 1,037.5 s = 3.470 steps/s x 4,096 tokens = 14,213 tok/s. The same-session 80-step probe gave 15,462 tok/s; the whole-run time minus the 760 s stall gives about 13,900. All three agree on roughly 1.6x the baseline. The step counts and losses are unaffected by the stall (step budgets).

### #18 interpretation (honest)
* **Throughput.** On v16384, sampled softmax raises chars/s 1.63x (33.4k -> 54.4k clean). On v32768 it is 1.43x the v16 full-softmax baseline (47.6k), and about 2.2x v32768 full softmax (probe 5.3k -> 13.2k tok/s): the bigger the vocab, the more the full head cost, and the more the sampled head saves.
* **Guardrail.** Both ssm runs land far under the 1.1210 bar: 1.0924 and 1.0902, 0.032-0.034 under the proxy mean. The proxy seed spread is 0.0018, so this is about 18x the spread, not noise. The quality loss from sampling the softmax at a fixed step count is more than paid back by the extra steps.
* **Compute match is approximate.** Steps were set from the probes, so the actual training time came out 6.6% (v16, clean) and 10.1% (v32) above the baseline's 1,174 s. That slack is small next to the gap in characters trained (1.6-1.7x), and I did not measure how much 10% more baseline time would buy, so a small part of the 0.03 gain may come from it. A baseline run at 1,293 s would settle it.
* **v16 vs v32 under ssm.** 1.0902 vs 1.0924 (0.0022) is about one seed spread and v32 used 3% more training time, so it is a tie. v16 is faster per char; v32 has more characters per token. Pick on the 1 h run, not on this.
* **Caveats.** One seed each. 15-min proxy only, and the proxy has been unreliable for capacity changes; the head change is not a capacity change, but a proxy keep is still a lead until the 1-hour confirm (vs #8's 1.0480). The v16 throughput depends on the clean-segment estimate because of the VM stall; its ledger line is flagged SLOW for that reason.
* **Verdict:** ssm=1024 beats the full-softmax baseline on chars trained per unit of compute (1.43x-1.63x) and on full val at matched compute, well outside noise. Next: the 1-hour confirm run, waiting on Matthew's OK.

### #18 Confirm (1-hour runs)

Queue `18-confirm/queue18c.sh` (approved by Matthew 2026-10-26 13:35 CT), v16384 d256-L6 ctx 64, batch 64 x 64 = 4,096 tokens/step, Muon + WSD, seed 1, same code as the proxy (gpt-eff.lisp sha256 6d37ad6a3ac0161d, default gemm=lisp512). Step budgets were set for 1 h of clean training: ssm=1024 12,960 steps (~3.6 steps/s from e18-v16-s1024's clean segments), full softmax 7,666 steps (2,500 / 1,174.1 s from e18-proxy-base). Both ran under `hardening/supervise.sh` with state checkpoints every 540 s, one segment each (no resumes). Run dirs `18-confirm/ssm` and `18-confirm/base` (exposed during the run as `hardening/runs.d/e18c-ssm` and `hardening/runs.d/e18c-base`; the queue removes those links when a run is done). Logs: `18-confirm/ssm/e18c-v16-s1024-1h.log`, `18-confirm/base/e18c-v16-full-1h.log`; `queue18c.out` ends with QUEUE18C_DONE. chars = tokens x 3.82458 train chars/token (v16384).

| run | head | steps | tokens / chars trained | train s logged | train s clean | tok/s clean | chars/s clean | val 1M (end) | full val | vs #8 1.0480 |
|---|---|---|---|---|---|---|---|---|---|---|
| #8 v16384 (reference) | full softmax, OpenBLAS GEMM, 1 h time budget | 10194 | 41.75M / 159.7M | ~3,600 | ~3,600 | 11,597 | 44,354 | 0.9257 (60 min) | 1.0480 | - |
| e18c-v16-s1024-1h | ssm=1024 | 12960 | 53.08M / 203.0M | 3,228.7 | 3,228.7 (no stalls) | 16,442 | 62,881 | 0.9244 | **1.0441** | -0.0039 |
| e18c-v16-full-1h | full softmax | 7666 | 31.40M / 120.1M | 6,546.0 (SLOW) | ~3,263 | 9,602 | 36,724 | 0.9355 | 1.0580 | +0.0100 |

FINAL lines: ssm `2026-10-26 14:40:03 CDT [supervise e18c-ssm] finished rc=0 FINAL: val(full fixed val, 2588640 targets scored, 9988883 chars)=1.0441 nats/char`; base `2026-10-26 18:22:42 CDT [supervise e18c-base] finished rc=0 FINAL: val(full fixed val, 2588640 targets scored, 9988883 chars)=1.0580 nats/char`.

How the clean rates were computed. The run's own timer counted box stalls as training time. The ssm log has none: every 200-step segment is 46-57 s (median 0.2475 s/step), so its logged 16,442 tok/s is the clean rate. The base log has 7 stalled segments (steps 200-400, 1400-1477, 3400-3456, 4000-4171, 4800-4859, 6200-6339, 7000-7071; 237 to 1,088 s each against a median of 0.417 s/step), about 3,283 s of excess in total, so its logged 4,797 tok/s (18,346 chars/s) is not a speed. The remaining 6,893 steps took 2,940.4 s = 2.344 steps/s = 9,602 tok/s; at that rate 7,666 steps are about 3,263 s. The two runs therefore got nearly the same clean training time (3,229 vs ~3,263 s). Wall clock: ssm 13:37 to 14:40 CT, base 14:41 to 18:22 CT (box paused); losses are unaffected (step budgets).

Ledger: neither run went through `17-method/run.sh`, so I appended their lines to `17-method/results.tsv` by hand in its format on 2026-10-26 (probe_tok_s and load_avg = na because no probe or load sampling was done; tok_s, train_s, wall_s, eval_s as the log reports them; base flagged SLOW for the stalls):

```
e18c-v16-s1024-1h	2026-10-26 14:40	6d37ad6a3ac0161d	1	12960	16442	na	ok	na	3228.7	3726.7	498.1	0.9244	1.0441	pending	/home/box/opt/sbcl/bin/sbcl --dynamic-space-size 4000 --script gpt-eff.lisp --run tokens=08-vocab-up/tokens16384.u16 vocab=08-vocab-up/vocab16384.hex vstart=23505859 fullval-targets=238949 d=256 layers=6 heads=8 ctx=64 batch=64 threads=8 dropout=0 lr=1e-3 optim=muon mlr=0.01 ew=f32 sched=wsd warm=0.01 decay=0.2 lrmin=0 steps=12960 seed=1 ssm=1024 ssm-alpha=0.75 ackpt=540 ackpt-keep=2 tag=/home/box/lisp-charmodel/efficiency/hardening/runs.d/e18c-ssm/e18c-v16-s1024-1h
e18c-v16-full-1h	2026-10-26 18:22	6d37ad6a3ac0161d	1	7666	4797	na	SLOW	na	6546.0	7782.5	1236.4	0.9355	1.0580	pending	/home/box/opt/sbcl/bin/sbcl --dynamic-space-size 4000 --script gpt-eff.lisp --run tokens=08-vocab-up/tokens16384.u16 vocab=08-vocab-up/vocab16384.hex vstart=23505859 fullval-targets=238949 d=256 layers=6 heads=8 ctx=64 batch=64 threads=8 dropout=0 lr=1e-3 optim=muon mlr=0.01 ew=f32 sched=wsd warm=0.01 decay=0.2 lrmin=0 steps=7666 seed=1 ackpt=540 ackpt-keep=2 tag=/home/box/lisp-charmodel/efficiency/hardening/runs.d/e18c-base/e18c-v16-full-1h
```

### #18 confirm interpretation (honest)
* **ssm=1024 vs full softmax at 1 h:** 1.0441 vs 1.0580, **-0.0139**, at the same clean training time. The sampled head trained 1.69x the characters (203.0M vs 120.1M) at 1.71x the clean chars/s (62.9k vs 36.7k). Val 1M agrees (0.9244 vs 0.9355, -0.0111), and the direction matches the 15-min proxy (-0.032).
* **ssm=1024 vs #8 (1.0480):** -0.0039. That is about 2x the 15-min proxy seed spread (0.0018), right at the edge of the 2x rule, and #8 differs in more than the head: it used OpenBLAS GEMM (11,597 tok/s) and a 1 h time budget, so it got 10,194 steps. Treat it as a tie or a weak lead, not a win. ssm=1024 matches #8's quality while training 1.27x #8's characters per clean hour on the slower pure-Lisp GEMM.
* **Why today's full-softmax baseline is worse than #8 (+0.0100):** it got 25% fewer steps (7,666 vs 10,194). The pure-Lisp GEMM runs at about 0.83x OpenBLAS (9,602 vs 11,597 tok/s, as expected from the 0.84x in all-lisp/AUDIT.md), and the step budget came from the proxy rate (2.13 steps/s), which is below today's clean 2.34 steps/s, so the run got ~54 min of clean training, not 60. It is not a replicate of #8 and says nothing about #8's seed noise.
* **Noise:** one seed each. REPORT.md has no seed repeats at the 1-hour length; the only measured spreads are 0.0018 (three 15-min proxy seeds) and 0.0004 (two 2 h branch decays). The 0.0139 gap is about 7.7x the proxy spread, so it passes the 2x rule, but that spread was not measured at 1 h; with one seed per arm, it is strong but not proven. A second seed of each arm would settle it.
* **Verdict:** at 1 h on this box, ssm=1024 beats the full-softmax baseline run under the same conditions (-0.0139, suggestive-to-strong, one seed each) and ties or slightly beats #8's 1.0480 (-0.0039, within noise). Adopt ssm=1024 as the default head for v16384, with a second seed as the cheap check.


## #5 lower-precision math: findings (honest)
* **Hardware:** Intel Xeon (Sapphire Rapids class) VM, 8 cores, 1 thread per core. Flags include f16c, avx512_bf16, avx512_fp16, avx512_vnni/avx_vnni, amx_bf16, amx_int8, amx_tile. Debian's OpenBLAS 0.3.29 has no sbgemm/shgemm (not built with BUILD_BFLOAT16). For testing I built OpenBLAS 0.3.30 locally (/home/box/opt, TARGET=SAPPHIRERAPIDS, and COOPERLAKE without AMX) and installed the MKL 2026.2 wheel in the venv.
* **Kernel benchmark** (single thread, worker shapes R=512 rows, d=256, V=2048; GFLOP/s; `05-precision/gemmbench*.txt`):
  * fp32 sgemm: OpenBLAS ~190-210, MKL ~200-225 (equal within noise).
  * bf16 on **AMX**: MKL `cblas_gemm_bf16bf16f32` 400-1000; OpenBLAS-SPR `sbgemm` 570-1200. That is **2.5-6x** fp32. Including the fp32->bf16 conversion of both inputs each call (AVX512-BF16 shim, `libgemmlp.so`): 330-625, i.e. 1.6-3x.
  * bf16 **without AMX** (AVX512-BF16 `vdpbf16ps`): MKL (forced AVX512_E3) ~95-120; OpenBLAS COOPERLAKE sbgemm ~95-120. Both **~0.55x of fp32, slower**.
  * fp16 (`cblas_gemm_f16f16f32`, AVX512-FP16; this CPU has no AMX-FP16): ~165/thread vs ~200 fp32. **Slower.**
  * int8 (AMX-int8/VNNI) was not pursued. It needs per-tensor quantization of activations and gradients, which is not a drop-in "fp32 master + accumulate" change for training, and AMX-int8 would hit the same problem as AMX-bf16 below.
* **AMX is unusable here: silent wrong results.** With 8 threads calling the AMX GEMM concurrently on identical inputs, about 1% of calls return garbage (max |diff| ≈ the magnitude of the outputs). This happens with MKL (15-33 per 2,400 calls in 4 trials) and with the independently written OpenBLAS SPR kernel (29+ per 2,400). Non-AMX paths gave 0 per 2,400 every time. In training this showed up as non-deterministic, sometimes much worse losses: bf16 runs with identical seeds gave 1.5267 / 1.5323 / 1.5237 / **2.0046** train loss at step 200 vs 1.4872 for fp32. The likely cause is the VM/hypervisor not preserving AMX tile state across context switches (dmesg shows "virtual machine fork" snapshots). This is a platform problem, not a code bug: the same shim with the non-AMX instruction set is exact and repeatable. The end-to-end bf16 speed would have been **+52-55%** (7.65-7.78 vs 4.96-5.03 steps/s), but the results cannot be trusted, so **no 1-hour bf16 run**.
* **Library control:** MKL fp32 in place of OpenBLAS fp32 gave the same speed (4.96 vs 5.03 steps/s; an earlier +31% reading was an artifact of the concurrent OpenBLAS build) and essentially the same curve.
* **Cheap alternative, same spirit (done):** the profile of the #4 step (`profile-fp32.txt`) shows GEMMs ~60% and the double-precision GELU forward+backward ~12% (GELU 9.7%, GELU-BACK 2.4%). I moved GELU forward/backward from double (4 lanes) to single precision (8 lanes, Cephes-style expf/tanh; the tanh cache is now single). Accuracy vs the double version on [-12,12]: max |ΔGELU| 4.8e-7, max |Δtanh| 1.3e-7. Speed: **+8%** (5.38 vs 4.94-5.03 steps/s over 1,000 steps). The 1,000-step loss curve matches fp32: identical to 4 decimals through step 400, then within 0.003 (step 1000: train 1.2683 vs 1.2688, val subset 1.1481 vs 1.1481). That is the size of normal trajectory noise; MKL-fp32 differs from OpenBLAS-fp32 by the same amount. The default (ew=f64) path is still bitwise identical. Softmaxes were left in double; the ~6% they account for (inside FORWARD-BACKWARD/ATTENTION) would be a further small win.
* **1 h run (#4 + f32 GELU):** val 1M **0.9595** and full val **1.0729** nats/char (1.548 bits/char), vs #4's 0.9612 / 1.0775 (-0.002 / -0.005). 17,742 steps vs 16,447 (+7.9%); 20,186 tokens/s = 59.5k chars/s vs 18,712 = 55.2k. Mid-run 1M val was 1.0736 / 1.0409 / 1.0248 vs #4's 1.0832 / 1.0447 / 1.0253.
* **Speed confound:** the 3-min fp32 #4-config probe just before the run gave 4.85 steps/s, while #4's own hour averaged 4.57. So the box was ~6% faster tonight than during #4, and only part of the +7.9% steps is the f32 GELU. The clean same-conditions measurement is the 1,000-step A/B (+8%). The quality gain (0.005 full val) is small and of the order of run-to-run noise.
* **Overfitting:** same picture as #4 (see table): train batch above the val subset, a small gap to the full val as the data repeats (~2.4 passes this time).
* **Recommendation:** keep the f32 GELU (ew=f32): a free ~8% with no measurable quality cost. **Do not use bf16/AMX on this box** until the VM preserves AMX state correctly. If it ever does, the measured end-to-end gain would be ~+52%.


## #4 BPE tokens: interpretation (honest)
* **Tokenizer:** BPE trained on the training split only (first 89,900,043 chars, the same boundary as the Lisp runs). It uses GPT-2-style chunking, so merges never cross word/space/punctuation-run boundaries; it is lossless (decode(encode(x)) == x, checked).
* **Vocab survey** (chars/token on 2M val chars; embedding params at d=256): 512: 2.34 (131k); 1024: 2.77 (262k); 2048: 3.13 (524k); 4096: 3.47 (1.05M). Estimated training compute per character, (6*4.72M block + 6*V*d head) / (chars/token): 512 2.07, 1024 1.80, **2048 1.67**, 4096 1.66 MFLOP-equiv/char. 2048 is the knee: 4096 doubles the embedding for no compute gain. Picked 2048 (2.95 chars/token on train, 3.00 on the full val, 3.26 on the first 1M val chars).
* **Model:** same width/depth (d256, L6, H8). ctx 64 tokens ≈ 190 chars, covering more than the old 128 chars. Eval uses the same protocol in tokens: stride 32, second half scored, so each scored token has 32-63 tokens (~95-190 chars) of context. 64 seqs x 64 = 4,096 tokens/step, the same tokens/step and per-step compute as before, so Muon's fixed per-step cost is amortized the same way. Each step covers ~12k chars instead of 4k. Params 5,279,744 vs 4,796,160 (+0.52M embedding, -16k positions).
* **Units:** val = total nats over the scored val tokens / characters those tokens cover. The scored targets are contiguous from the val start, so the char count is exact: 1M slice = 1,000,087 chars (the last window overruns by 87 chars, 0.009%), full = 9,988,863 chars (the char runs scored 9,988,864). The val text was tokenized as [first 1M chars] + [rest], so the slice boundary is exact.
* **Gradient check** (only the embedding/head size changed; the code is size-generic): single-float finite differences at V=5 and V=37 in gpt-eff.lisp (global rel. err 6.5e-5 / 4.2e-4, single-float noise level). Double-precision check of the reference maths (gpt-big, proven bitwise-equal to the fast code in Phase A) at V=37: max rel. err 1.4e-6 (dropout 0), 2.3e-6 (0.2), 1.1e-6 (0.5). The char path is still bitwise identical to before (400-step check).
* **15-min sweep:** Muon 0.01: **1.0289** nats/char (4,106 steps); 0.02: 1.0437 (4,272). For comparison, char-level Muon at 15 min scored 1.1317-1.1335, so the gain is about 0.10 at equal wall time.
* **1 h (mlr 0.01):** val 1M **0.9612** nats/char (1.387 bits/char) and full val **1.0775** (1.555 bits/char). vs #2: 1.0483 / 1.1419 (-0.087 / -0.064). vs baseline: 1.0721 / 1.1680 (-0.111 / -0.091). Already at 30 min (1.0447) it matches #2's final 1M number.
* **Speed:** 18,712 tokens/s = **55,197 chars/s** (x2.95), vs 19,743 tok(=char)/s for the 3-min #2 char-config speed check just before the run. Per step the speed is about the same (4.57 vs 4.82 steps/s, -5% from the bigger softmax/head), but each step covers 2.95x the text. 16,447 steps. Load ~7.9-8.0 throughout (our 8 threads plus the usual small node process); no outside heavy jobs.
* **Overfitting / data reuse:** 16,447 x 4,096 = 67M tokens = **2.2 passes** over the 30.5M train tokens. Over the last 10%, train batch 1.042 vs val subset 0.935, so the train batch is still above the easy val slice. But train batch vs full val (1.078) leaves a gap of 0.035 vs 0.007 for #2. A small generalization gap is opening as the data repeats, worth watching in #6 (bigger models or longer runs).
* **Recommendation:** **carry BPE-2048 forward**. It is by far the biggest win so far (-0.064 nats/char full val over #2), and it comes from ~2.9x more text per second at the same step cost.


## #3 short training context: interpretation (honest)
* **Setup:** start from #2 (Muon mlr 0.01 + WSD). Train at ctx 64 for the first part of the hour, then at ctx 128. wpe is sized 128 from the start. Eval is always the same ctx-128 protocol (windows of 128, the second half scored).
* **Batch choice:** the ctx-64 phase uses 64 seqs x 64 = 4,096 tokens/step, the same as 32 x 128. This keeps the gradient noise and the tuned AdamW/Muon LRs valid, so the only change is the context length. The model is identical: the forward/backward code already handles any length, and the worker buffers are reused (b x T rows).
* **posoff=1** (my addition) gives each worker's ctx-64 sequences a random position offset in 0..64 each step, so all 128 position embeddings keep training during the ctx-64 phase.
* **Gradient check:** the short-context path (workers built for T=8 running 2x as many length-4 sequences at position offset 3, with and without dropout) gives the same single-float finite-difference error as the standard path (global rel. err 1.1e-4 vs 9.1e-5). With ctx1=ctx, the code is bitwise identical to the pre-#3 file. Log: `03-ctx/gradcheck.log`.
* **15-min sweep (Muon + WSD, val 1M):** ctx-128 control 1.1335 (4,258 steps). ctx64->128 at 75%: 1.1515 (4,516). At 75% + posoff: 1.1444 (4,926). At 60% + posoff: 1.1506 (4,581). At 85% + posoff: 1.1680 (4,601). **Every short-context arm lost to the control**, even with 6-16% more steps. The best arm (sw0.75 + posoff) went to the 1-h run.
* **1 h (sw0.75 + posoff):** val 1M 1.0575 and full val **1.1490**. That is worse than #2 (1.0483 / 1.1419) and about equal to #1 (1.0506 / 1.1477); still better than the baseline (1.0721 / 1.1680). It had 18,796 steps vs 17,263 for #2. The 15/30/45-min points (1.3854 / 1.3502 / 1.3767) are not comparable with the other runs: the eval scores chars with 64-127 chars of context, which the model never used during the ctx-64 phase. After the switch at 45 min, val subset fell from ~1.35 to 1.09 within ~750 steps, but the remaining 25% could not fully recover.
* **Speed:** 21,385 tok/s overall vs 18,832 for the 3-min ctx-128 Muon speed check just before the run. So the ctx-64 phase is roughly 15-20% faster per token. Attention is only part of the cost at d=256, so halving the context buys only a little. Outside load was normal: load ~7.9, i.e. the 8 sbcl threads plus the usual ~15% node process; no other heavy jobs were seen. Box speed still drifted ±10% between runs over the evening (the ctx-128 control ran at 4.73 steps/s vs 5.21 for the same config at 20:19).
* **Overfitting:** none (see the table; train batch stays above val subset).
* **Recommendation:** **do not carry #3 forward.** The ~15% throughput gain does not pay for the loss in long-context quality at this scale and budget. Keep #2 (Muon + WSD, ctx 128) as the base for #4-#6.


## #2 Muon: interpretation (honest)
* **Setup:** Muon runs on the 24 block matrices (Wqkv, Wproj, Wfc, Wfc2 per layer): Nesterov momentum 0.95, 5 Newton-Schulz steps, scale sqrt(max(1, dout/din)), the same decoupled weight-decay shrink as AdamW, LR following the WSD multiplier. AdamW (lr 1e-3, WSD) still handles wte/tied head, wpe, LayerNorm and biases. Global grad clip as before.
* **Correctness:** on tiny matrices (6x10, 10x6, 8x8) the Lisp Muon step matches a numpy float64 reference of Keller Jordan's algorithm to 5e-7 after 2 steps. NS5 singular values come out at 0.68-1.12, as expected. With optim=adamw, the code is bitwise identical to the pre-Muon file (400-step check). Logs: `02-muon/muon_check.log`, `02-muon/muon_check.py`.
* **Muon LR sweep (15 min):** 0.01 gave 1.1317. 0.02 gave 1.1855: no spikes, just consistently higher loss, and fewer steps. 0.04 gave 2.354 (diverged after ~600 steps). The same-day AdamW+WSD control gave 1.1636. Muon 0.01 beat AdamW by 0.032 despite 11% fewer steps (4,694 vs 5,278). An extra point at 0.005 got 1.1337 (4,264 steps, 4.74 steps/s, a slower box), statistically a tie with 0.01 (1.1317, 4,694 steps). 0.01 was kept because it is not beaten, and it is carried into #3.
* **1 h (mlr 0.01):** val 1M 1.0483 (#1 1.0506, baseline 1.0721); full val **1.1419** (#1 1.1477, baseline 1.1680). That is better than #1 by 0.002 / 0.006 and better than the baseline by 0.024 / 0.026, with only 17,263 steps (#1 21,896, baseline 18,899). Mid-run, Muon leads clearly: at 15/30/45 min it is at 1.1999 / 1.1587 / 1.1336, vs #1 1.2296 / 1.1692 / 1.1518.
* **Speed (confound):** Muon ran at 19,641 tok/s; the AdamW speed probe just before it ran at 23,904. In the 15-min sweep, Muon's own overhead measured about 11% (5.21 vs 5.86 steps/s). The rest of the ~18% gap came from box load during the run: another agent's Chrome/Playwright desktop plus the sand-host node process at about 90% CPU, with load spiking to 14-15 around 22:10-22:17 CDT. Unstable runs were also slower (0.02: 4.53, 0.04: 2.49 steps/s), probably from saturated exp/tanh paths. So the 1-h Muon run was handicapped on steps and still won.
* **Overfitting:** none. Over the last 10% of the run, train batch 1.135 vs val subset 1.015, the same pattern as the baseline and #1.
* **Recommendation:** carry Muon (mlr 0.01) forward together with WSD. The gain over #1 at 1 h is small (0.006 full val) but comes with 21% fewer steps. The 15-min result (-0.032) shows Muon is much more sample-efficient. Muon may need a retune if the batch size or schedule changes.


## #1 interpretation (honest)
* **A higher peak LR did not help.** In the 15-min WSD sweep, val 1M got monotonically worse as the peak rose: 1e-3 1.1708, 2e-3 1.1789, 3e-3 1.1926, 5e-3 1.1951. An extra point below the range, 5e-4, got 1.1802 (5,426 steps), worse than 1e-3 even with 12% more steps. So 1e-3 is a true interior optimum: the default LR was already right.
* **So the real test is the WSD schedule vs cosine at the same lr 1e-3.** At 1 h, WSD finished better on both measures: val 1M 1.0506 vs 1.0721 (-0.0215) and full val 1.1477 vs 1.1680 (-0.0203). WSD is worse mid-run (1.1518 vs 1.1018 at 45 min) because the LR is still at its peak; all of its gain comes in the final 20% decay.
* **Confound:** the code is identical, but the box ran faster during the WSD run (24.9k vs 21.5k tok/s), so WSD got 21,896 steps vs 18,899 (+16%). Part of the 0.02 gain is from those extra steps.
* **Cleaner same-length check:** the 15-min sweep has WSD 1e-3 at 1.1708 with 4,830 steps vs cosine 1e-3 at 1.1725 with 5,588 steps. WSD is slightly better there despite 14% fewer steps, so it is genuinely at least as good per step, with a small edge.
* **Overfitting:** none. In the last 10% of each run, the train batch loss (about 1.16) stays well above the val subset (1.026), the same pattern as the baseline. The val slices are easier text than train.
* **Recommendation:** carry WSD forward (lr 1e-3, warm 0.01, decay 0.2, lrmin 0). It is at least as good and slightly better. It also gives a constant-LR phase, which makes later variants easy to compare and resume. The true gain is probably smaller than the 0.02 seen at 1 h.


### Sample (baseline)
```
(T=0.8) ===
 Now I have often seen my service to pass upon this woman in this
     reason, and must be used to influence the specific at eight.  I
      have seen the child of Lord Strap to the States of
       many fathers and other people of the State.  Not enough, between
      some pleasures of the century
```

### Sample (#1)
```
(T=0.8) ===
"I daresay not to say to the man, by distress who is regarded as a
power that has seemed to express and frequent pavilion and all the
power who, as the prize, by the chamber of his voice, and so penned and
said, whither the wounds knew, were still another nervous tenderness.


CHAPTER XX

To The Kin
```

## 2026-10-25: default GEMM is now pure Lisp (gemm=lisp512)
gpt-eff.lisp defaults to the AVX-512 pure-Lisp GEMM (all-lisp/lgemm512.lisp, custom SBCL VOPs). It trains at 10,545 tok/s, 0.84x OpenBLAS (12,515), with identical losses (400-step full val 1.3236). OpenBLAS stays selectable with gemm=openblas. Details are in all-lisp/AUDIT.md.

## #19 Speed: faster ssm=1024 training code (step 1 of 2)

Goal. Raise the plain training speed (tok/s, chars/s) of `gpt-eff.lisp` at the #18 Confirm settings (v16384, d256-L6, ctx 64, batch 64 = 4,096 tokens/step, Muon + WSD, ssm=1024 ssm-alpha=0.75, gemm=lisp512, threads=8) without changing any result. The #18 Confirm speed was 62,881 chars/s (16,442 tok/s). Everything is under `19-speed/` (running notes in `19-speed/STATE.md`). The original file is `gpt-eff.lisp.pre19` (sha256 6d37ad6a3ac0161d, same as the #18 code); the final diff is `19-speed/final.diff`.

Method. A 400-step bench (`19-speed/bench.sh`, seed 1, `savefinal=1`) on the same box, run one at a time under nice/ionice with no other jobs. Each candidate was interleaved with fresh baseline runs (ABBA, `19-speed/ab.sh`) because this VM drifts by 3 to 10% between runs. Speed is the clean segment (steps 200 to 400) and the whole-run rate. Correctness has three checks: the gradchecks (`19-speed/gradcheck.sh`: full softmax, ssm=3, ssm=12; outputs identical to `gradcheck-pre19.txt`), the step/loss lines, and a byte compare of the final weights against the baseline run (runs are deterministic, so bitwise equal means exactly the same training). All 28 runs are in `17-method/results.tsv` as `e19-*` (`tok_s` = clean segment, `probe_tok_s` = whole run).

Profile (main thread, ms/step before any change, about 235 ms total): trunk forward 62 (linear 40, attention 11, GELU 5, LN 4), trunk backward 93 (linear 72, attention 10, LN 5, GELU 3), sampled-softmax head about 14 (gather, scatter and softmax; the softmax itself 4.4), gradient reduce 26 (all in one thread: wte), optimizer update 34 (Muon Newton-Schulz GEMMs plus AdamW). Allocation is negligible (3 GB consed in 300 steps, GC 0.12 s). The GEMMs run at about 120 to 135 GF/s per core, near the AVX-512 ceiling.

| change | result | verdict |
|---|---|---|
| 1. spread the wte (V x d = 16 MB x 8 workers) gradient reduce and AdamW update over all 8 pool threads instead of one thread (`reduce-range`, `adam-range`, `wte-chunk`) | clean seg 16,525 to 19,074 tok/s (63.2k to 72.9k chars/s), +15.4%; whole run 15,249 to 17,892 tok/s (+17.3%). Reduce phase 26 to 8 ms/step. Bitwise identical weights (12/12 runs), gradcheck identical | **KEPT** |
| 2. pack-a with one 12-lane gather per two k-steps in `lgemm512.lisp` (candidate in `19-speed/lg/`, selected with GPT_LGEMM) | ABBA whole-run mean 70,307 vs 69,716 chars/s, +0.8%; bitwise identical (9/9) | rejected: inside box noise; shared GEMM file untouched |
| 3. Newton-Schulz Gram matrix computed only for the upper row blocks, then mirrored (`sym-gram`, `19-speed/gpt-eff.c3.lisp`) | ABBA whole-run mean 67,250 vs 66,872 chars/s (+0.6%), clean seg +2.2%; bitwise identical (4/4) | rejected: inside box noise |
| allocation/consing, sampled-softmax head | head is about 14 of 235 ms; consing negligible | no change (nothing to gain) |
| sparse wte reduce, pre-transposed W for dX, f32 attention softmax | estimated at most 1 to 3%, below what this box can resolve; f32 softmax changes numerics | not tried |

Result. Final code = pre19 + change 1 (sha256 ed0f085f64273d89). Clean speed about 72,950 chars/s against a fresh same-box baseline of about 63,200 chars/s (+15.4%); all 12 repeats of the final code average 17,852 tok/s whole-run (68,280 chars/s) against a baseline of 15,249 (58,320 chars/s). Training output is bitwise identical to the old code. Caveats: the box clock drifts, so the 15 to 17% has an error of a few percent; the baseline was measured on the same box in the same session (runs `e19-base-a/b/c` interleaved with `e19-c1-a/b`). Gains beyond this would need a faster GEMM kernel or less FLOPs, not overhead trimming. Step 2 (full-length test with this code, vs #8 1.0480) is not started.

## #19 Full test: one 1-hour ssm=1024 run with the #19 code (step 2 of 2) (nats/char)

Setup. Queue `19-full/queue19f.sh`, one run under `hardening/supervise.sh` (state checkpoints every 540 s wall, one segment, no resumes), run dir `19-full/ssm` (exposed during the run as `hardening/runs.d/e19f-ssm`; the queue removed the link at the end). Code: `gpt-eff.lisp` sha256 ed0f085f64273d89 = pre19 + change 1 from #19 Speed (bitwise-identical training to the #18 code, only faster). Settings are the #8 protocol: v16384 d256-L6 ctx 64, batch 64 x 64 = 4,096 tokens/step, seed 1, Muon (mlr 0.01) + AdamW lr 1e-3, WSD warm 0.01 decay 0.2 to 0, val 1M every 900 s of training time, final full val; head ssm=1024 ssm-alpha=0.75, gemm=lisp512.

**Deviation from #8: step budget `steps=15700` instead of #8's `secs=3600`.** 15,700 steps = 1 h at the 4.36 steps/s (17,852 tok/s) measured in the 12 e19-c1 benches. A step budget makes the run pause-tolerant: box stalls cost wall time but not steps, and the WSD schedule runs on step/15700 instead of on the clock. The consequence is that the run's logged training time (3,888.6 s) is above 3,600 s because the timer counted ~348 s of stalls, and the 15/30/45/60-min val-1M points are at logged training time, so the later ones fall a few hundred steps earlier in the schedule than a clean run would put them. Clean-equivalent training time was ~3,541 s (15,700 / 4.434 steps/s), about 59 minutes, so the compute matches #8's hour within ~2%.

| metric | value |
|---|---|
| full val (end) | **1.0388** (2,588,640 targets, 9,988,883 chars) |
| val 1M at 15 / 30 / 45 / 60 min (steps 3843 / 6939 / 10454 / 14453) + end (step 15700) | 1.0201 / 0.9950 / 0.9785 / 0.9404 / **0.9187** |
| steps | 15,700 |
| tokens / chars trained | 64.31M / 245.9M (x 3.82458 train chars/token) |
| train s logged / clean-equivalent | 3,888.6 / ~3,541 |
| tok/s, chars/s logged (whole run, includes stalls) | 16,537 / 63,248 |
| **tok/s, chars/s clean** | **18,162 / 69,462** (4.434 steps/s) |
| wall s (process timer) | 4,383.3 at end of training (+494.6 s in-run evals); 4,646.9 after the 243.4 s final full val |

How the clean rate was computed. Logged training time `t` excludes evals (val 1M and the final full val). Using the 79 logged 200-step points plus the end (step 15700, t=3888.6 s), the median segment is 0.2247 s/step. Eight segments are more than 1.3x the median and were dropped as stalls: steps 3200-3400 (72.6 s), 4200-4400 (61.5), 5400-5600 (151.2), 5600-5800 (68.9), 6800-7000 (98.3), 8000-8200 (118.6), 9400-9600 (74.2), 9600-9800 (62.6), about 348 s of excess over the median rate. The remaining 13,900 steps took 3,134.8 s = 4.434 steps/s x 4,096 = 18,162 tok/s x 3.82458 = 69,462 chars/s. Cross-check: (3,888.6 - 348.3) s for 15,700 steps gives the same 4.435 steps/s. This is +10.5% over #18 Confirm's clean 62,881 chars/s and 1.57x #8's 44,354, in line with #19 Speed's bench (+15% clean, ~72.9k chars/s) given this box's 3-10% run-to-run drift. Wall clock: started 2026-10-26 22:31:43 CDT, training ended about 05:58 CDT (final weights written), FINAL at 06:37:51 CDT Oct 27; the process timers saw only ~77 min, so the box was paused/suspended for most of the night. Losses are unaffected (step budget).

FINAL line: `2026-10-27 06:37:51 CDT [supervise e19f-ssm] finished rc=0 FINAL: val(full fixed val, 2588640 targets scored, 9988883 chars)=1.0388 nats/char`. Logs: `19-full/ssm/e19f-v16-s1024-1h.log` (training log), `supervisor.log`, `e19f-v16-s1024-1h.csv` (train/val-subset every 200 steps), `e19f-v16-s1024-1h-val1m.csv`, `e19f-v16-s1024-1h-final.csv`, `run.conf`, `DONE`; final weights `e19f-v16-s1024-1h-final-015700.f32`; queue output `19-full/queue19f.out` (QUEUE19F_DONE).

Ledger. The run did not go through `17-method/run.sh`, so its line was appended to `17-method/results.tsv` by hand on 2026-10-27 in the same format as the e18c lines (probe_tok_s and load_avg = na; tok_s, train_s, wall_s, eval_s as the log reports them; slow_flag ok because logged/clean = 0.91):

```
e19f-v16-s1024-1h	2026-10-27 06:37	ed0f085f64273d89	1	15700	16537	na	ok	na	3888.6	4383.3	494.6	0.9187	1.0388	pending	/home/box/opt/sbcl/bin/sbcl --dynamic-space-size 4000 --script gpt-eff.lisp --run tokens=08-vocab-up/tokens16384.u16 vocab=08-vocab-up/vocab16384.hex vstart=23505859 fullval-targets=238949 d=256 layers=6 heads=8 ctx=64 batch=64 threads=8 dropout=0 lr=1e-3 optim=muon mlr=0.01 ew=f32 sched=wsd warm=0.01 decay=0.2 lrmin=0 steps=15700 seed=1 ssm=1024 ssm-alpha=0.75 fullval-every=900 savefinal=1 ackpt=540 ackpt-keep=2 tag=/home/box/lisp-charmodel/efficiency/hardening/runs.d/e19f-ssm/e19f-v16-s1024-1h [19-full/queue19f.sh via hardening/supervise.sh, one segment; #19 step 2; tok_s=logged whole-run (includes ~348 s of box stalls); clean rate 18,162 tok/s = 69,462 chars/s over 13,900 unstalled steps; row appended by hand 2026-10-27]
```

Ranking by full val (lower is better). Duration/protocol is taken from the sections above.

| rank | run | full val | length | protocol (from this report) |
|---|---|---|---|---|
| 1 | #17 final-10h-stack | 1.0080 | longer (~10.3 h) | sqrt branch decay of 2,996 steps from #10's 10 h ckpt, 100,604 total steps |
| 2 | #13 cyclic | 1.0152 | longer (11 h) | 5 x (2 h constant + 720 s decay), 100,604 steps |
| 3 | #11 | 1.0155 | longer (10 h + 720 s) | 720 s linear decay from #10's 10 h ckpt |
| 4 | #12 / #16 | 1.0176-1.0447 | longer (2 h or 5 h ckpt + 360-2,880 s decay) | branch decays; #12 2 h 1.0332, 5 h 1.0209; #16 (time budgets, unreliable) 1.0447 / 1.0298 / 1.0176 (and 1.0498 at 720 s) |
| 5 | **#19 full (e19f-v16-s1024-1h)** | **1.0388** | **1 h** (15,700-step budget, ~3,541 s clean) | ssm=1024, #19 code, #8 protocol |
| 6 | #18 confirm ssm (e18c-v16-s1024-1h) | 1.0441 | 1 h (12,960-step budget, 3,228.7 s clean) | ssm=1024, #18 code |
| 7 | #8 v16384 | 1.0480 | 1 h (secs=3600) | full softmax, OpenBLAS GEMM, 10,194 steps |
| 8 | #18 confirm full softmax (e18c-v16-full-1h) | 1.0580 | 1 h (7,666-step budget, ~3,263 s clean) | full softmax, pure-Lisp GEMM |
| 9 | #10 long constant LR (best eval) | 1.0605 | longer (10 h constant, 12 h wall cap) | no decay |

#19 (1.0388) sits inside the #12/#16 band: it beats #16's 360 s decay from the 2 h checkpoint (1.0447) and #16's 720 s point (1.0498), but not #12's 2 h decay (1.0332) or the longer runs.

### #19 full interpretation (honest)
* **vs #8 (1.0480): -0.0092**, about 5.1x the 0.0018 15-min proxy seed spread (2x rule bar 0.0036), at about equal clean training time (~3,541 s vs 3,600 s). The gain comes from throughput: 245.9M chars trained vs 159.7M (1.54x), from the sampled head plus the #19 speedup, on the pure-Lisp GEMM.
* **vs #18 confirm ssm (1.0441): -0.0053**, about 2.9x the spread, past the 2x bar. Per-step training is bitwise identical to the #18 code, so the whole difference is more steps: 15,700 vs 12,960 (+21%). Part of that is the #19 speedup and part is that #18c's step budget gave only ~54 min of clean training (3,228.7 s) against ~59 min here, so this is "same code family, more compute in the hour", not a pure speedup effect.
* **Val 1M agrees:** 0.9187 at the end vs 0.9244 (#18c) and 0.9257 (#8 at 60 min).
* **Caveats:** one seed; the only measured seed spreads are 0.0018 (15-min proxy) and 0.0004 (2 h branch decays), none at 1 h. The step budget is the one protocol deviation from #8 (see above); the val-1M time points are not at the same schedule positions as #8's.
* **Verdict:** #19 full is the best 1-hour model of the series (1.0388), beating #8 by 0.0092 and #18 confirm ssm by 0.0053, both beyond 2x the ~0.0018 seed noise, though with one seed each.

## #20 Long run: the #19 fast code (ssm=1024) for #17's 100,604 steps (nats/char)

Setup. Queue `20-long/queue20.sh`, one run under `hardening/supervise.sh` (state checkpoints every 540 s wall, one segment, fresh start, no resumes), run dir `20-long/run` (exposed during the run as `hardening/runs.d/e20-long`). Code: `gpt-eff.lisp` sha256 ed0f085f64273d89, the same #19 code as #19 full. Settings: v16384 d256-L6 ctx 64, batch 64 x 64 = 4,096 tokens/step, seed 1, Muon (mlr 0.01) + AdamW lr 1e-3, head ssm=1024 ssm-alpha=0.75, gemm=lisp512, `steps=100604`, WSD with constant LR to step 97,608 then a 2,996-step sqrt decay to 0 (`dsteps=2996 dshape=sqrt`), the same decay length, shape and total steps as #17's final.

**Deviations from #17's final (honest):**
* **One WSD run from scratch instead of #17's branch.** #17 decayed from #10's 10 h constant-LR checkpoint (full softmax, older code) with seed 11 for the decay; #20 is a single run, seed 1, that does the constant phase and the decay in one go.
* **Warmup `warm=0.0016`** (about 161 steps of 100,604) instead of the 0.01 fraction used by the 1-hour runs, so the warmup length in steps is close to #19's (157 steps).
* **No mid-run full vals.** `fullval-every=900` was passed, but the log shows only val-1M points (29 of them) and one full val at the end, so there is no full-val curve to compare with #10's.
* **Different output head than #17.** #20 uses the sampled ssm=1024 head; #17's model was trained with the full softmax. This is the main difference in what is being compared.

| metric | value |
|---|---|
| full val (end) | **1.0138** (2,588,640 targets, 9,988,883 chars) |
| val 1M at end (step 100,604) | **0.8942** (step 99,222: 0.9056; step 95,190, last point before most of the decay: 0.9429) |
| val 1M before the decay | flat at 0.942 to 0.955 from step ~38,700 to 97,608 (best 0.9423 at step 84,377) |
| train batch / val subset at step 100,600 | 0.7042 / 0.8458 |
| steps | 100,604 |
| tokens / chars trained | 412.07M / 1,576.0M (x 3.82458 train chars/token), the same as #17 and #13 |
| train s logged / clean-equivalent | 25,515.1 / ~22,129 (about 6.1 h) |
| tok/s, chars/s logged (whole run, includes stalls) | 16,150 / 61,768 |
| **tok/s, chars/s clean** | **18,622 / 71,220** (4.546 steps/s) |
| wall s (process timer) | 28,849.0 at end of training (incl. 3,334.0 s of in-run evals: val subset every 200 steps plus val 1M), 29,082.2 after the final evals (19.8 s val 1M + 213.4 s full val) |
| calendar time | started 2026-10-27 07:26:23 CDT, FINAL at 21:56:07 CDT (about 14.5 h; the box slept about 11:53 to 14:52 CT and the run stalled, which cost wall time but no steps) |

How the clean rate was computed (same method as #19 full). Using the 503 logged 200-step segments plus the end, the median is 0.2195 s/step. 36 segments were more than 1.3x the median and were dropped as stalls (about 3,385 s of excess over the median rate). The remaining 93,204 steps took 20,501.1 s = 4.546 steps/s x 4,096 = 18,622 tok/s x 3.82458 = 71,220 chars/s, +2.5% over #19 full's clean 69,462 chars/s (within this box's 3-10% drift).

FINAL line: `2026-10-27 21:56:07 CDT [supervise e20-long] finished rc=0 FINAL: val(full fixed val, 2588640 targets scored, 9988883 chars)=1.0138 nats/char`. Logs: `20-long/run/e20-v16-s1024-100k.log` (training log), `supervisor.log`, `e20-v16-s1024-100k.csv` (train/val-subset every 200 steps), `e20-v16-s1024-100k-val1m.csv`, `e20-v16-s1024-100k-final.csv`, `run.conf`, `DONE`; final weights `e20-v16-s1024-100k-final-100604.f32`; state checkpoints at steps 97,608 and 99,616; queue output `20-long/queue20.out` (QUEUE20_DONE).

Ledger. `20-long/append-row.sh` appended the row to `17-method/results.tsv`. Checked against the log: steps, train_s 25,515.1, tok/s 16,150, val 1M 0.8942 and full val 1.0138 match. Two columns were wrong and were fixed by hand on 2026-10-27 (backup `17-method/results.tsv.bak-pre-e20fix`): wall_s was 52,217 (calendar seconds since run.conf was written, including the box sleep), now 28,849.0 (process timer at end of training, the convention of the #19 row); eval_s was 19.8 (the script's EVALTIME regex only caught the final val-1M line), now 3,334.0 (the in-run eval time the log reports).

Ranking by full val (lower is better).

| rank | run | full val | steps / length | protocol |
|---|---|---|---|---|
| 1 | #17 final-10h-stack | 1.0080 | 100,604 (~10.3 h) | full softmax; sqrt branch decay of 2,996 steps from #10's 10 h ckpt |
| 2 | **#20 long (e20-v16-s1024-100k)** | **1.0138** | **100,604 (~6.1 h clean)** | **ssm=1024, #19 code, one WSD run, 2,996-step sqrt decay** |
| 3 | #13 cyclic | 1.0152 | 100,604 (11 h) | full softmax; 5 x (2 h constant + 720 s decay) |
| 4 | #11 | 1.0155 | 10 h + 720 s | 720 s linear decay from #10's 10 h ckpt |
| 5 | #12 / #16 | 1.0176-1.0447 | 2 h or 5 h ckpt + decay | branch decays |
| 6 | #19 full (e19f-v16-s1024-1h) | 1.0388 | 15,700 (1 h) | ssm=1024, #19 code, #8 protocol |
| 7 | #18 confirm ssm | 1.0441 | 12,960 (1 h) | ssm=1024, #18 code |
| 8 | #8 v16384 | 1.0480 | 10,194 (1 h) | full softmax |
| 9 | #10 long constant LR (best eval) | 1.0605 | 10 h constant | no decay |

### #20 interpretation (honest)
* **vs #17 (1.0080): +0.0058, worse**, at the same 100,604 steps and the same 1,576M chars. Val 1M agrees (0.8942 vs 0.8914). The gap is 3.2x the 0.0018 proxy seed spread and 14x the 0.0004 decay spread, so it is unlikely to be noise. Hunch, not tested: the sampled ssm=1024 head costs a little quality per step, which was hidden at 1 h because the speedup bought more steps, but shows at matched steps. The seed (1 vs 11 for the decay), the warmup and the single-run setup also differ, so this run cannot pin it on the head.
* **vs #13 (1.0152): -0.0014, slightly better**, at matched steps. That is under the 2x bar for the proxy spread (0.0036) and there is no seed spread measured at this length, so call it a tie.
* **vs #19 full (1.0388): -0.0250**, from 6.4x more steps with the same code; the decay at the end did most of the late gain (val 1M 0.9429 at step 95,190 to 0.8942 at the end).
* **Speed:** the same 100,604 steps took about 6.1 h of clean training time, against about 10.3 h for #17 and 11 h for #13, so #20 gets within 0.0058 of the best result in roughly 60% of the compute time.
* **Overfit check:** train batch 0.704 vs val subset 0.846 at the end, and val 1M was flat (0.942 to 0.955) for the last ~60% of the constant phase, the same plateau #10 showed. Longer constant training at this size buys little; the decay is where the gain is.
* **Verdict:** #20 is the second-best model of the series (1.0138), beating #13 by a hair and #19 by 0.025, but it does not beat #17's 1.0080. One seed. The fast ssm code is the better deal per hour; the full-softmax recipe is still the better deal per step.

## #21 Full-softmax control: #20's exact run with the full softmax head (nats/char)

Setup. Queue `21-long/queue21.sh`, one run under `hardening/supervise.sh` (one segment, fresh start 2026-10-27 22:21:57 CDT, no resumes), run dir `21-long/run`. Same code (`gpt-eff.lisp` sha256 ed0f085f64273d89) and the same flags as #20 (v16384 d256-L6 ctx 64, batch 64, seed 1, Muon mlr 0.01 + AdamW lr 1e-3, gemm=lisp512, warm=0.0016, WSD constant to step 97,608 then 2,996-step sqrt decay, 100,604 steps), with one change: no `ssm=1024 ssm-alpha=0.75`, so the output head is the full softmax. The only difference from #20 is the head.

| metric | value |
|---|---|
| full val (end) | **1.0060** (2,588,640 targets, 9,988,883 chars) |
| val 1M at end (step 100,604) | **0.8895** (step 100,318: 0.8904; step 98,611: 0.9115; step 96,843, before the decay: 0.9431) |
| val 1M before the decay | flat at about 0.939 to 0.955 from step ~33,000 to 97,608 (best 0.9391 at step 89,386) |
| train batch / val subset at step 100,600 | 0.9180 / 0.8417 |
| steps | 100,604 |
| tokens / chars trained | 412.07M / 1,576.0M, the same as #20, #17 and #13 |
| train s logged / clean-equivalent | 46,981.9 / ~38,540 (about 10.7 h) |
| tok/s, chars/s logged (whole run, includes stalls) | 8,771 / 33,545 |
| **tok/s, chars/s clean** | **10,692 / 40,894** (2.610 steps/s) |
| wall s (process timer) | 50,437.2 after the final evals (24.1 s val 1M + 241.5 s full val); 50,171.6 at end of training |
| calendar time | started 2026-10-27 22:21:57 CDT, FINAL at 2026-10-28 12:55:54 CDT (about 14.6 h) |

Clean rate, same method as #19 full and #20: 504 logged segments (200 steps each plus the end), median 0.3788 s/step; 91 segments over 1.3x the median were dropped as stalls (about 8,522 s of excess); the remaining 82,404 steps took 31,567.1 s = 2.610 steps/s x 4,096 = 10,692 tok/s x 3.82458 = 40,894 chars/s. #20's ssm=1024 head ran at 71,220 clean chars/s, so the sampled head is 1.74x faster at this size. The run had many more stalls than #20 (91 vs 36 dropped segments), so the logged rate understates it more.

FINAL line: `2026-10-28 12:55:54 CDT [supervise e21-long] finished rc=0 FINAL: val(full fixed val, 2588640 targets scored, 9988883 chars)=1.0060 nats/char`. Logs: `21-long/run/e21-v16-full-100k.log`, `supervisor.log`, `e21-v16-full-100k.csv`, `e21-v16-full-100k-val1m.csv`, `e21-v16-full-100k-final.csv`, `run.conf`, `DONE`; final weights `e21-v16-full-100k-final-100604.f32`; state checkpoints at steps 99,365 and 100,306; queue output `21-long/queue21.out` (QUEUE21_DONE).

Ledger. `21-long/append-row.sh` appended row `e21-v16-full-100k` to `17-method/results.tsv`. Checked against the log on 2026-10-28: steps 100,604, train_s 46,981.9, tok/s 8,771, val 1M 0.8895, full val 1.0060, wall_s 50,437.2 (matches the log's "total wall so far") and eval_s 3,455.3 (= wall - train) all match; nothing needed fixing. Note the convention differs slightly from #20's hand-fixed row: #21's wall_s and eval_s include the 265.6 s of final evals (the row's note says so); on #20's convention they would be 50,171.6 and 3,189.7.

Ranking by full val (lower is better).

| rank | run | full val | steps / length | protocol |
|---|---|---|---|---|
| 1 | **#21 full-softmax control (e21-v16-full-100k)** | **1.0060** | **100,604 (~10.7 h clean)** | **full softmax, #19 code, one WSD run, 2,996-step sqrt decay, seed 1** |
| 2 | #17 final-10h-stack | 1.0080 | 100,604 (~10.3 h) | full softmax; sqrt branch decay of 2,996 steps from #10's 10 h ckpt, seed 11 decay |
| 3 | #20 long (e20-v16-s1024-100k) | 1.0138 | 100,604 (~6.1 h clean) | ssm=1024, otherwise identical to #21 |
| 4 | #13 cyclic | 1.0152 | 100,604 (11 h) | full softmax; 5 x (2 h constant + 720 s decay) |
| 5 | #11 | 1.0155 | 10 h + 720 s | 720 s linear decay from #10's 10 h ckpt |
| 6 | #19 full (e19f-v16-s1024-1h) | 1.0388 | 15,700 (1 h) | ssm=1024, #8 protocol |
| 7 | #8 v16384 | 1.0480 | 10,194 (1 h) | full softmax |

### #21 interpretation (honest)
* **vs #20 (1.0138): -0.0078, better**, with everything the same except the head. That is 4.3x the 0.0018 proxy seed spread, and val 1M agrees (0.8895 vs 0.8942). So **yes, #20's gap to #17 comes from the sampled head**: at matched steps the ssm=1024 head costs about 0.008 nats/char.
* **vs #17 (1.0080): -0.0020, a tie.** That is about 1.1x the seed spread, under the 2x bar. It does show that the other #20-vs-#17 differences (one run from scratch instead of a branch, seed 1, warm=0.0016) cost nothing; #21 is the new best number, but not a clear improvement over #17.
* **Speed trade:** #20 reached 1.0138 in ~6.1 h clean; #21 reached 1.0060 in ~10.7 h clean. Per step, full softmax wins; per hour, the sampled head is 1.74x faster, and #19 showed it wins at a fixed 1 h budget. Untested: whether ssm=1024 trained for ~10.7 h (about 175k steps) beats 1.0060.
* **Caveat:** one seed per arm. The 0.0018 noise figure comes from 15-min proxies, not from runs this long.
* **Verdict:** the head explains #20's gap. Use the full softmax when the step count is fixed, and the ssm=1024 head when wall time is fixed.


## #22a KD A/B: distillation from #21 at the 15-min proxy (nats/char)

Setup. Plan in `22-research/NOTES.md`. Teacher = #21's final weights (`21-long/run/e21-v16-full-100k-final-100604.f32`, full val 1.0060). `22-research/dump-teacher.lisp` (via `22-research/run-dump.sh`) ran the teacher over the train tokens [0, 23,505,859) and stored 16 sampled ids per position from the teacher's T=1 softmax (`22-research/teacher/kd16.u16`, 752 MB). New `gpt-eff.lisp` flags `kd=<file> kd-alpha=0.5`: loss = (1-a) x CE(true, ssm=1024) + a x sampled KD loss over the stored ids. Code sha256 de1236d17e4b3ee7 (backup of the pre-KD file: `gpt-eff.lisp.pre22`); with kd off, training is bitwise equal to the #19 code (`22-research/bitwise/result.txt`). Queue `22-research/q22.sh 22-research/ab-kd.list` under `hardening/supervise.sh`, ABBA order, #19 flags with `steps=3925`, seeds 1 and 2.

Teacher dump compute (must be counted, amortized over every KD run): **2,467.9 s wall, 16,673 s CPU** (about 4.6 CPU-hours) for 23,505,792 positions, teacher train NLL 0.8702 nats/char (`22-research/teacher/dump.log`, DUMP-DONE line, finished 2026-10-31 12:57:55 CDT). A first attempt died at start-up (`teacher/dump-failed-attempt1.log`, rc=1 at 12:09:55) before any output and is not counted.

Gradcheck (`22-research/gradcheck-kd.txt`): the KD loss was checked alone and combined with the sampled head (ssm=3, ssm=12), at kd-alpha 0.5 and 1.0, K=16 and K=2 (V=400 partial candidate set), dropout 0 and 0.2, normal and short-context buffers. Global relative error |g-num|/|num| was 7.5e-5 to 4.7e-4 in every case, the same range as the #19 sampled-head gradcheck (the "max rel err 1.000" entries are tiny-magnitude entries, as in earlier gradchecks). **Verdict: pass.**

| run | seed | full val | val 1M | clean chars/s |
|---|---|---|---|---|
| e22-base-s1 | 1 | 1.0982 | 0.9750 | 69,693 |
| e22-kd-s1 | 1 | **1.0953** | 0.9703 | 59,388 |
| e22-kd-s2 | 2 | **1.0906** | 0.9671 | 63,995 |
| e22-base-s2 | 2 | 1.0985 | 0.9761 | 73,547 |

Means: baseline 1.09835, KD 1.09295, **KD gain -0.0054 nats/char**, 3.0x the ~0.0018 seed spread (2x bar 0.0036). Per seed: -0.0029 (seed 1), -0.0079 (seed 2). Val 1M agrees (-0.0069 mean). Baseline seed spread here is 0.0003; KD seed spread is 0.0047, so the KD effect itself is noisier.

Ledger. All 4 rows are in `17-method/results.tsv` (appended by `22-research/append-row.sh`; backup `22-research/results.tsv.bak-pre-e22`), checked 2026-10-31 against the FINAL lines in `22-research/q22-ab-kd.out`: full val 1.0982 / 1.0953 / 1.0906 / 1.0985, steps 3925 each, code sha de1236d17e4b3ee7. The e20 row already reads wall_s 29082.3 and eval_s 3567.2 (fixed by hand, earlier).

### #22a interpretation (honest)
* **KD keeps** by the pre-registered rule (mean gain 0.0054 > 2 x 0.0018), at equal steps.
* **Cost caveat:** at equal steps KD trained about 14% slower (clean 59.4k-64.0k vs 69.7k-73.5k chars/s; the box's run-to-run drift is 3-10%, so part of this may be noise), and the teacher dump cost ~41 min wall once. A fixed-step win is not yet a fixed-time win; the 1-hour confirm uses the #19 full step budget (15,700) so it is directly comparable to e19f (1.0388) but will take somewhat more than an hour of clean time.
* **Caveats:** two seeds per arm; the seed-1 gain alone (0.0029) is under the bar; the teacher (1.0060) bounds how far the student can go.
* **Next:** 1-hour confirm `e22-kd-1h-s1` (steps=15700, seed 1, same flags plus kd) started 2026-10-31 15:09 CDT via `22-research/q22.sh 22-research/kd-confirm.list`, output `22-research/q22-kd-confirm.out`. After that, the modded-nanogpt tricks one at a time on the KD setup.

### #22a KD 1-hour confirm

Run `e22-kd-1h-s1`: the #19 full protocol (same flags as e19f, `steps=15700`, seed 1) plus `kd=22-research/teacher/kd16.u16 kd-alpha=0.5`, code sha de1236d17e4b3ee7, via `22-research/q22.sh 22-research/kd-confirm.list` under `hardening/supervise.sh` (one segment, fresh start, no resumes). Started 2026-10-31 15:09:01 CDT, FINAL at 16:33:40 CDT, rc=0. Run dir `22-research/runs/e22-kd-1h-s1/` (training log `e22-kd-1h-s1.log`, `supervisor.log`, `e22-kd-1h-s1.csv`, `-val1m.csv`, `-final.csv`, `DONE`, final weights `e22-kd-1h-s1-final-015700.f32`); queue output `22-research/q22-kd-confirm.out` (Q22_DONE).

FINAL line (`22-research/runs/e22-kd-1h-s1/supervisor.log`): `2026-10-31 16:33:40 CDT [supervise e22-kd-1h-s1] finished rc=0 FINAL: val(full fixed val, 2588640 targets scored, 9988883 chars)=1.0462 nats/char`

| metric | e19f (#19 full, no KD) | e22-kd-1h-s1 (KD) | KD - e19f |
|---|---|---|---|
| full val (end, step 15,700) | 1.0388 | **1.0462** | **+0.0074 (worse)** |
| val 1M (end) | 0.9187 | 0.9257 | +0.0070 |
| val 1M at ~step 6,935 | 0.9950 (step 6,939) | 0.9848 (step 6,931) | -0.0102 |
| val 1M at ~step 10,455 | 0.9785 (step 10,454) | 0.9705 (step 10,456) | -0.0080 |
| train_s logged (incl. stalls) | 3,888.6 | 4,236.7 | |
| tok/s logged (whole run) | 16,537 | 15,179 | |
| **clean chars/s** | **69,462** (REPORT #19; `clean.py` gives 69,423 over 13,800 steps) | **65,072** (4.154 steps/s = 17,014 tok/s over 14,200 unstalled steps, 6 of 77 segs dropped) | **-6.3%** |
| clean-equivalent training time for 15,700 steps | ~3,541 s | ~3,779 s (15,700 / 4.154) | +~240 s |
| eval_s / wall_s (ledger) | 494.6 / 4,383.3 | 636.3 / 4,873.0 | |

Clean rates come from `22-research/clean.py` (same method as #19 full / #20 / #21: 200-step logged segments of training time, segments > 1.3x the median s/step dropped as stalls) run on `19-full/ssm/e19f-v16-s1024-1h.log` and `22-research/runs/e22-kd-1h-s1/e22-kd-1h-s1.log`; the KD figure is also in the ledger row's cmd note. The KD run had several long stalls (e.g. steps 12,000-12,200 took 152 s of logged train time, 14,800-15,000 took 202 s), so its logged train_s overstates its cost; the clean rate is the fair number. So per unit time KD costs about 6% throughput at 1 hour (the proxy showed ~14%, partly box drift), on top of the one-off teacher dump (2,467.9 s wall, ~4.6 CPU-hours, `22-research/teacher/dump.log`). It is behind at equal steps and also behind at equal time.

Ledger. Row `e22-kd-1h-s1` in `17-method/results.tsv` checked against the log on 2026-10-31: full val 1.0462, val 1M 0.9257, steps 15700, train_s 4236.7, tok_s 15179 all match. Fixed: wall_s 4872.9 -> 4873.0 and eval_s 636.2 -> 636.3, to match the log's own summary (in-run eval 407.9 s from the `trained ...` line + `EVALTIME final val1m 18.8s` + `EVALTIME final full 209.6s` = 636.3; `total wall so far 4873.0s`). `append-row.sh` sums eval time only up to the last 200-step line and missed 0.1 s. Backup before the edit: `22-research/results.tsv.bak-pre-e22c`.

The curves (val subset = first 32,000 val targets, every 200 steps; `19-full/ssm/e19f-v16-s1024-1h.csv` vs `22-research/runs/e22-kd-1h-s1/e22-kd-1h-s1.csv`). KD leads the whole constant-LR phase: -0.039 at step 800, -0.012 at 3,800, -0.007 to -0.010 from 7,800 to 11,800. The WSD decay starts at step 12,560 (last 20%). The lead shrinks from -0.0076 at 12,600 to -0.0045 at 13,600 and -0.0006 at 14,200, then flips: +0.0021 at 14,400 and +0.0065 to +0.0074 from 15,000 to 15,600. The val-1M checkpoints agree: KD is ahead by 0.008-0.010 up to step ~10,450 and behind by 0.0070 at the end.

### #22a 1-hour confirm interpretation (honest)
* **Verdict: KD does not keep at 1 hour on one seed.** 1.0462 vs e19f's 1.0388 is +0.0074 worse, about 4x the 0.0018 proxy seed spread (no 1-hour seed spread has been measured). It is also ~6% slower per unit time and needs the teacher dump. #19 full (e19f, 1.0388) stays the best 1-hour model, and the next experiments go back to the #19 baseline without KD.
* **Proxy vs 1 hour disagree.** At the 15-min proxy (3,925 steps) KD won by 0.0054 mean (3.0x noise, `#22a KD A/B` above). At 15,700 steps it loses by 0.0074. The curves show why the proxy was fooled: the proxy ends at step 3,925, where KD was still far ahead on the val subset (about -0.01 at the 1-hour run's step 3,800), and its short decay (785 steps) did not have time to erase that lead. The 1-hour run's 3,140-step decay erased it and then reversed it.
* **Possible reasons (hunches only, not tested).** (1) The soft targets are a strong early-training regularizer (smoother targets, a head start on rare tokens), but in the low-LR decay the student is pulled toward the teacher's *sampled* 16-id distribution, and half the loss weight (kd-alpha 0.5) no longer goes to the true next token, which is what full val scores. A KD weight that anneals to 0 during the decay might keep the early gain. (2) With 16 samples per position, the KD gradient is noisy, and the noise matters most when the LR is small and the model is fine-tuning. (3) The teacher was trained 17.5 epochs on the same train split, so its train-split distribution is partly memorized (teacher train NLL 0.8702 vs its full val 1.0060). Matching it on train data may transfer worse to val than the hard labels do once the student gets close. (4) It is one seed, and the KD seed spread at the proxy was 0.0047, so part of the gap could be seed noise, though the sign flip in the curves looks systematic.
* **Lesson for the method.** A 15-min proxy that decays over 785 steps can rank changes that act mostly early in training (regularizers, auxiliary targets) wrongly. For those, compare at matched schedule position or run the confirm before stacking.
* **Next:** the modded-nanogpt tricks (NOTES.md idea 2), one at a time as 15-min proxy A/Bs on the #19 baseline, not KD. Queue in `22-research/q22-tricks.out`.

### #22b modded-nanogpt tricks: queued (2026-10-31 17:02 CT)
* Code: `gpt-eff.lisp` sha256 8be0dbaa…5cb (backup of the #22a code: `gpt-eff.lisp.pre22b`, sha256 de1236d1…2bb). New flags, all off by default: `untie=1` (untied zero-init head, also in the ssm gather/scatter), `ve=N` (N zero-init value-embedding tables added to V; layer l<N uses table l, layer l≥nl-N uses table l-(nl-N); no learned lambdas, a simplification), `unet=1` (decoder layer nl/2+i adds w_i·x[nl/2-i] to its input, w init 1), and `softcap=C` now also works with sampled softmax (cap applied before the logQ correction). Existing #17 flags reused: `relu2=1`, `qknorm=1 qkg=2`, `zinit=1`.
* Checks: flags off is bitwise identical to the old code (BASE flags, 200 steps: identical final weights and curves, `22-research/bitwise22b/result.txt`). Gradchecks for every new path, alone and combined, with and without ssm, global rel. error 1.5e-4 to 5e-3, same range as the baseline (`22-research/gradcheck-22b.txt`). All flags together at full size train without error (`22-research/bitwise22b/allflags.log`, 21.5M params).
* Queue: `22-research/tricks.list` → `22-research/q22-tricks.out`. First a recheck of the baseline on the new code (e22b-base-s1, which should reproduce 1.0982; its step 200 matches e22-base-s1 exactly), then each trick at seeds 1 and 2 (steps=3925). They are compared with e22-base-s1/s2 (1.0982/1.0985). A trick keeps if its mean gain is >0.0036 (2x noise). Earlier #17 results on the old full-softmax proxy: qknorm kept, relu2/zinit/softcap-15 discarded. These are re-tested on #19 because the head and setup changed.

## #22b Tricks: modded-nanogpt tricks one at a time on the #19 baseline (nats/char)

Setup. Code, flags and checks as in `### #22b modded-nanogpt tricks: queued` above (code sha256 8be0dbaa692053ad; plan `22-research/NOTES.md` idea 2). Queue `22-research/q22.sh 22-research/tricks.list` under `hardening/supervise.sh`, #19 flags with `steps=3925` (15-min proxy), each trick alone at seeds 1 and 2. Started 2026-10-31 17:02 CDT; last FINAL 2026-11-01 00:13:26 CDT; `Q22_DONE` in `22-research/q22-tricks.out`. All 15 runs rc=0, one segment each. Run dirs `22-research/runs/e22b-*/` (`supervisor.log`, training log, csvs). Baseline = the #22a controls e22-base-s1 1.0982 and e22-base-s2 1.0985, **mean 1.09835**; noise ~0.0018, so the pre-registered keep bar is a mean gain > 0.0036 (mean < 1.09475).

Baseline recheck on the #22b code. `e22b-base-s1` (baseline flags, seed 1, new code) gave **1.0983** vs e22-base-s1 **1.0982** on the #22a code (val 1M 0.9752 vs 0.9750). The flags-off check `22-research/bitwise22b/result.txt` says `BITWISE_EQUAL (2026-10-31): old gpt-eff.lisp.pre22b vs new gpt-eff.lisp (#22b flags off)`: 200 steps, final weights identical (sha256 cf7d075e… both), curves and val1m (1.3191) identical. Its step 200 also matches e22-base-s1 exactly. So the 0.0001 gap at 3,925 steps is the usual run-to-run float drift on 8 threads, well under noise, and not a code change. The tricks are compared with the #22a baseline pair as pre-registered (using e22b-base-s1 instead of e22-base-s1 would shift every delta by only +0.00005).

| trick (flags) | s1 full val | s2 full val | mean | Δ vs base 1.09835 | Δ per seed (s1 / s2) | clean chars/s s1 / s2 | mean chars/s (vs base 71,620) | verdict |
|---|---|---|---|---|---|---|---|---|
| relu2 (`relu2=1`) | 1.1038 | 1.1047 | 1.10425 | **+0.0059** | +0.0056 / +0.0062 | 65,175 / 68,718 | 66,947 (-6.5%) | no (worse) |
| qknorm (`qknorm=1 qkg=2`) | 1.0958 | 1.0955 | **1.09565** | **-0.0027** | -0.0024 / -0.0030 | 67,724 / 65,643 | 66,684 (-6.9%) | no (best, under the 0.0036 bar) |
| zinit (`zinit=1`) | 1.0988 | 1.0990 | 1.0989 | +0.00055 | +0.0006 / +0.0005 | 68,910 / 70,008 | 69,459 (-3.0%) | no |
| softcap (`softcap=30`) | 1.0989 | 1.0987 | 1.0988 | +0.00045 | +0.0007 / +0.0002 | 60,231 / 63,635 | 61,933 (-13.5%) | no (and slow eval) |
| ve (`ve=2`) | 1.0981 | 1.0977 | 1.0979 | -0.00045 | -0.0001 / -0.0008 | 69,280 / 66,846 | 68,063 (-5.0%) | no |
| unet (`unet=1`) | 1.0997 | 1.0989 | 1.0993 | +0.00095 | +0.0015 / +0.0004 | 72,704 / 68,198 | 70,451 (-1.6%) | no |
| untie (`untie=1`) | 1.1009 | 1.1008 | 1.10085 | +0.0025 | +0.0027 / +0.0023 | 65,877 / 72,614 | 69,246 (-3.3%) | no (worse) |
| baseline (#22a) | 1.0982 | 1.0985 | 1.09835 | 0 | | 69,693 / 73,547 | 71,620 | control |
| recheck e22b-base-s1 | 1.0983 | | | (+0.0001 vs s1) | | 71,491 | | control |

Clean chars/s are from the `clean ... chars/s` note in each ledger row's cmd column (`22-research/clean.py`: 200-step segments, segments > 1.3x the median s/step dropped as stalls). The three baseline runs span 69.7k-73.5k (about 5%), so speed differences under ~5% are box drift. Softcap's -13.5% is larger than that, and so is its eval cost: eval_s 1,539.8 / 1,401.7 s vs 313-334 s for the baselines. In `e22b-softcap-s1.log` the final full val took 1,053.2 s vs 209.4 s for e22b-base-s1, the final val 1M took 93.5 s vs 18.6 s, and the in-run evals took 393.1 s vs 85.9 s. That is about 5x on every eval: in eval the cap is a tanh on every one of the 16,384 full-softmax logits (likely the cost; not profiled). Wall_s for softcap was 2,644.4 / 2,390.4 s vs ~1,300 s for the others.

Ledger. All 15 e22b rows in `17-method/results.tsv` were checked on 2026-11-01 against the FINAL line in each `22-research/runs/<run_id>/supervisor.log` and in `22-research/q22-tricks.out`. Full val matches in all 15, so nothing needed fixing. The keep column of the 14 trick rows was changed from `pending` to `no` (e22b-base-s1 stays `control`). Backup before the edit: `22-research/results.tsv.bak-pre-e22b-verdict`.

### #22b interpretation (honest)
* **Verdict: no trick keeps.** None clears the pre-registered bar (mean gain > 0.0036). The #19 baseline stays as is.
* **qknorm is the closest:** -0.0027 mean (1.5x noise), and both seeds agree (-0.0024, -0.0030). It also kept in #17 on the old full-softmax proxy. So it is a plausible small win, but two seeds cannot confirm it. Confirming it would need more seeds (e.g. 3-4 per arm, or a 1-hour run vs e19f 1.0388). Those runs have not been started. It also ran ~7% slower in clean chars/s, which is partly box drift but would eat into any fixed-time gain. And the #22a KD lesson applies: a proxy win must survive the 1-hour schedule.
* **ve (-0.00045) is a tie.** zinit, softcap=30 and unet are ties on the worse side (+0.0005 to +0.0010, inside noise). Softcap also costs ~13% training speed and ~5x eval time in this code.
* **relu2 (+0.0059, 3.3x noise) and untie (+0.0025, both seeds) are worse.** relu2 was already discarded in #17. A possible reason (hunch, not tested): with the ssm=1024 sampled head, an untied zero-init head gets few updates per row in 3,925 steps, and it loses the shared embedding signal.
* **Caveats:** two seeds per arm. Each trick is tested alone at modded-nanogpt-style defaults with no tuning (qkg=2, softcap=30, ve=2 without learned lambdas, unet skip weights init 1), on a 3,925-step proxy. A trick that helps at GPT-2 scale or after tuning can be neutral on this small character model.
* **Next:** needs Matthew/parent decision. Options: confirm qknorm with more seeds or a 1-hour run, tune one trick, or move to NOTES.md ideas 3-5. No runs started.

## #23 Lisp BPE port

Common Lisp (SBCL, `/home/box/opt/sbcl`) port of `08-vocab-up/train_bpe.py` (HF `tokenizers` 0.23.3 BpeTrainer and BPE encode, V=16384): `23-lisp-bpe/bpe.lisp`. It is single-threaded. It reproduces HF's behaviour: Split(PAT, isolated) pretokenization on 1M-char training chunks, an alphabet sorted by code point, a merge heap ordered by (count desc, pair ids asc) with lazy count refresh, the HF `Word::merge` left-to-right change bookkeeping, min_frequency=2, and a stop at |vocab|>=V. Encoding merges the lowest-rank pair first (leftmost on ties) on the same ~1 MB newline-cut pieces. It matched on the first run, so no fixes were needed.

- **Merges:** all 16,289 merges (16,384 vocab minus the 95-char alphabet) are identical to the original `08-vocab-up/bpe16384.json` and to a fresh Python rerun (`cmp`). sha256 aef489c2…ad3. The vocab hex is also identical.
- **Tokens:** `23-lisp-bpe/out/tokens16384.u16` is byte-identical to `08-vocab-up/tokens16384.u16` and to the rerun (sha256 0967cae9…1fa9). The train slice (23,505,859 tokens) has sha256 60443969…db5d and the val slice (2,588,643 tokens) has 79ffd26e…d0e. Every `cmp` reports IDENTICAL.
- **Round-trip:** decoding the train and val tokens gives back exactly corpus[:89,900,043] and corpus[89,900,043:]. I checked this in Lisp and separately with numpy.
- **Wall time:** I ran both under `nice -n 10 ionice -c3 taskset -c 0-7`. Training (including the corpus read) took 4.66 s in Python (Rust, 8 threads) and 4.77 s in Lisp (1 thread). Encoding took 9.36 s in Python and 3.08 s in Lisp (1 thread, with a per-word cache).
- **Proof:** `23-lisp-bpe/proof/compare.txt`, `proof/roundtrip-numpy.txt`, `lisp.log`, `py-scratch/py.log`. The existing 08-vocab-up outputs were not touched.
