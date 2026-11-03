# #22 research: can a 1-hour run reach ~1.006 nats/char? (notes, 2026-10-31, no runs launched)

## Facts from REPORT.md / 17-method/results.tsv
- #21 full softmax: 100,604 steps x 4,096 = 412.1M tokens (1,576M chars), full val 1.0060, 8,771 tok/s logged, ~13 h train_s (46,982 s).
- #20 same with ssm=1024: 1.0138 (ssm costs ~0.008 at long horizon).
- #19 1 h ssm=1024: 15,700 steps = 64.3M tokens, 1.0388, ~18.2k tok/s clean.
- Train split = vstart 23,505,859 tokens (~90M chars). 1 h run = ~2.7 epochs; #21 = ~17.5 epochs (heavily repeated data).
- Profile (#19, ~235 ms/step): trunk fwd 62, trunk bwd 93, head 14, reduce 26->8, optimizer 34 (Muon NS ~14% of step!). GEMMs 120-135 GF/s/core.

## Physical feasibility (estimates, 6N rule)
- Trunk non-embedding N ~ 12*L*d^2 = 12*6*256^2 = 4.72M; full head 256*16384 = 4.19M; ssm head ~256*1025 = 0.26M.
- FLOP/token: full ~6*8.9M = 53M (+~1M attention); ssm ~6*5.0M = 30M.
- #21 total ~412M * 54M = ~2.2e16 FLOP. #19 1 h ~64M * 31M = ~2.0e15 FLOP. Gap ~11x in FLOPs, 6.4x in tokens.
- Box ceiling: 8 cores * ~130 GF/s = ~1.0 TF/s fp32 -> ~3.7e15 FLOP/h. #19 achieves ~0.55 TF/s effective (~55%).
- => Even perfect fp32 utilization gives only ~1.8x. Matching #21 by brute compute is impossible in fp32. int8 VNNI (vpdpbusd, 4x int8 MACs per lane vs fp32 FMA) caps at ~4x on GEMMs theoretically, but int8 *training* (backward, Muon) is research-grade; forward-only int8 maybe 1.3-1.5x end to end. Not a path to 10x.
- So ~1.006 in 1 h needs ~5-10x better *loss per FLOP*, not speed. Honest odds: 1.006 exact is unlikely; 1.015-1.025 plausibly reachable by stacking ideas below.
- Note: the 10.7 h run itself is far past Chinchilla for a ~9M model on a 23.5M-token corpus (17.5 epochs); returns per compute are strongly diminishing there (Muennighoff et al.: repeated data loses value fast after ~4 epochs, https://arxiv.org/abs/2305.16264). That cuts both ways: a lot of #21's compute bought little, so the "true" gap at 1 h may be smaller than 11x.

## Ranked ideas
1. Distillation from #21 (teacher 1.0060). Precompute sparse teacher targets once over the 23.5M train tokens (teacher fwd ~2*9M*23.5M = ~4e14 FLOP = ~10-15 min of box; must be reported as compute, amortized). Store Random-Sampling KD targets (~12 sampled tokens/pos with importance weights; unbiased, unlike top-K which hurts at K<25): https://arxiv.org/abs/2503.16870 . Soft targets also replace the head cost (loss over ~12-64 stored ids + true token, like ssm). Distillation scaling laws (https://arxiv.org/abs/2502.08606) say KD helps most when student compute is small and a teacher already exists - exactly this case. Also kills multi-epoch overfitting signal (soft targets differ from memorized hard labels). Expected: biggest single lever; guess 0.01-0.025 at 1 h (uncertain). Cost: teacher dump script + new loss path + gradcheck, ~1 day; one 15-min proxy pair to test.
   Caveat: student can't beat teacher's distribution much; target 1.006 = teacher's own loss.
2. modded-nanogpt architecture pack (one at a time): ReLU^2, QK-norm, zero-init out projections, value embeddings (+gates), U-net skips, logit softcap 15-30, untied embed/head. Writeup reports speedups 45% (arch modernization incl ReLU^2/QK-norm/zero-init), 36% (skips+value embeds), 21% (softcap): https://github.com/KellerJordan/modded-nanogpt/discussions/211 , https://huggingface.co/blog/mishig/ten-ideas-from-the-nanogpt-speedrun , https://github.com/KellerJordan/modded-nanogpt/pull/186 . Note REPORT #17 had a proxy-relu2 run (killed by VM restart) - check status. Expected 0.005-0.015 total at 1 h; value embeds are cheap on CPU (gather + add). Cost: each a 15-min proxy pair.
3. Cut optimizer overhead: Muon NS = ~34 ms/235 (14%). Fewer NS iterations, or NS in lower precision is not available; update every step is needed, but NS on d256 matrices is small - batch them / 3 iterations. Plus remove 2nd gradient reduce passes. ~5-10% more steps = ~0.002-0.004. Cheap, bitwise-checkable path exists.
4. Shape/sizing + batch: at 64M tokens for 2.7 epochs, test smaller batch (32x64, more steps; Muon tolerates small batches) and ctx 128 (modded-nanogpt's long-seq record cut tokens 40%: https://www.tylerromero.com/posts/nanogpt-speedrun-worklog/). Also compare d320 L4 vs d256 L6 at equal FLOPs. Expected 0.003-0.01, uncertain sign.
5. Progressive growth / warm-start from a smaller model (e.g. train L3 then stack to L6, "stacking" https://arxiv.org/abs/2405.15319). Expected modest (published ~1.5-2x speedups at large scale). Higher implementation cost.
Also considered, lower rank: schedule-free (WSD already strong), SOAP (more optimizer FLOPs on CPU - bad), curriculum (little evidence at this scale), better tokenizer (16k BPE already tuned in #8), int8 VNNI (see above), low-rank/sparse (hurts quality at this tiny size).

## First experiment (proposal, not launched)
#22a distillation proxy, one change vs baseline:
- Step 0: write `22-research/dump-teacher.lisp`: load 21-long final weights, forward over train tokens [0, vstart), full softmax at T=1, for each position sample S=16 ids from teacher p (with replacement) and store ids (u16) + counts; store also true token. ~23.5M*16*2B = 750 MB. Log the teacher-dump wall time as compute.
- Step 1: gpt-eff.lisp flag `kd=<file> kd-alpha=0.5` : loss = (1-a)*CE(true, ssm=1024) + a*RS-KD loss over the stored ids (sampled softmax over union of stored ids + ssm negatives with logQ correction). Gradcheck at ssm=3/12 as in #19.
- Step 2: ABBA pair at 15-min proxy (steps=3925, same flags as e19f except steps; seeds 1 and 2), baseline vs kd. Keep if full-val gain > 2x seed noise (~0.0036). Then one 1 h run (steps=15700).
