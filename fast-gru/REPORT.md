# Fast GRU in SBCL: speed report

Model: char-level GRU H=144, vocab 65, 100,577 params; B=32 streams, TBPTT T=32, Adam lr 2e-3, clip 1.
Data: Tiny Shakespeare, full-data fixed split (val = last 10%, 111,540 chars).
Code: `gru-fast.lisp`. Compile-time switches: `FT=double|single`, `MM=lisp|blas`. Runtime options: `threads=`, `blas-threads=`, `secs=`.
Every run used `nice -n 10`. Before the multi-core runs I checked `ps` and `uptime`. No science-team or Experimenter jobs were running, so the 8-thread runs used 8 cores. OpenBLAS was pinned to 1 thread per worker.
Speedup is relative to the earlier ablation GRU (s4) at **9.9 steps/s**.

| Run | Config | steps/s | chars/s | speedup | steps | val @480 s (full fixed val) |
|---|---|---|---|---|---|---|
| A | double, pure Lisp, 1 thread | 8.98 | 9,193 | 0.91x | 4,310 | 1.6649 |
| B | float32, pure Lisp, 1 thread | 9.42 | 9,649 | 0.95x | 4,523 | 1.6561 |
| C | double, OpenBLAS dgemm, 1 thread | 42.90 | 43,929 | 4.33x | 20,592 | 1.6135 |
| D | double, pure Lisp, 8 threads | 46.12 | 47,225 | 4.66x | 22,137 | 1.6167 |
| E | float32 + BLAS, 1 thread (extra run) | 51.92 | 53,171 | 5.24x | 24,924 | 1.6207 |
| F | float32 + BLAS + 8 threads (combined) | 178.73 | 183,019 | 18.05x | 85,800 | 1.6447 |
| G | same as F, 1800 s budget | 173.26 | 177,423 | 17.50x | 311,877 | about 1.535 (subset val at 479 s, see below) |

**G final (1800 s): val (full fixed val) = 1.6391.** Train (32x3000-char subset) = 1.2867.

Notes
- Steps/s and chars/s include time spent on periodic validation every 200 steps. For example, that was 31.7 s of F's 480 s and 118.8 s of G's 1800 s.
- G only does its full-val evaluation at the end. So for G's value at 480 s I used the periodic val-subset (first 32x1000 val chars): **1.5352 at step 83,800, t=479.0 s**. F's subset val at the same point was 1.5327 (step 85,600). The two runs track each other closely. G's best subset val was 1.5054 (step 111,600, t=637 s), and it ended at about 1.52–1.53.
- Speed and quality don't go together here. The combined run is 18x faster than baseline, but its val at 480 s (1.6447) is *worse* than C, D or E (1.61–1.62). Giving it 30 minutes (G, about 280 epochs-worth of chars) only reached 1.6391, while the train subset loss kept falling (1.32 to 1.29). The constant-LR 2e-3 GRU with H=144 has plateaued or is overfitting. The extra compute would need a larger model, an LR decay, or regularization to turn into lower val loss.
- A (double, pure Lisp, 1 thread) comes out slightly slower than the 9.9 steps/s ablation baseline. It is a restructured code path running on the same algorithm, and B shows that float32 alone helps little without BLAS.
- Gradient check (`gradcheck.log`). The check used 2 workers x 2 streams, so it also exercises the gradient summing across workers. It covered 157 params.
  - double, Lisp and BLAS: max rel err 4.38e-7, max abs err 2.8e-11, global |g-num|/|num| about 2.3e-10.
  - float32 (central diff, eps=1e-2): max rel err 0.137 (Lisp) / 0.140 (BLAS). Those are on near-zero entries; max abs err is 8.4e-6 / 9.9e-6 and global relative error is 2.2e-5 / 2.3e-5. So in float32 the gradients are correct up to float32 rounding in the finite differences. The double check is the strict test.
- Logs and CSVs: `A-base.*`, `B-f32.*`, `C-blas.*`, `D-threads8.*`, `E-blas-f32.*`, `F-combined.*`, `G-combined-30min.*`.
