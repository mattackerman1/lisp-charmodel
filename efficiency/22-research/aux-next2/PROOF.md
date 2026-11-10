# Synthetic verification (not the ABBA)

`verify.sh` on SBCL 2.4.10 with `gemm=lisp` and a 20,000-character synthetic corpus printed `VERIFY_PASS`.

- 200-step final weights and optimizer-state bins: baseline, flag-off, and `n2=1 n2-alpha=0` are byte-identical. Full val on that toy run was 3.4717 (2000 chars).
- 40-step threads=2, alpha 0, matches the baseline weights.
- Gradcheck global error: alpha 0 matches the baseline file (worst 1.200e-3). alpha 0.1 worst 1.638e-3. alpha 1 worst 1.856e-3. Each figure is the worst of dropout 0, short context, and short context with dropout 0.2, at ssm 3 or 12 as in `aux-next2-proof.md`.
- `n2check`: label indices stay below `vstart`, causal `xf` rows do not see a flipped future id, and at alpha 1 with `U=0` only `dU` is nonzero.
- Resume of an enabled 40-step run from its step-20 state matches the uninterrupted final weights. The checkpoint stores `U`, its Adam moments, the aux RNG, and the primary sampled-softmax RNG.

The printed extra-FLOP line on that toy run is not a #19 measurement: vocab was 32, so candidate counts sat on the cap. No lisp512 run, no v16384 corpus, and no 3925-step training was done here.
