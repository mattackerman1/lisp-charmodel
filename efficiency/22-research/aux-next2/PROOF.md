# Synthetic verification (not the ABBA)

`verify.sh` on SBCL 2.4.10 with `gemm=lisp` and a 20,000-character synthetic corpus printed `VERIFY_PASS`.

- 200-step final weights and optimizer-state bins: baseline, flag-off, and `n2=1 n2-alpha=0` are byte-identical. Full val on that toy run was 3.4717 (2000 chars).
- 40-step threads=2, alpha 0, matches the baseline weights.
- Gradcheck global error: alpha 0 matches the baseline file (worst 1.200e-3). alpha 0.1 worst 1.638e-3. alpha 1 worst 1.856e-3. Each figure is the worst of dropout 0, short context, and short context with dropout 0.2, at ssm 3 or 12 as in `aux-next2-proof.md`.
- `n2check`: label indices stay below `vstart`, causal `xf` rows do not see a flipped future id, and at alpha 1 with `U=0` only `dU` is nonzero.
- Resume of an enabled 40-step run from its step-20 state matches the uninterrupted final weights. The checkpoint stores `U`, its Adam moments, the aux RNG, and the primary sampled-softmax RNG.

The printed extra-FLOP line on that toy run is not a #19 measurement: vocab was 32, so candidate counts sat on the cap. No lisp512 run, no v16384 corpus, and no 3925-step training was done here.

## Finite-difference diagnosis (`mode=n2eps`, ssm=3, same tiny net as gradcheck)

Stock gradcheck uses central differences at eps=1e-2. At that step the primary-only run reproduces the baseline fingerprint exactly: global 1.200e-3, max abs 1.764e-3 on `wpe[9]`. Alpha 0.1 is global 1.572e-3 (max abs 2.849e-3 on `wte[92]`). Alpha 1 is global 1.856e-3 (max abs 3.953e-3, also `wte[92]`). The horizon-2 adapter `U` is not the source: its group global error at eps=1e-2 is 1.48e-5.

Shrinking the step to 3e-3 multiplies those max-abs errors by about 1/11, which is (0.01/0.003)^2. Past that, roundoff wins and the global error rises again. Best global residuals: primary 1.68e-4, alpha 0.1 2.09e-4, alpha 1 2.15e-4. A wrong analytic term would have stayed put as eps shrank. Symmetric directional derivatives at a unit-length step of 1e-3 agree to a relative 2.5e-4 on `U` and under 3e-3 on `wte`, `wpe`, and the trunk, at both alpha 0.1 and alpha 1. The `U` directional derivative at alpha 1 is 10× the alpha 0.1 value, which is the intended mix weight.

Acceptance in `n2-eps` is therefore: best global residual under 5e-4 (3× the measured primary floor) and at least a 4× drop from the eps=1e-2 value, plus the directional check. Logs: `grad-eps/`. The real-corpus 200-step bitwise check is `verify-corpus.sh`, which `q-abba.sh` requires. It has not been run in this environment.
