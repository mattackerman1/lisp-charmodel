# Apply the horizon-2 head without touching the live baseline

Live file that must stay byte-for-byte in place:

`efficiency/gpt-eff.lisp`
sha256 `8be0dbaa692053ad6f7a125ff5880d3462100cdffb47c8a3132e54261627b5cb`

Do not edit `efficiency/22-research/q22.sh` or `efficiency/hardening/supervise.sh`. Those hardcode the live script. This directory has its own `supervise.sh` and `q-abba.sh`.

## Install into the non-git training tree

From the repo (or from a patch that only adds this directory):

```bash
EFF=/home/box/lisp-charmodel/efficiency
# creates 22-research/aux-next2/; does not write $EFF/gpt-eff.lisp
rsync -a --exclude proof efficiency/22-research/aux-next2/ "$EFF/22-research/aux-next2/"
sha256sum "$EFF/gpt-eff.lisp"
# must print 8be0dbaa692053ad6f7a125ff5880d3462100cdffb47c8a3132e54261627b5cb
sha256sum "$EFF/22-research/aux-next2/gpt-eff.lisp"
```

If you apply the git patch instead of rsync, run it from the training tree with `patch -p1` or `git apply`. The patch adds files only.

## Synthetic harness

```bash
cd $EFF
bash 22-research/aux-next2/verify.sh
```

On the box this selects `/home/box/opt/sbcl/bin/sbcl` and `gemm=lisp512`. It does not read `tokens16384.u16`. Gradient acceptance is `mode=n2eps`: the central-difference residual must shrink at least 4× from step 1e-2 and land under 5e-4, and each parameter group's symmetric directional derivative at step 1e-3 must agree within 1e-2. That 5e-4 is the converged floor (3× the measured primary-only floor of 1.68e-4), not a loosening of the stock eps=1e-2 line.

## Real-corpus gate (required before ABBA)

```bash
cd $EFF
bash 22-research/aux-next2/verify-corpus.sh
```

This runs live `gpt-eff.lisp` against the isolated file, flag-off and `n2=1 n2-alpha=0`, at v16384 d256 L6 ctx 64 batch 64 ssm 1024 lisp512 seed 1 for 200 steps, with the fixed full val. It also checks labels, the causal mask, and xf leakage at that shape, and an enabled alpha-0.1 resume. It writes `22-research/aux-next2/proof/corpus200/VERIFIED` only if every cmp passes. Expect several minutes because the full val runs three times.

## ABBA (not started here)

```bash
cd $EFF
bash 22-research/aux-next2/q-abba.sh
```

The queue refuses to start without a `VERIFIED` marker whose hashes match the current live baseline and the current isolated lisp. Order is base seed 1, aux seed 1, aux seed 2, base seed 2. 3925 steps. No 15700-step run. Working directory of each trainer is `$EFF`. The script path is the isolated lisp.

Keep iff mean full-val gain (baseline minus n2) is greater than 0.0036. Equal steps, not equal FLOPs. Read `N2 ACCOUNT` for realized candidate counts and the GEMM-flop estimate.
