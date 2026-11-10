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

## Correctness harness

```bash
cd $EFF
bash 22-research/aux-next2/verify.sh
```

On the box this selects `/home/box/opt/sbcl/bin/sbcl` and `gemm=lisp512`. It builds a 20,000-character synthetic corpus under `22-research/aux-next2/proof/` and does not read `tokens16384.u16`. Expect `VERIFY_PASS`.

## ABBA (not started here)

```bash
cd $EFF
bash 22-research/aux-next2/q-abba.sh
```

Order is base seed 1, aux seed 1, aux seed 2, base seed 2. 3925 steps. No 15700-step run. The queue refuses to start if the live `gpt-eff.lisp` hash changed. Working directory of each trainer is `$EFF`. The script path is the isolated lisp.

Keep iff mean full-val gain (baseline minus n2) is greater than 0.0036. Equal steps, not equal FLOPs. Read `N2 ACCOUNT` for realized candidate counts and the GEMM-flop estimate.
