# Hardening gpt-eff.lisp runs against box restarts (2026-10-21, Grok Bot)

Why: the Oct 21 ~8:15 CT VM restart wiped everything outside /home (sbcl, OpenBLAS, matplotlib), and #13's first resume died with rc=127.
Checkpoints held weights only, so every resume reset Adam/Muon. All new options are off by default: without them, final weights are
bitwise identical to `gpt-eff-pre-harden.lisp` (checked, 400-step Muon+dropout run).

## 1. Full-state checkpoints and exact resume (gpt-eff.lisp)
- `ackpt=SECS`: saves the full state every SECS of wall time. Use 600 for 10 min.
- `ackpt-steps=N`: saves every N steps instead (meant for tests).
- `ackpt-keep=2`: how many checkpoints to keep.
- `resume=<tag>-state-<step>.meta`: continues the run exactly.
- Files: `<tag>-state-<step 9 digits>.bin` (raw little-endian f32: weights, Adam m, Adam v, Muon momentum) and `.meta`, a readable plist.
  The plist holds step, total train seconds, Adam t, fnext, cycle index, the SBCL `*rs*` random state, each worker's dropout xorshift
  state, the hist/fhist/lhist series (so the CSVs keep their history) and argv. The .bin is written first and the .meta last, so the
  .meta marks a complete checkpoint.
- Writes are atomic: data goes to `*.tmp`, then fsync, then rename(2). Only the newest `ackpt-keep` pairs are kept. Time spent saving
  counts as eval time, so it does not use up the training budget.
- On resume, `secs=` stays the TOTAL budget. Train time continues from the saved value, so you no longer need `toff=` or `secs=` arithmetic.
  The schedule (lin/wsd/cosine/cyc) continues from the right position. `toff=` still works for the old weights-only `init=` resumes.
- Resume checks the float count (same model and optim) and the thread count (it must match: one RNG per worker).
- Exactness caveat: given the same steps, the result is bitwise identical. In real runs the LR schedule and full-val points depend on
  wall time, which no run reproduces exactly, so a resumed run is a faithful continuation rather than a replay.
- `GPT_OPENBLAS` env var overrides the OpenBLAS path (default unchanged: /usr/lib/...).

## 2. Toolchain under /home/box/opt (survives restarts)
- `/home/box/opt/sbcl/bin/sbcl`: wrapper around a copy of Debian SBCL 2.5.2 (binary, core and contribs including sb-simd).
- `/home/box/opt/openblas/lib/libopenblas.so.0`: the same 0.3.29 pthread build as before, so numerics are unchanged.
- `/home/box/opt/lib`: libgfortran5, libquadmath, libzstd.
- `/home/box/opt/pyenv`: venv with matplotlib 3.11.2 and numpy.
- The venv still uses the base image's /usr/bin/python3, which survived the restart.
- `hardening/env.sh` sets PATH, SBCL, GPT_OPENBLAS and PY.
- `queue13.sh` and `queue13r.sh` now source env.sh and run preflight. Originals are in `hardening/*.bak`.
- `hardening/preflight.sh [--full]` checks four things: sbcl runs, sb-simd loads, OpenBLAS loads and a 2x2 sgemm is correct, and
  matplotlib imports. `--full` also runs the 3 gradchecks. On failure it exits 1 with a `PREFLIGHT FAIL: <what>` line.
- To rebuild from scratch: `apt-get download sbcl libopenblas0-pthread` and unpack them into the same paths, or rerun the cp steps in this
  file's history. Then `python3 -m venv /home/box/opt/pyenv && pip install matplotlib numpy`.

## 3. Auto-resume
- `hardening/supervise.sh RUNDIR` starts or resumes one run. RUNDIR/run.conf sets TAG and ARGS, and optionally DYN, ACKPT (600), KEEP
  and NICE (10). See `hardening/run.conf.example` (run #13's args).
- The supervisor is idempotent:
  - exits if RUNDIR has a DONE or STUCK marker;
  - takes an flock so only one supervisor runs per RUNDIR;
  - exits if the tag is already running;
  - runs preflight first (rc 2 if it fails);
  - resumes from the newest `.meta`, or starts fresh if there is none;
  - appends each segment to RUNDIR/TAG.log with a header that includes the boot id;
  - writes DONE on success;
  - after 2 failures in a row from the same checkpoint, writes STUCK (team rule).
- `hardening/check-runs.sh`: for every run linked in `hardening/runs.d/`, launches a detached supervisor unless the run is DONE, STUCK or
  already running. It also records the boot id, so restarts show up in `hardening/check-runs.log`.
- **Nothing on this box runs automatically after a restart.** PID 1 is `tini`; there is no cron and no systemd. Agent shells are
  non-interactive, and ~/.bashrc and ~/.profile are re-provisioned at boot (they were dated 08:12 on Oct 21). So check-runs.sh **must be
  called periodically** from a routine (every 10–15 min is enough, and each call is cheap and a no-op when nothing is needed), or at the
  start of any agent session.

## Proof (logs in 14-hardening/, hardening/)
- Gradcheck (3 cases) output is identical to the pre-harden file. Global relative error: 6.5e-5, 7.7e-5, 7.2e-4. Muon numpy check: PASS (worst 5.1e-7).
- Resume test (14-hardening/test/): d128 L4, Muon, dropout 0.1, 2000 steps. Straight run vs. a run killed with -9 after its step-1000
  checkpoint and then resumed from it: the losses at every logged step match exactly. Step 2000: train 1.5750 / val 1.6134 in both;
  VAL1M 1.6493 in both. The final weights are **bitwise equal** (cmp).
- Preflight: `hardening/preflight-output.txt` (OK) and `hardening/preflight-fail-demo.txt` (missing OpenBLAS gives a clear FAIL, rc=1).
- Simulated restart (14-hardening/demo/): a 90 s cyclic run was launched by check-runs.sh. All of its processes were killed with -9 at
  step ~2400, check-runs.sh was rerun, and the run resumed from state 2215. It reproduced step 2400 exactly (1.5664 / 1.5826), passed
  CYCLE-END 2 at abs 80.0 s, finished at 90.0 s, and wrote DONE.
