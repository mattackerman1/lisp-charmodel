# #16 decay-length sweep (linear LR decay 360 / 720 / 1440 / 2880 s from #10 2h ckpt 020410)

> **CAVEAT: these numbers are unreliable as a decay-length comparison.** Budgets were wall-clock training seconds
> (`secs=`, measured with get-internal-real-time minus eval), and the box was busy and froze during the runs, so each
> run got a different, unplanned number of steps (916 / 1,262 / 3,089 / 8,182). The 720 s run got 1,262 steps vs
> 1,868 in #12's identical config. The sweep is being redone with step budgets as **#17**.

Setup: queue16.sh, args identical to queue12.sh 2h except `secs=` (seed 11, Muon mlr 0.01 + AdamW 1e-3, sched=lin to 0,
init=10-long/v16384-const-ckpt-020410.f32 (#10 full val 1.0815), vstart=23505859, fullval-targets=238949,
fullval-every=360 periodic-full=1). One run at a time. Metric: full fixed val (2,588,640 targets, 9,988,883 chars),
nats/char; the csv column is named `val_full10pct` but holds this same full-val number (it equals the log's FINAL line).

| decay s | steps | steps/s | final full val | val 1M (end) | #12-pace steps (2.59/s) |
|---|---|---|---|---|---|
| 360  |   916 | 2.54 | 1.0447 | 0.9232 |   932 |
| 720  | 1,262 | 1.75 | 1.0498 | 0.9268 | 1,865 |
| 1440 | 3,089 | 2.14 | 1.0298 | 0.9104 | 3,730 |
| 2880 | 8,182 | 2.84 | **1.0176** | 0.8983 | 7,460 |
| #12 720 (ref) | 1,868 | 2.59 | 1.0332 | 0.9129 | |

Conclusion: longer decays help (1.0447 -> 1.0176 from 360 s to 2880 s), but the step counts are confounded with box
load/freezes, so the size of each gain (and the 720 s point, which is worse than 360 s) cannot be trusted. See #17.

720 vs #12 (1.0498 vs 1.0332): not a metric, seed, init or arg difference. Both scripts pass the same args (queue16.sh
comment: "Same args as queue12.sh 2h; only secs= changes"); both logs load 10-long/v16384-const-ckpt-020410.f32 and
both FINAL lines are full val on 2,588,640 targets. The 720 run kept #12's pace to step 1200 (t=433.5 s vs #12 t=462.1 s)
then did only 62 steps in the last 287 "training" seconds: heartbeat.log shows the box frozen 18:06:14-18:10:14 and
18:10:14-18:12:30 CT, which the real-time clock counted as training while the time-based LR decayed. Result: 1,262 vs
1,868 steps (7,175 vs 10,621 tok/s). The midpoint full val agreed with #12 (1.0560 at step 993 vs 1.0578 at step 909).
Gap = lost steps, not a real gap.

Side note: the 2880 run printed FINAL, saved weights and the sample, then exited rc=1 with "The variable CONTEXT is
unbound" because gpt-eff.lisp was edited for #17 at 20:09 CT while SBCL --script was still reading it form by form.
Results were written before this; harmless here, but launch runs from a frozen copy of the script.

Files: chart.png, v16384-decay-2h-{360,720,1440,2880}{.log,.csv,-final.csv,-val1m.csv}, final weights *-final-*.f32.
