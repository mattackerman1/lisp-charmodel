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
