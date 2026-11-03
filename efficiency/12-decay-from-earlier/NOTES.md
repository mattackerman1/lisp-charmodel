### #12 interpretation

Setup. This test repeated #11's 720 s linear LR decay to zero (Muon 0.01 and AdamW 1e-3 down to 0, seed 11, otherwise the #10 config) from two earlier #10 checkpoints. The checkpoints were step 20,410 (2.00 training hours, #10 full val 1.0815) and step 50,960 (5.00 h, #10 full val 1.0679). As in #11, the checkpoints hold weights only, so the optimizer state started from zero. The runs went one at a time through queue12.sh (2h, then 5h). Each wrote a midpoint checkpoint and, through the new savefinal=1 option, its final weights.

Results, in full val nats/char:
- Decay from 2 h: 1.0815 to 1.0578 at the midpoint, then 1.0332 at the end (1,868 steps).
- Decay from 5 h: 1.0679 to 1.0440 at the midpoint, then 1.0209 at the end (1,955 steps).
- Decay from 10 h (#11): 1.0605 to 1.0388 at the midpoint, then 1.0155 at the end (1,966 steps).

All three beat the 1-hour WSD baseline of 1.0480, by 0.015, 0.027 and 0.033. The decay itself was worth about 0.045 to 0.048 nats/char every time, roughly the same size wherever it started. The final result therefore tracks the constant-LR value it started from. Extra constant-LR training keeps paying off once it is decayed, with diminishing returns: 2 h to 5 h gained 0.012, and 5 h to 10 h gained another 0.005. The 2-hour run got about 100 fewer steps than the other two because the box was busier (2.59 vs 2.71 to 2.73 steps/s). That makes its result slightly pessimistic, but the effect is far smaller than the gaps between runs.

Train/val gap. Train loss here is the average of the logged last-batch losses over the second half of each decay (steps 1000 to 1800). Full val minus train came out at +0.083 for the 2 h start, +0.081 for 5 h and +0.086 for 10 h. #10 itself sat at +0.079 to +0.082 at those checkpoints. The decay lowers train and val loss together, and the later starts gain without a larger gap. The extra passes in #10 did not turn into overfitting that the decay exposes.

Takeaway. For this model, the best recipe found is a long constant-LR phase followed by a short linear decay to zero. The decay can be applied to any saved checkpoint, so one long constant run can give a decayed model at any budget for an extra 12 minutes of training plus evals. At the margin, 5 more hours of constant LR before the decay bought only 0.005.
