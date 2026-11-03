### #9 interpretation

- **Dropout 0.1 hurts badly:** full val 1.0760 vs 1.0480 without dropout (+0.028 nats/char). That erases most of the vocab gain: the result is worse than v8192 (1.0582) and about level with #5 (1.0729). It is behind on val-1M at every checkpoint (0.9520 vs 0.9257 at 60 min).
- **Why:** in a 1-hour, ~1.7-pass budget the model is compute-limited, not data-limited. Dropout slows learning per step. It also cost about 7% throughput (mask generation and multiplies): 10.8k vs 11.6k tok/s, 9,470 vs 10,194 steps.
- **Gap caveat:** the train-batch loss is measured with dropout on, so the train-vs-val gap of the dropout run is not comparable as an overfitting measure. The val curves are the fair comparison.
- **Recommendation:** keep dropout=0 at this budget. If overfitting from the large embedding table matters later (longer runs, more passes), try targeted regularisation instead: weight decay on wte, or a lower dropout such as 0.02–0.05.
