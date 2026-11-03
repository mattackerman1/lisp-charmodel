### #6 interpretation

- **No size beats #5.** d256-L8 (6.9M) ties #5 (d256-L6, 5.3M) on full val (1.0729 vs 1.0729). On val-1M at 60 min it is slightly ahead (0.9574 vs 0.9595), but that gap is within the run-to-run noise we've seen. The smaller d192 (+0.010) and the larger d384 (+0.007) are both worse on full val.
- **Compute is the limit, not overfitting.** Moving from d192 to d384 cuts the passes over the data from 3.8 to 1.1, and the train-vs-full-val gap only shrinks from 0.024 to 0.021 (0.029 for #5). With data repetition this mild, the larger models don't gain from seeing fresh data, and they lose too many steps (d384: 8.4k steps vs 17.7k).
- **Speed confound:** the probes at the #5 config before each run measured 20.8k–21.8k tok/s, versus 19.9k before #5's run. The box was 5–10% faster during #6, which favours the #6 runs slightly. Even with that edge, none of them won.
- **Recommendation:** keep #5 (d256-L6) as the base. It matches d256-L8 with 23% fewer params and about 23% more tok/s. At this budget, the curve is flat around 5–7M params.
- Note: 1M-val vs full-val rankings differ (d192 wins val-1M at 15 min, loses full val); always decide on full val.
