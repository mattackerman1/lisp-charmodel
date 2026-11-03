# next/: transformer dropout and bigger GRU

Setup:
- Every run used the same fixed full-data split (val = last 10%), a 1800 s budget, 8 threads and nice -n 10.
- queue.sh and step3.sh ran the jobs one at a time. No science-team or Experimenter jobs were running at any job start (see queue.log).
- Both "val" columns are full fixed-val losses at the end of the run (nats/char). "Best subset" is the periodic subset val; see compare.png.

Gradient checks (float64, masks fixed), all PASS:
- gpt-drop.lisp (`gradcheck-drop.log`) at dropout 0 / 0.2 / 0.5: global rel err 3.6e-10 / 3.9e-10 / 9.0e-10. Max rel err over non-negligible entries was 8.6e-7 or less.
- gru-drop.lisp (`gradcheck-grudrop.log`), Lisp and BLAS backends, at dropout 0 / 0.2 / 0.5: global rel err 3.8e-10 or less, max rel err 4.2e-7 or less.

| Run | Params | steps | tok/s | best subset val (t) | final subset | **full val** | val rises late? |
|---|---|---|---|---|---|---|---|
| T1800 transformer, dropout 0 (earlier run) | 809,856 | 25,904 | 29,472 | 1.4100 (875 s) | 1.5254 | 1.6271 | yes |
| **D1 transformer, dropout 0.1** | 809,856 | 24,813 | 28,231 | 1.3531 (1785 s) | 1.3546 | **1.4379** | no |
| D2 transformer, dropout 0.2 | 809,856 | 23,502 | 26,739 | 1.4089 (1792 s) | 1.4089 | 1.4783 | no |
| GRU-G H=144, constant lr (earlier run) | 100,577 | 311,877 | 177,423 | 1.5054 (637 s) | 1.5282 | 1.6391 | flat |
| G288 H=288, warmup+cosine | 325,505 | 108,379 | 61,656 | 1.5002 (147 s) | 1.6909 | 1.8333 | yes (train subset 1.037) |
| G512 H=512, warmup+cosine | 922,689 | 28,586 | 16,262 | 1.4924 (245 s) | 1.7647 | 1.8767 | yes (train subset 0.904) |
| GD H=512 + non-recurrent dropout 0.25 (step 3) | 922,689 | 27,785 | 15,806 | 1.3769 (1728 s) | 1.3773 | 1.4930 | no |
| GD288 H=288 + dropout 0.25 | 325,505 | 98,744 | 56,174 | 1.3905 (1699 s) | 1.3918 | 1.4974 | no (train subset 1.224) |

- **Transformer.** Dropout removes the late val rise. D1 (p=0.1) gives the best model overall, with full val 1.4379 vs 1.6271 without dropout. p=0.2 is still improving at 1800 s but stays behind p=0.1 within this budget. Dropout costs 4% (p=0.1) to 9% (p=0.2) in steps per second.
- **Bigger GRU without dropout.** Both sizes overfit hard. Val bottoms out within the first 150–250 s and then climbs steadily while train loss keeps falling. Both end worse than GRU-G (1.6391).
- **Step 3 ran**, because both conditions held: dropout helped the transformer, and the bigger GRU's val rose over time. It used H=512, the larger size that overfit. Dropout (p=0.25) went on the input projection Ein[x] and on h before the output layer; the recurrence had none. Val then decreased steadily, giving full val 1.4930: 0.38 better than G512 and 0.15 better than GRU-G. It is still 0.055 behind the D1 transformer, and was still improving slowly at the end. The dropout rate was a single untuned choice.
- The protocols differ. The GRU's full val streams with carried state (unbounded context); the transformer's has 32–63 chars of context.

- **GD288 vs GD.** At H=288 with dropout 0.25, full val is 1.4974, 0.004 worse than GD (H=512, 1.4930), so H=288 does not beat it. It is 0.34 better than G288 without dropout (1.8333) and 0.060 behind D1 (1.4379). It runs at 3.6x GD's tokens/s and gets through 3.6x more data, but ends at essentially the same val. Its val was still edging down at 1800 s and never rose.

Best model (D1) 300-char sample, T=0.8: `best-sample-D1.txt`.
Files: gpt-drop.lisp, gru-big.lisp, gru-drop.lisp (+ .bak and backup/), queue.sh, step3.sh, queue.log, {D1,D2,G288,G512,GD}.{log,csv}, compare.png, plot_compare.py.
