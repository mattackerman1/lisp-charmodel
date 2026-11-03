# #11 decay-from-checkpoint chart: train/val subset during the 720 s decay + midpoint and final full val vs #10 and #8
import csv, os, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D = os.path.dirname(os.path.abspath(__file__)); f = lambda x: float(x.replace('d0', ''))
rd = lambda p: [{k: f(v) for k, v in r.items()} for r in csv.DictReader(open(os.path.join(D, p)))]
sub = rd('v16384-decay.csv'); lng = rd('v16384-decay-long.csv'); fin = rd('v16384-decay-final.csv')
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
s = [r['seconds'] for r in sub]
ax[0].plot(s, [r['train_batch'] for r in sub], '^:', c='tab:gray', label='train batch (every 200 steps)')
ax[0].plot(s, [r['val_subset'] for r in sub], 'o-', c='tab:blue', ms=4, label='val subset (32k targets)')
ax[0].set_xlabel('training seconds of the decay'); ax[0].set_ylabel('nats/char'); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
ax[0].set_title('#11 linear decay 0.01 -> 0 over 720 s from #10 step 97,608')
pts = [(0, 1.0605)] + [(r['train_seconds'], r['val_full']) for r in lng] + [(r['train_seconds'], r['val_full10pct']) for r in fin]
ax[1].plot([p[0] for p in pts], [p[1] for p in pts], 'o-', c='tab:red', label='#11 full 10M val (start = #10 ckpt)')
for x, y in pts: ax[1].annotate(f'{y:.4f}', (x, y), textcoords='offset points', xytext=(5, 5), fontsize=8)
ax[1].axhline(1.0480, c='k', ls=':', lw=1, label='#8 1 h WSD 1.0480'); ax[1].axhline(1.0605, c='tab:gray', ls='--', lw=1, label='#10 best (const LR, 10 h) 1.0605')
ax[1].set_xlabel('training seconds of the decay'); ax[1].set_ylabel('full val nats/char'); ax[1].grid(alpha=.3); ax[1].legend(fontsize=8); ax[1].set_title('full validation')
fig.tight_layout(); fig.savefig(os.path.join(D, 'chart.png'), dpi=110)
