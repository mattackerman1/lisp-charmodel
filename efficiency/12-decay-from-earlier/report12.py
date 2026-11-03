# #12 chart: 720 s linear decay started from #10 checkpoints at 2 h, 5 h (this folder) and 10 h (#11)
import csv, os, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D = os.path.dirname(os.path.abspath(__file__)); E = os.path.dirname(D); f = lambda x: float(x.replace('d0', ''))
rd = lambda p: [{k: f(v) for k, v in r.items()} for r in csv.DictReader(open(p))]
runs = [('2 h', 2.0, 1.0815, os.path.join(D, 'v16384-decay-2h'), 'tab:orange'), ('5 h', 5.0, 1.0679, os.path.join(D, 'v16384-decay-5h'), 'tab:green'),
        ('10 h (#11)', 10.0, 1.0605, os.path.join(E, '11-decay/v16384-decay'), 'tab:red')]
c10 = rd(os.path.join(E, '10-long/v16384-const-long.csv'))
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
ax[0].plot([r['train_seconds'] / 3600 for r in c10], [r['val_full'] for r in c10], 'o-', c='tab:gray', ms=4, label='#10 constant LR (full val)')
for lab, h, v0, p, c in runs:
    fv = [(0, v0)] + [(r['train_seconds'], r['val_full']) for r in rd(p + '-long.csv')] + [(r['train_seconds'], r['val_full10pct']) for r in rd(p + '-final.csv')]
    ax[0].plot([h + s / 3600 for s, _ in fv], [v for _, v in fv], 's-', c=c, lw=2, label=f'decay from {lab}: {fv[-1][1]:.4f}')
    ax[1].plot([s for s, _ in fv], [v for _, v in fv], 'o-', c=c, label=f'from {lab}')
    for s, v in fv: ax[1].annotate(f'{v:.4f}', (s, v), textcoords='offset points', xytext=(4, 4), fontsize=7)
for a in ax: a.axhline(1.0480, c='k', ls=':', lw=1, label='#8 1 h WSD 1.0480'); a.grid(alpha=.3); a.set_ylabel('full 10M val, nats/char'); a.legend(fontsize=7)
ax[0].set_xlabel('training hours (eval excluded)'); ax[0].set_title('720 s linear decay to 0 from #10 checkpoints')
ax[1].set_xlabel('seconds into the 720 s decay'); ax[1].set_title('full val during the decay (start, midpoint, end)')
fig.tight_layout(); fig.savefig(os.path.join(D, 'chart.png'), dpi=110)
