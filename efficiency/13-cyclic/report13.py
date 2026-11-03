# #13 chart: cyclic warm restarts (5 x [2 h constant + 720 s linear decay]), original segment + resumed segment (toff=21600.2 s)
import csv, os, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D = os.path.dirname(os.path.abspath(__file__)); E = os.path.dirname(D); f = lambda x: float(x.replace('d0', ''))
rd = lambda p: [{k: f(v) for k, v in r.items()} for r in csv.DictReader(open(p))]
TOFF = 21600.2
a = rd(os.path.join(D, 'v16384-cyc-long.csv')); b = rd(os.path.join(D, 'v16384-cyc-r1-long.csv'))
pts = [(r['train_seconds'], r['val_full'], r['train_recent']) for r in a] + [(TOFF + r['train_seconds'], r['val_full'], r['train_recent']) for r in b]
fin = rd(os.path.join(D, 'v16384-cyc-r1-final.csv'))[0]
ends = [(7920.3, 1.0317), (15840.4, 1.0227), (23760.4, 1.0212), (31680.2, 1.0164), (39600.4, fin['val_full10pct'])]
c10 = rd(os.path.join(E, '10-long/v16384-const-long.csv'))
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
ax[0].plot([r['train_seconds'] / 3600 for r in c10], [r['val_full'] for r in c10], 'o-', c='tab:gray', ms=3, label='#10 constant LR')
ax[0].plot([s / 3600 for s, _, _ in pts], [v for _, v, _ in pts], 'o-', c='tab:blue', ms=4, label='#13 cyclic (full val)')
for i, (s, v) in enumerate(ends):
    ax[0].plot(s / 3600, v, '*', c='tab:blue', ms=12); ax[0].annotate(f'c{i+1} {v:.4f}', (s / 3600, v), textcoords='offset points', xytext=(-10, -14), fontsize=7)
for h, v, lab in ((2.2, 1.0332, '2 h'), (5.2, 1.0209, '5 h'), (10.2, 1.0155, '10 h')):
    ax[0].plot(h, v, 's', c='tab:red', ms=7); ax[0].annotate(f'#12 {lab} {v:.4f}', (h, v), textcoords='offset points', xytext=(4, 6), fontsize=7, color='tab:red')
ax[0].axvline(TOFF / 3600, c='tab:purple', ls='--', lw=1, label='VM restart + resume (optimizer moments reset)')
ax[0].set_xlabel('training hours (eval excluded)'); ax[0].set_title('#13 cyclic warm restarts vs #10 constant and #12 branch decays')
ax[1].plot([s / 3600 for s, _, _ in pts], [v - t for _, v, t in pts], 'o-', c='tab:green', ms=4, label='full val - recent train')
ax[1].axvline(TOFF / 3600, c='tab:purple', ls='--', lw=1); ax[1].set_ylabel('gap, nats/char'); ax[1].set_xlabel('training hours'); ax[1].set_title('train vs val gap'); ax[1].grid(alpha=.3); ax[1].legend(fontsize=7)
ax[0].axhline(1.0480, c='k', ls=':', lw=1, label='1 h WSD baseline 1.0480'); ax[0].grid(alpha=.3); ax[0].set_ylabel('full 10M val, nats/char'); ax[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(D, 'chart.png'), dpi=110)
