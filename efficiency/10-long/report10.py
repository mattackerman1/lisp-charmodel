# #10 long-run analysis: reads 10-long/v16384-const-long.csv, writes chart.png and summary.md (included by make_report.py)
import csv, os, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D = os.path.dirname(os.path.abspath(__file__)); f = os.path.join(D, 'v16384-const-long.csv')
TT = 23505859; TOK = 4096; B1H = 1.0480   # v16384 1 h WSD full val (#8)
rows = [{k: float(v.replace('d0', '')) for k, v in r.items()} for r in csv.DictReader(open(f))] if os.path.exists(f) else []
if rows:
    h = [r['train_seconds'] / 3600 for r in rows]; fv = [r['val_full'] for r in rows]; v1 = [r['val_1m'] for r in rows]; tr = [r['train_recent'] for r in rows]
    bi = min(range(len(rows)), key=lambda i: fv[i]); b = rows[bi]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    ax[0].plot(h, fv, 'o-', c='tab:red', label='FULL 10M val (every 30 min)')
    ax[0].plot(h, v1, 's--', c='tab:blue', ms=4, label='val, first 1M chars (separate series)')
    ax[0].plot([x for x, t in zip(h, tr) if t > 0], [t for t in tr if t > 0], '^:', c='tab:gray', ms=4, label='train batch, avg of last 10 logs (2000 steps)')
    ax[0].axhline(B1H, c='k', lw=0.8, ls=':', label=f'1 h WSD v16384 full val {B1H}')
    ax[0].axvline(h[bi], c='tab:red', lw=0.8, ls=':'); ax[0].set_xlabel('training hours (eval excluded)'); ax[0].set_ylabel('nats/char')
    ax[0].set_title('#10 constant-LR long run, v16384 d256-L6'); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    sec = ax[0].secondary_xaxis('top', functions=(lambda x: x * 3600 * (rows[-1]['step'] / rows[-1]['train_seconds']) * TOK / TT, lambda p: p * TT / TOK / (rows[-1]['step'] / rows[-1]['train_seconds']) / 3600)); sec.set_xlabel('passes over train tokens (approx.)')
    ax[1].plot(h, [a - t if t > 0 else float('nan') for a, t in zip(fv, tr)], 'o-', c='tab:purple'); ax[1].axhline(0, c='k', lw=.5)
    ax[1].set_xlabel('training hours'); ax[1].set_ylabel('full val - train (nats/char)'); ax[1].set_title('train-vs-val gap'); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(D, 'chart.png'), dpi=110)
    L = ['| eval | train h | wall h | step | passes | val 1M | full val | train (recent) | full - train |', '|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        L.append(f"| {'**best**' if r is b else ''} | {r['train_seconds']/3600:.2f} | {r['wall_seconds']/3600:.2f} | {int(r['step'])} | {r['step']*TOK/TT:.2f} | {r['val_1m']:.4f} | {r['val_full']:.4f} | {r['train_recent']:.4f} | {r['val_full']-r['train_recent']:+.3f} |")
    L += ['', f"Best full val {b['val_full']:.4f} at {b['train_seconds']/3600:.2f} training h, step {int(b['step'])}, {b['step']*TOK/TT:.2f} passes (checkpoint 10-long/v16384-const-ckpt-{int(b['step']):06d}.f32)."]
    open(os.path.join(D, 'summary.md'), 'w').write('\n'.join(L) + '\n')
