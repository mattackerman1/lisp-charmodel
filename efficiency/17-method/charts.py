#!/usr/bin/env python3
"""#17 charts: (1) proxy tricks vs 3 baseline seeds (val-subset curves + final full-val bars with 2x-noise bar),
(2) branch-decay tricks vs 2 baseline seeds, (3) final experiment curve vs #11 (10 h branch lin 720 s) with 1.0152 line."""
import csv, os, sys
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D = os.path.dirname(os.path.abspath(__file__)); R = D + '/runs/'; O = D + '/charts/'
f = lambda x: float(x.replace('d0', ''))
def sub(t):
    p = R + t + '.csv'
    return [(f(r['step']), f(r['val_subset'])) for r in csv.DictReader(open(p))] if os.path.exists(p) else []
led = {r['run_id']: r for r in csv.DictReader(open(D + '/results.tsv'), delimiter='\t')}
fv = lambda t: float(led[t]['full_val']) if t in led and led[t]['full_val'] else None
def panel(name, base, tricks, bar, title):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for b in base:
        s = sub(b); ax[0].plot([x for x, _ in s], [y for _, y in s], color='gray', lw=1, label=b)
    for t in tricks:
        s = sub(t)
        if s: ax[0].plot([x for x, _ in s], [y for _, y in s], lw=1.6, label=t)
    ax[0].set_xlabel('step'); ax[0].set_ylabel('val subset (32k targets), nats/char'); ax[0].legend(fontsize=7); ax[0].set_title(title + ': curves')
    names = [b for b in base if fv(b)] + [t for t in tricks if fv(t)]; vals = [fv(n) for n in names]
    cols = ['gray' if n in base else ('tab:green' if fv(n) < bar else 'tab:red') for n in names]
    ax[1].barh(names, vals, color=cols); ax[1].axvline(bar, color='k', ls='--', label=f'2x-noise bar {bar:.4f}')
    lo = min(vals) - 0.003; ax[1].set_xlim(lo, max(vals) + 0.002)
    for i, v in enumerate(vals): ax[1].text(v, i, f' {v:.4f}', va='center', fontsize=8)
    ax[1].set_xlabel('final full val (nats/char), lower is better'); ax[1].legend(fontsize=8); ax[1].set_title(title + ': final full val')
    fig.tight_layout(); fig.savefig(O + name, dpi=110); plt.close(fig); print('wrote', O + name)
panel('proxy-tricks.png', ['proxy-base-s1', 'proxy-base-s2', 'proxy-base-s3'],
      ['proxy-relu2', 'proxy-qknorm', 'proxy-qknorm-s2', 'proxy-zinit', 'proxy-softcap'], 1.1210, 'proxy (2500 steps from scratch)')
panel('decay-tricks.png', ['decay-base-s11', 'decay-base-s12'],
      ['decay-sqrt', 'decay-elr3e-4', 'decay-mlr0.007b', 'decay-elr3e-3', 'decay-mlr0.014b', 'decay-stack'], 1.0326, 'branch decay from 2 h ckpt (1868 steps)')
if len(sys.argv) > 1:   # final run id
    fid = sys.argv[1]; fig, ax = plt.subplots(figsize=(9, 5))
    s = sub(fid); ax.plot([x for x, _ in s], [y for _, y in s], lw=1.8, color='tab:green', label=f'#17 final {fid}: sqrt-shape decay (dshape=sqrt only; id named before the pick), {int(led[fid]["steps"])} steps (full val {fv(fid):.4f})')
    p = D + '/../11-decay/v16384-decay.csv'
    rr = [(f(r['step']), f(r['val_subset'])) for r in csv.DictReader(open(p))]
    ax.plot([x for x, _ in rr], [y for _, y in rr], color='gray', lw=1.2, label='#11 plain lin decay 1,966 steps from same 10 h ckpt (full val 1.0155)')
    # #13 cycle-5 decay (resumed segment v16384-cyc-r1; steps local to the 6 h resume at 56,223). Cycle 5 decay starts at
    # 4*7920+7200 = 38,880 training s total = 17,279.8 s local (toff 21,600.2); interpolate the step from the seconds column.
    p13 = D + '/../13-cyclic/v16384-cyc-r1.csv'
    if os.path.exists(p13):
        c = [(f(r['step']), f(r['seconds']), f(r['val_subset'])) for r in csv.DictReader(open(p13))]
        t0 = 38880.0 - 21600.2
        s0 = next(a[0] + (t0 - a[1]) / (b[1] - a[1]) * (b[0] - a[0]) for a, b in zip(c, c[1:]) if a[1] <= t0 < b[1])
        cc = [(x - s0, y) for x, _, y in c if x >= s0 - 1000]
        ax.plot([x for x, _ in cc], [y for _, y in cc], color='tab:orange', lw=1.2,
                label=f'#13 cyclic run, last cycle: lin decay 720 s (~{44381 - s0:.0f} steps) ending at step 100,604 (full val 1.0152)')
        print('#13 cycle-5 decay start (local step)', round(s0))
    ax.axvline(0, color='k', lw=0.6, ls=':')
    ax.set_xlabel('step after decay start (#17/#11: #10 10 h ckpt, step 97,608; #13: cycle-5 decay start)'); ax.set_ylabel('val subset (32k targets), nats/char')
    ax.legend(fontsize=8); ax.set_title('#17 final vs #11 and #13 (val-subset curves; full val in legend)')
    fig.tight_layout(); fig.savefig(O + 'final.png', dpi=110); plt.close(fig); print('wrote', O + 'final.png')
    # summary: full val of every #17 decision vs its baseline mean, plus the final vs 1.0152
    fig, ax = plt.subplots(figsize=(10, 5.5))
    rows = [(r, float(led[r]['full_val']) - (1.0334 if r.startswith('decay') else 1.1246), led[r]['keep']) for r in led
            if led[r]['keep'] in ('keep', 'discard') or r in ('decay-stack', 'proxy-qknorm-s2')]
    rows.append((fid + ' vs 1.0152', fv(fid) - 1.0152, 'final'))
    cmap = {'keep': 'tab:green', 'discard': 'tab:red', 'final': 'tab:blue'}
    ax.barh([r[0] for r in rows], [r[1] for r in rows], color=[cmap.get(r[2], 'tab:purple') for r in rows])
    ax.axvline(-0.0008, color='k', ls=':', lw=1, label='decay 2x-noise bar (-0.0008 vs mean 1.0334)')
    ax.axvline(-0.0036, color='k', ls='--', lw=1, label='proxy 2x-noise bar (-0.0036 vs mean 1.1246)')
    for i, r in enumerate(rows): ax.text(r[1], i, f' {r[1]:+.4f}', va='center', fontsize=8)
    ax.axvline(0, color='gray', lw=0.8); ax.set_xlabel('full val minus baseline (nats/char), negative = better')
    ax.legend(fontsize=8); ax.set_title('#17 keep/discard summary (green keep, red discard, purple stack/2nd seed, blue final)')
    fig.tight_layout(); fig.savefig(O + 'summary.png', dpi=110); plt.close(fig); print('wrote', O + 'summary.png')
