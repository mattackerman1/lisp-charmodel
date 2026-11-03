# speed.png: left = tokens/s before/after; right = val loss vs training time (baseline Lisp, PyTorch, gpt-fast).
import csv, os, re, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
g = lambda x: float(str(x).replace('d0', ''))
B = '../big/'
def tps(log):
    if not os.path.exists(log): return None
    m = re.search(r'([\d.]+) tokens/s \(training time\)', open(log).read())
    return float(m.group(1)) if m else None
bars = [('gpt-big.lisp\n1 h run (baseline)', tps(B+'lisp.log'), 'C0'),
        ('PyTorch\n1 h run', tps(B+'torch.log'), 'C3'),
        ('gpt-big.lisp\n300 s bench', tps('bench-old.log'), 'C0'),
        ('gpt-fast.lisp\n300 s bench', tps('bench-new.log'), 'C2'),
        ('gpt-fast.lisp\n1 h run', tps('fast.log'), 'C2')]
bars = [b for b in bars if b[1]]
fig, ax = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={'width_ratios': [1, 1.4]})
xs = range(len(bars))
ax[0].bar(xs, [b[1] for b in bars], color=[b[2] for b in bars])
for x, b in zip(xs, bars): ax[0].text(x, b[1], f'{b[1]/1000:.1f}k', ha='center', va='bottom', fontsize=10)
ax[0].set_xticks(list(xs)); ax[0].set_xticklabels([b[0] for b in bars], fontsize=8)
ax[0].set_ylabel('training tokens/s (eval excluded), 8 threads'); ax[0].set_title('Throughput: same model, same maths, dropout 0.1'); ax[0].grid(axis='y', alpha=.3)
for tag, d, lab, c in [('lisp', B, 'gpt-big.lisp (baseline)', 'C0'), ('torch', B, 'PyTorch', 'C3'), ('fast', '', 'gpt-fast.lisp', 'C2')]:
    if os.path.exists(d+tag+'.csv'):
        r = list(csv.DictReader(open(d+tag+'.csv')))
        ax[1].plot([g(x['seconds']) for x in r], [g(x['val_subset']) for x in r], color=c, lw=.8, alpha=.5, label=lab+' val subset (32k)')
    if os.path.exists(d+tag+'-full.csv'):
        r = list(csv.DictReader(open(d+tag+'-full.csv')))
        ax[1].plot([g(x['train_seconds']) for x in r], [g(x['val_full']) for x in r], 'o--', color=c, label=lab+' FULL val')
        ax[1].annotate(f"{g(r[-1]['val_full']):.4f}", (g(r[-1]['train_seconds']), g(r[-1]['val_full'])), textcoords='offset points', xytext=(5, -12 if tag == 'fast' else 6), color=c, fontsize=9)
ax[1].set_ylim(1.15, 2.0); ax[1].grid(alpha=.3); ax[1].legend(fontsize=8); ax[1].set_xlabel('training seconds (eval excluded)'); ax[1].set_ylabel('val loss (nats/char)')
ax[1].set_title('Val loss vs training time, 3600 s budget, 9.58 MB corpus')
plt.tight_layout(); plt.savefig('speed.png', dpi=110); print('wrote speed.png', bars)
