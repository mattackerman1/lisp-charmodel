# compare.png: val loss vs training time (left) and vs tokens seen (right) for Lisp vs PyTorch (scaled model) + old small Lisp model.
import csv, os, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
g = lambda x: float(str(x).replace('d0', ''))
runs = [('lisp-big', 'Lisp gpt-huge d256 L6 H8 ctx128 (4.8M)', 'C2', 4096), ('torch-big', 'PyTorch same model', 'C3', 4096),
        ('lisp-small', 'Lisp small d128 L4 H4 ctx64 (0.81M), dropout 0.1', 'C0', 2048)]
fig, ax = plt.subplots(1, 2, figsize=(15, 5.6))
for tag, lab, c, tps in runs:
    if os.path.exists(tag + '.csv'):
        r = list(csv.DictReader(open(tag + '.csv')))
        ax[0].plot([g(x['seconds']) / 3600 for x in r], [g(x['val_subset']) for x in r], color=c, lw=.7, alpha=.45, label=lab + ' val subset (32k)')
        ax[0].plot([g(x['seconds']) / 3600 for x in r], [g(x['train_batch']) for x in r], color=c, lw=.4, alpha=.25, ls=':')
        ax[1].plot([g(x['step']) * tps / 1e6 for x in r], [g(x['val_subset']) for x in r], color=c, lw=.7, alpha=.45, label=lab + ' val subset')
    if os.path.exists(tag + '-full.csv'):
        r = list(csv.DictReader(open(tag + '-full.csv')))
        ax[0].plot([g(x['train_seconds']) / 3600 for x in r], [g(x['val_full']) for x in r], 'o--', color=c, label=lab + ' val (1M targets; last = full 10M)')
        ax[1].plot([g(x['step']) * tps / 1e6 for x in r], [g(x['val_full']) for x in r], 'o--', color=c)
        ax[0].annotate(f"{g(r[-1]['val_full']):.4f}", (g(r[-1]['train_seconds']) / 3600, g(r[-1]['val_full'])), textcoords='offset points', xytext=(4, 4), color=c, fontsize=9)
for a, xl in [(ax[0], 'training hours (eval excluded)'), (ax[1], 'training tokens seen (millions); 1 epoch = 89.9M')]:
    a.set_ylim(1.0, 2.0); a.grid(alpha=.3); a.set_xlabel(xl); a.set_ylabel('val loss (nats/char)')
ax[0].legend(fontsize=7.5); ax[1].axvline(89.9, color='k', lw=.6, ls=':')
fig.suptitle('Char transformers on huge/corpus.txt (121 Gutenberg books, 99.9 MB, vocab 95; val = last 10%). Dotted thin lines: train batch loss.')
plt.tight_layout(); plt.savefig('compare.png', dpi=110); print('wrote compare.png')
