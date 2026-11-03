# Writes REPORT.md from the run logs/CSVs (re-runnable; called by finalize.sh when the queue is done).
import json, re, os, csv
g = lambda x: float(str(x).replace('d0', ''))
meta = json.load(open('corpus_meta.json'))
def parse(tag):
    lg = tag + '.log'
    if not os.path.exists(lg): return None
    s = open(lg).read(); d = {'log': lg}
    m = re.search(r'trained (\d+) steps in ([\d.]+)s training time \(wall ([\d.]+)s, eval ([\d.]+)s\): ([\d.]+) steps/s, ([\d.]+) tokens/s', s)
    if m: d.update(steps=int(m[1]), secs=float(m[2]), wall=float(m[3]), evals=float(m[4]), sps=float(m[5]), tps=float(m[6]))
    m = re.search(r'FINAL: val\(full fixed val, (\d+) targets scored\)=([\d.]+)', s)
    if m: d.update(final=float(m[2]), ntgt=int(m[1]))
    m = re.search(r'params=(\d+)', s); d['params'] = int(m[1]) if m else None
    m = re.search(r'=== sample \(T=0.8\) ===\n(.*)', s, re.S); d['sample'] = m[1].strip() if m else None
    d['full'] = [(int(r['step']), g(r['train_seconds']), g(r['val_full'])) for r in csv.DictReader(open(tag + '-full.csv'))] if os.path.exists(tag + '-full.csv') else []
    d['hist'] = [(int(r['step']), g(r['seconds']), g(r['train_batch']), g(r['val_subset'])) for r in csv.DictReader(open(tag + '.csv'))] if os.path.exists(tag + '.csv') else []
    return d
R = {t: parse(t) for t in ['lisp-big', 'torch-big', 'lisp-small']}
def tail_gap(d, n=25):   # mean(val_subset - train_batch) over the last n periodic points, and the same over points 25..50
    h = d['hist']
    if len(h) < 60: return None
    f = lambda pts: sum(v - t for _, _, t, v in pts) / len(pts)
    return f(h[-n:]), f(h[25:50])
L = []
L.append('# huge/: bigger char transformer on a 100 MB Gutenberg corpus, Lisp (gpt-huge.lisp) vs PyTorch\n')
L.append('## Corpus (`corpus.txt`, built by `build_corpus.py` from `raw/`; per-book offsets/splits in `corpus_meta.json`)')
L.append(f"- **{len(meta['books'])} public-domain English Project Gutenberg books, {meta['chars']:,} chars ({meta['chars']/1e6:.1f} MB), vocab {len(meta['vocab'])}** "
         "(printable ASCII 32-126 + newline; curly quotes/dashes mapped, accents stripped via NFKD, tab -> space, other characters dropped).")
L.append("- Gutenberg header/license stripped (only text between the `*** START OF` / `*** END OF` markers). Downloaded 2026-10-17 from `https://www.gutenberg.org/cache/epub/<id>/pg<id>.txt` (`fetch.sh`, 1 request/s).")
L.append("- Order: the 14 books of big/corpus.txt first, then the rest shuffled with a fixed seed; filled up to 100 MB (14 further downloaded books did not fit and were left out). "
         f"Val = last 10% (chars from {meta['val_start']:,}; {meta['chars']-meta['val_start']:,} chars), which spans: " +
         ', '.join(b['title'] for b in meta['books'] if b['split'] != 'train') + '.')
L.append('- Largest books: ' + ', '.join(f"{b['title']} ({b['chars']/1e6:.1f} MB)" for b in sorted(meta['books'], key=lambda b: -b['chars'])[:6]) + '. Full list:\n')
L.append('<details><summary>all books</summary>\n\n' + '\n'.join(f"- {b['id']} {b['title']} -- {b['author']} ({b['chars']:,} chars, {b['split']})" for b in meta['books']) + '\n</details>\n')
L.append('## Setup (identical in both implementations)')
L.append('- Model: pre-LN GPT, **d=256, 6 layers, 8 heads (head dim 32), ctx 128**, GELU (tanh), LM head tied to wte, learned positions. '
         f"**{R['lisp-big']['params'] if R['lisp-big'] else 4796160:,} params** (vocab 95).")
L.append('- **Dropout 0** (100 MB of data, ~2-3 passes in 3 h; see overfitting check below). Init as before (N(0,0.02); proj/fc2 N(0,0.02/sqrt(2L))).')
L.append('- AdamW lr 1e-3, betas 0.9/0.99, eps 1e-8, wd 0.1 on matrices+embeddings, grad-norm clip 1.0, 2% warmup then cosine to lr/10 over **10800 s of training time** (eval excluded). Batch 32 x 128 = 4096 tokens. float32, 8 threads, `nice -n 10`, run back to back by `queue.sh` (`queue.log`).')
L.append('- Eval: windows of 128, stride 64, score the 2nd half (64-127 chars of context). Subset val (first 32k val targets) every 200 steps; '
         '"val (1M)" = first 1,000,000 val targets every 1800 s; **FINAL = the whole 10% val set (9.99M chars)**.')
L.append('- Code: `gpt-huge.lisp` = fast-gpt/gpt-fast.lisp + octet corpus loading + `fullval-targets=`; `gpt_torch.py` = big/gpt_torch.py + uint8 data + `fullval-targets=`. Runtime options only, no maths changes.')
L.append('- Gradient check of the scaled code (`gradcheck.log`, `equiv.log`, scripts in `check/`): analytic gradients of gpt-huge.lisp are **bitwise identical** to the reference gpt-big.lisp on a '
         'scaled-shape tiny config (V=11 d=16 L=6 H=8 T=16; 20,144 params, dropout 0/0.2) and on the full 4.8M-param config (8 workers, dropout 0/0.05): 0 differing gradients. '
         'The reference in double precision matches finite differences on the scaled-shape tiny config with global rel. error 1.5e-9 / 2.2e-9 / 3.1e-9 (dropout 0 / 0.2 / 0.5).\n')
L.append('## Results')
hdr = '| | Lisp gpt-huge.lisp (4.8M) | PyTorch 2.14 CPU (4.8M) | Lisp small model (0.81M, d128 L4 H4 ctx64, dropout 0.1, 1 h) |'
L.append(hdr); L.append('|---|---|---|---|')
def cell(d, k, fmt):
    return fmt.format(d[k]) if d and k in d and d[k] is not None else 'n/a'
for lab, k, fmt in [('training time', 'secs', '{:.0f} s'), ('steps', 'steps', '{:,}'), ('tokens/s (training time)', 'tps', '**{:,.0f}**'),
                    ('eval time (excluded)', 'evals', '{:.0f} s'), ('**final val, full 10% (9.99M chars)**', 'final', '**{:.4f}**')]:
    L.append(f'| {lab} | ' + ' | '.join(cell(R[t], k, fmt) for t in ['lisp-big', 'torch-big', 'lisp-small']) + ' |')
L.append('| tokens seen (epochs of 89.9M train chars) | ' + ' | '.join(
    (f"{R[t]['steps']*(4096 if t!='lisp-small' else 2048)/1e6:.0f}M ({R[t]['steps']*(4096 if t!='lisp-small' else 2048)/89.9e6:.2f})" if R[t] and 'steps' in R[t] else 'n/a') for t in ['lisp-big', 'torch-big', 'lisp-small']) + ' |')
L.append('| val (first 1M targets) at 0.5/1/1.5/2/2.5 h | ' + ' | '.join(
    (' / '.join(f'{v:.4f}' for _, _, v in R[t]['full'][:-1]) if R[t] and R[t]['full'] else 'n/a') for t in ['lisp-big', 'torch-big', 'lisp-small']) + ' |')
L.append('')
lb, tb = R['lisp-big'], R['torch-big']
if lb and tb and 'tps' in lb and 'tps' in tb:
    L.append(f"- Lisp ran {lb['tps']/tb['tps']:.2f}x the tokens/s of PyTorch on this model ({lb['tps']:,.0f} vs {tb['tps']:,.0f}).")
if lb and tb and 'final' in lb and 'final' in tb:
    L.append(f"- Final full val: Lisp {lb['final']:.4f} vs PyTorch {tb['final']:.4f} (difference {tb['final']-lb['final']:+.4f} nats/char in Lisp's favour if positive); "
             "per optimizer step the two implementations compute the same function, so the gap comes from the number of steps in the time budget (see right panel of compare.png).")
if R['lisp-small'] and 'final' in R['lisp-small'] and lb and 'final' in lb:
    L.append(f"- The old small model (0.81M params, 1 h) reaches {R['lisp-small']['final']:.4f} on the same val set; the 4.8M model improves on it by {R['lisp-small']['final']-lb['final']:.4f} nats/char.")
L.append('\n## Overfitting check')
for t, lab in [('lisp-big', 'Lisp 4.8M'), ('torch-big', 'PyTorch 4.8M'), ('lisp-small', 'Lisp small')]:
    d = R[t]
    if not d or not d['full']: continue
    vs = [v for _, _, v in d['full'][:-1]]
    mono = all(b <= a + 1e-4 for a, b in zip(vs, vs[1:]))
    gp = tail_gap(d)
    L.append(f"- {lab}: periodic val (1M) {'falls at every checkpoint' if mono else 'does NOT fall monotonically: ' + ' / '.join(f'{v:.4f}' for v in vs)}"
             + (f"; mean (val subset - train batch) = {gp[0]:+.3f} over the last 25 checkpoints vs {gp[1]:+.3f} early in training" if gp else '') + '.')
L.append('- Negative or near-zero val-minus-train gaps mean no measurable overfitting (the val books differ in style from the train mix, so absolute train/val levels are not directly comparable; the trend is what matters).\n')
L.append('## Samples (300 chars, T=0.8, starting from a newline)')
for t, lab in [('lisp-big', 'Lisp 4.8M'), ('torch-big', 'PyTorch 4.8M'), ('lisp-small', 'Lisp small (0.81M)')]:
    if R[t] and R[t]['sample']: L.append(f'{lab}:\n```\n{R[t]["sample"]}\n```')
L.append('\n## Files (all in /home/box/lisp-charmodel/huge/)')
L.append('corpus.txt, corpus_meta.json, build_corpus.py, fetch.sh, ids.txt, raw/ | gpt-huge.lisp, gpt_torch.py | queue.sh, queue.log | '
         'lisp-big.{log,csv}, lisp-big-full.csv, torch-big.{log,csv}, torch-big-full.csv, torch-big.pt, lisp-small.{log,csv}, lisp-small-full.csv | '
         'gradcheck.log, equiv.log, check/ | compare.png (plot_compare.py) | REPORT.md (make_report.py, finalize.sh)')
open('REPORT.md', 'w').write('\n'.join(L) + '\n'); print('wrote REPORT.md')
