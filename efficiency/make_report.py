#!/usr/bin/env python3
"""Builds per-variant charts (variant vs baseline) and the running REPORT.md from whatever results exist."""
import csv, glob, os, re
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
os.chdir(os.path.dirname(os.path.abspath(__file__)))
f = lambda x: float(x.replace('d0', ''))
def rd(p):
    if not os.path.exists(p): return []
    return [{k: f(v) for k, v in r.items()} for r in csv.DictReader(open(p))]
def run(tag):
    log = open(tag + '.log').read() if os.path.exists(tag + '.log') else ''
    m = re.search(r'trained (\d+) steps in ([\d.]+)s training time .*?([\d.]+) tokens/s', log)
    smp = log.split('=== sample', 1)[1].strip() if '=== sample' in log else ''
    return dict(tag=tag, sub=rd(tag + '.csv'), v1m=rd(tag + '-val1m.csv'), fin=rd(tag + '-final.csv'),
                steps=int(m[1]) if m else None, tps=float(m[3]) if m else None, sample=smp, done=bool(m))
def overfit(r):
    s = r['sub']
    if len(s) < 8: return None
    q = s[-max(4, len(s) // 10):]           # last 10% of the 200-step points
    tb = sum(x['train_batch'] for x in q) / len(q); vs = sum(x['val_subset'] for x in q) / len(q)
    return tb, vs
def chart(var, base, name, out, extra=(), blabel='baseline'):
    fig, ax0 = plt.subplots(figsize=(8, 5)); fig1, ax1 = plt.subplots(figsize=(8, 5)); ax = [ax0, ax1]
    for r, lab, c in ((base, blabel, 'tab:gray'),) + tuple(extra) + ((var, name, 'tab:red'),):
        if r is None: continue
        if r['v1m']:
            ax[0].plot([x['train_seconds'] / 60 for x in r['v1m']], [x['val_1m'] for x in r['v1m']], 'o-', c=c, label=f'{lab}: val, first 1M val chars (every 15 min + end)')
        for x in r['fin']:
            ax[0].plot(x['train_seconds'] / 60, x['val_full10pct'], 's', ms=10, mfc='none', mew=2, c=c, label=f'{lab}: FULL 10M val at end = {x["val_full10pct"]:.4f}')
        if r['sub']:
            t = [x['seconds'] / 60 for x in r['sub']]
            k = 10; tb = [x['train_batch'] for x in r['sub']]
            tbs = [sum(tb[max(0, i - k + 1):i + 1]) / len(tb[max(0, i - k + 1):i + 1]) for i in range(len(tb))]
            ax[1].plot(t, tbs, '-', c=c, alpha=.5, label=f'{lab}: train batch loss (10-pt avg)')
            ax[1].plot(t, [x['val_subset'] for x in r['sub']], '--', c=c, label=f'{lab}: val subset (32k chars)')
    ax[0].set(title='Fixed val measures (separate series; not mixed)', xlabel='training minutes', ylabel='loss (nats/char)')
    ax[1].set(title='Overfitting check: train batch vs val subset', xlabel='training minutes', ylim=(0.95, 1.6))
    for a in ax: a.grid(alpha=.3); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out, dpi=110); fig1.tight_layout(); fig1.savefig(out.replace('chart.png', 'overfit.png'), dpi=110); plt.close('all')
base = run('baseline/baseline') if os.path.exists('baseline/baseline.log') else None
sweep = {}
for lg in sorted(glob.glob('01-lr/sweep/*.log')):
    t = lg[:-4]; sweep[os.path.basename(t)] = run(t)
lr1 = run('01-lr/lr-wsd') if os.path.exists('01-lr/lr-wsd.log') else None
best = open('01-lr/best_lr.txt').read().strip() if os.path.exists('01-lr/best_lr.txt') else None
mu = run('02-muon/muon') if os.path.exists('02-muon/muon.log') else None
mbest = open('02-muon/best_mlr.txt').read().strip() if os.path.exists('02-muon/best_mlr.txt') else None
msweep = {os.path.basename(lg[:-4]): run(lg[:-4]) for lg in sorted(glob.glob('02-muon/sweep/*.log'))}
probe = run('02-muon/speed-probe-adamw') if os.path.exists('02-muon/speed-probe-adamw.log') else None
cx = run('03-ctx/ctx') if os.path.exists('03-ctx/ctx.log') else None
cbest = open('03-ctx/best.txt').read().strip() if os.path.exists('03-ctx/best.txt') else None
cmlr = open('03-ctx/mlr.txt').read().strip() if os.path.exists('03-ctx/mlr.txt') else None
csweep = {os.path.basename(lg[:-4]): run(lg[:-4]) for lg in sorted(glob.glob('03-ctx/sweep/*.log'))}
cprobe = run('03-ctx/speed-probe-ctx128') if os.path.exists('03-ctx/speed-probe-ctx128.log') else None
if cx or csweep: chart(cx, base, f'#3 ctx64->128 ({cbest}) + Muon + WSD', '03-ctx/chart.png', extra=((lr1, f'#1 WSD', 'tab:blue'), (mu, f'#2 Muon', 'tab:green')))
bp = run('04-bpe/bpe') if os.path.exists('04-bpe/bpe.log') else None
bmlr = open('04-bpe/best_mlr.txt').read().strip() if os.path.exists('04-bpe/best_mlr.txt') else None
bsweep = {os.path.basename(lg[:-4]): run(lg[:-4]) for lg in sorted(glob.glob('04-bpe/sweep/*.log'))}
bprobe = run('04-bpe/speed-probe-char') if os.path.exists('04-bpe/speed-probe-char.log') else None
import json as _j
bmeta = _j.load(open('04-bpe/meta2048.json')) if os.path.exists('04-bpe/meta2048.json') else None
if bp or bsweep: chart(bp, base, f'#4 BPE-2048 + Muon + WSD (nats/char)', '04-bpe/chart.png', extra=((lr1, '#1 WSD', 'tab:blue'), (mu, '#2 Muon', 'tab:green'), (cx, '#3 ctx64->128', 'tab:orange')))
ew = run('05-precision/ew32') if os.path.exists('05-precision/ew32.log') else None
eprobe = run('05-precision/speed-probe-fp32') if os.path.exists('05-precision/speed-probe-fp32.log') else None
if ew: chart(ew, base, '#5 #4 + f32 GELU (nats/char)', '05-precision/chart.png', extra=((mu, '#2 Muon', 'tab:green'), (bp, '#4 BPE-2048', 'tab:purple')))
sizes = [(n, run('06-size/' + n)) for n in ('d192-L6', 'd256-L8', 'd384-L6') if os.path.exists(f'06-size/{n}.log')]
if sizes:
    cols = {'d192-L6': 'tab:cyan', 'd256-L8': 'tab:orange', 'd384-L6': 'tab:brown'}
    chart(None, ew, '', '06-size/chart.png', extra=tuple((r, f'#6 {n}', cols[n]) for n, r in sizes), blabel='#5 d256-L6 (5.3M)')
if base: chart(None, base, 'baseline', 'baseline/chart.png')
if mu or msweep: chart(mu, base, f'#2 Muon mlr={mbest} + WSD', '02-muon/chart.png', extra=((lr1, f'#1 WSD lr={best}', 'tab:blue'),))
if lr1 or base: chart(lr1, base, f'#1 WSD lr={best}', '01-lr/chart.png')
if sweep:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, r in sweep.items():
        if r['sub']: ax.plot([x['seconds'] / 60 for x in r['sub']], [x['val_subset'] for x in r['sub']], label=k + (f" (end val1M {r['v1m'][-1]['val_1m']:.4f})" if r['v1m'] else ''))
    ax.set(title='#1 LR sweep (15 min each): val subset over time', xlabel='training minutes', ylabel='val subset loss', ylim=(1.1, 1.8)); ax.grid(alpha=.3); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig('01-lr/sweep.png', dpi=110); plt.close()
def row(name, r):
    if r is None: return f'| {name} | (not run yet) | | | | | |'
    v1 = ' / '.join(f"{x['val_1m']:.4f}" for x in r['v1m']) or '-'
    fv = f"{r['fin'][0]['val_full10pct']:.4f}" if r['fin'] else '-'
    o = overfit(r); os_ = f'{o[0]:.3f} vs {o[1]:.3f}' if o else '-'
    return f"| {name} | {v1} | {fv} | {r['tps'] or 0:,.0f} | {r['steps'] or '-'} | {os_} | {'done' if r['done'] else 'running'} |"
L = ['# Efficiency series: 4.8M-param Lisp transformer, 1-hour budget per run', '',
     'Model: d256, 6 layers, 8 heads, ctx 128, batch 32x128, dropout 0, 100 MB Gutenberg corpus (`../huge/corpus.txt`). 8 threads, `nice -n 10`, one job at a time (`queue.sh`, resumable).',
     'Every run is scored on the same fixed measures, which are kept as separate series and never mixed:',
     '* **val 1M**: the first 1,000,000 val characters, every 15 min of training time and at the end.',
     '* **full val**: all 9,988,894 val characters (the last 10% of the corpus), once at the end. This slice is harder than the 1M slice, so the two numbers differ.',
     'Train-batch loss vs the 32k-char val subset (both every 200 steps) is the overfitting check, shown as the last-10% average train vs val.', '',
     '| run | val 1M at 15/30/45/60 min + end | full val (end) | tok/s | steps | overfit: train vs val subset | status |', '|---|---|---|---|---|---|---|',
     row('baseline: AdamW lr 1e-3, cosine (2% warmup, to lr/10)', base),
     row(f'#1: WSD lr {best} (1% warmup, constant, linear decay to 0 over last 20%)', lr1),
     row(f'#2: Muon (mlr {mbest}) on block matrices + AdamW rest, WSD', mu),
     row(f'#3: #2 (mlr {cmlr}) + train ctx 64 (64 seqs) then 128 ({cbest})', cx),
     row(f'#4: #2 + BPE-2048 tokens, ctx 64 tok, 64 seqs (mlr {bmlr}); tok/s = TOKENS/s', bp),
     row('#5: #4 + single-precision GELU (bf16 AMX GEMM rejected, see notes); tok/s = TOKENS/s', ew), '',
     'All losses are nats per character (#4: total val nats / val characters scored), so all rows compare directly.' + (f" #4 chars/s = tokens/s x {bmeta['chars_per_token_train']:.3f} chars/token (train) = {bp['tps'] * bmeta['chars_per_token_train']:,.0f} chars/s." if bp and bp['tps'] and bmeta else ''), '',
     '## #1 LR sweep (15 min each, schedule fit to 15 min, scored on val 1M at the end)', '', '| run | val 1M at end | tok/s | steps |', '|---|---|---|---|']
for k, r in sweep.items():
    L.append(f"| {k} | {r['v1m'][-1]['val_1m']:.4f} | {r['tps'] or 0:,.0f} | {r['steps'] or '-'} |" if r['v1m'] else f'| {k} | running | | |')
L += ['', f'Best WSD peak LR: **{best}**' if best else '', '',
      'Charts: `10-long/chart.png` (#10), `09-dropout/chart.png` (#9), `08-vocab-up/chart.png` (#8 vs v8192), `07-vocab/chart.png` (#7 vocab vs #5), `06-size/chart.png` (#6 sizes vs #5), `05-precision/chart.png` (#5 vs #4/#2/baseline), `04-bpe/chart.png` (#4 vs #3/#2/#1/baseline), `04-bpe/overfit.png`, `03-ctx/chart.png` (#3 vs #2 vs #1 vs baseline), `03-ctx/overfit.png`, `02-muon/chart.png` (#2 vs #1 vs baseline), `02-muon/overfit.png`, `01-lr/chart.png` (#1 vs baseline: val-1M series + full-val end markers), `01-lr/overfit.png` (train vs val subset), `01-lr/sweep.png`, `baseline/chart.png`.', '']
if msweep:
    L += ['## #2 Muon sweep (15 min each, WSD, AdamW lr 1e-3 for embeddings/norms/biases; val 1M at end)', '', '| run | val 1M at end | tok/s | steps |', '|---|---|---|---|']
    for k, r in msweep.items():
        L.append(f"| {k} | {r['v1m'][-1]['val_1m']:.4f} | {r['tps'] or 0:,.0f} | {r['steps'] or '-'} |" if r['v1m'] else f'| {k} | running | | |')
    if probe and probe['tps']: L.append(f"\nAdamW speed probe (3 min, right before the 1-h Muon run): {probe['tps']:,.0f} tok/s")
    L += ['', 'Muon optimizer check (tiny matrices, 2 steps, vs numpy float64 reference of the Keller Jordan algorithm): see `02-muon/muon_check.log`.', '']
if csweep:
    L += ['## #3 short-context sweep (15 min each, Muon + WSD; eval always at ctx 128; val 1M at end)', '', '| run | val 1M at end | tok/s | steps |', '|---|---|---|---|']
    for k, r in csweep.items():
        L.append(f"| {k} | {r['v1m'][-1]['val_1m']:.4f} | {r['tps'] or 0:,.0f} | {r['steps'] or '-'} |" if r['v1m'] else f'| {k} | running | | |')
    if cprobe and cprobe['tps']: L.append(f"\nctx-128 Muon speed probe (3 min, right before the 1-h #3 run): {cprobe['tps']:,.0f} tok/s")
    L += ['', 'Short-context gradient check (single float, finite differences): `03-ctx/gradcheck.log`.', '']
if bsweep:
    L += ['## #4 BPE sweep (15 min each; val 1M in nats/char at end)', '', '| run | val 1M at end (nats/char) | tokens/s | steps |', '|---|---|---|---|']
    for k, r in bsweep.items():
        L.append(f"| {k} | {r['v1m'][-1]['val_1m']:.4f} | {r['tps'] or 0:,.0f} | {r['steps'] or '-'} |" if r['v1m'] else f'| {k} | running | | |')
    if bprobe and bprobe['tps']: L.append(f"\n#2 char-config speed probe (3 min, right before the 1-h #4 run): {bprobe['tps']:,.0f} tok(=char)/s")
    L += ['', 'Tokenizer survey, gradient checks: `04-bpe/survey.json`, `04-bpe/gradcheck.log`.', '']
if eprobe and eprobe['tps']: L.append(f"#4 fp32 speed probe (3 min, right before the 1-h #5 run): {eprobe['tps']:,.0f} tokens/s\n")
if sizes:
    def params(r):
        m = re.search(r'params=(\d+)', open(r['tag'] + '.log').read()); return int(m[1]) if m else 0
    L += ['## #6 model-size sweep (1 h each, base #5; nats/char)', '',
          '| model | params | val 1M at 15/30/45/60 min | full val | tok/s | steps | passes over train | train(last 10%) - full val | 3-min #5-config probe before |',
          '|---|---|---|---|---|---|---|---|---|']
    allr = [('#5 d256-L6', ew, '05-precision/speed-probe-fp32')] + [(n, r, f'06-size/probe-before-{n}') for n, r in sizes]
    for n, r, pr in allr:
        if r is None: continue
        v1 = ' / '.join(f"{x['val_1m']:.4f}" for x in r['v1m']) or '-'; fv = r['fin'][0]['val_full10pct'] if r['fin'] else None
        o = overfit(r); gap = f"{o[0] - fv:+.3f}" if (o and fv) else '-'
        passes = f"{r['steps'] * 4096 / 30476268:.2f}" if r['steps'] else '-'
        prb = run(pr) if os.path.exists(pr + '.log') else None
        L.append(f"| {n} | {params(r)/1e6:.2f}M | {v1} | {fv if fv is None else f'{fv:.4f}'} | {r['tps'] or 0:,.0f} | {r['steps'] or '-'} | {passes} | {gap} | {(prb['tps'] if prb and prb['tps'] else 0):,.0f} tok/s |")
    L.append('')
if os.path.exists('06-size/NOTES.md'): L += [open('06-size/NOTES.md').read(), '']
vocs = [(V, run(f'07-vocab/v{V}')) for V in (512, 1024, 4096, 8192) if os.path.exists(f'07-vocab/v{V}.log')]
if vocs:
    import json as _j
    vc = {512: 'tab:cyan', 1024: 'tab:blue', 4096: 'tab:orange', 8192: 'tab:brown'}
    chart(None, ew, '', '07-vocab/chart.png', extra=tuple((r, f'#7 vocab {V}', vc[V]) for V, r in vocs), blabel='#5 vocab 2048')
    L += ['## #7 BPE vocab sweep (d256-L6, ctx 64 tokens, 1 h each, base #5; nats/char)', '',
          '| vocab | chars/token train / val | chars per 64-token window | params | val 1M at 15/30/45/60 min | full val | tok/s | chars/s | steps | full val - train(last 10%) | #5-config probe before |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    for V, r, mf, pr in [(2048, ew, '04-bpe/meta2048.json', '05-precision/speed-probe-fp32')] + [(V, r, f'07-vocab/meta{V}.json', f'07-vocab/probe-before-v{V}') for V, r in vocs]:
        m = _j.load(open(mf)); ct = m['chars_per_token_train']
        v1 = ' / '.join(f"{x['val_1m']:.4f}" for x in r['v1m']) or '-'; fv = r['fin'][0]['val_full10pct'] if r['fin'] else None
        o = overfit(r); gap = f"{fv - o[0]:+.3f}" if (o and fv) else '-'
        prb = run(pr) if os.path.exists(pr + '.log') else None
        pm = re.search(r'params=(\d+)', open(r['tag'] + '.log').read())
        L.append(f"| {V}{' (#5)' if V == 2048 else ''} | {ct:.2f} / {m['chars_per_token_val']:.2f} | {64*ct:.0f} | {int(pm[1])/1e6:.2f}M | {v1} | {'-' if fv is None else f'{fv:.4f}'} | {r['tps'] or 0:,.0f} | {(r['tps'] or 0)*ct:,.0f} | {r['steps'] or '-'} | {gap} | {(prb['tps'] if prb and prb['tps'] else 0):,.0f} tok/s |")
    L.append('')
if os.path.exists('07-vocab/NOTES.md'): L += [open('07-vocab/NOTES.md').read(), '']
ups = [(V, run(f'08-vocab-up/v{V}')) for V in (16384, 32768, 65536) if os.path.exists(f'08-vocab-up/v{V}.log')]
if ups:
    import json as _j
    b8 = run('07-vocab/v8192'); uc = {16384: 'tab:purple', 32768: 'tab:green', 65536: 'tab:red'}
    chart(None, b8, '', '08-vocab-up/chart.png', extra=tuple((r, f'#8 vocab {V}', uc[V]) for V, r in ups), blabel='#7 vocab 8192')
    L += ['## #8 vocab sweep upward (conditional; d256-L6, ctx 64 tokens, 1 h each, base v8192; nats/char)', '',
          '| vocab | chars/token train / val | chars per window | params | val 1M at 15/30/45/60 min | full val | tok/s | chars/s | steps | full val - train(last 10%) | #5-config probe before |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    for V, r, mf, pr in [(8192, b8, '07-vocab/meta8192.json', '07-vocab/probe-before-v8192')] + [(V, r, f'08-vocab-up/meta{V}.json', f'08-vocab-up/probe-before-v{V}') for V, r in ups]:
        m = _j.load(open(mf)); ct = m['chars_per_token_train']
        v1 = ' / '.join(f"{x['val_1m']:.4f}" for x in r['v1m']) or '-'; fv = r['fin'][0]['val_full10pct'] if r['fin'] else None
        o = overfit(r); gap = f"{fv - o[0]:+.3f}" if (o and fv) else '-'
        prb = run(pr) if os.path.exists(pr + '.log') else None
        pm = re.search(r'params=(\d+)', open(r['tag'] + '.log').read())
        L.append(f"| {V}{' (#7 best)' if V == 8192 else ''} | {ct:.2f} / {m['chars_per_token_val']:.2f} | {64*ct:.0f} | {int(pm[1])/1e6:.2f}M | {v1} | {'-' if fv is None else f'{fv:.4f}'} | {r['tps'] or 0:,.0f} | {(r['tps'] or 0)*ct:,.0f} | {r['steps'] or '-'} | {gap} | {(prb['tps'] if prb and prb['tps'] else 0):,.0f} tok/s |")
    L.append('')
if os.path.exists('08-vocab-up/NOTES.md'): L += [open('08-vocab-up/NOTES.md').read(), '']
if os.path.exists('09-dropout/v16384-drop0.1.log'):
    d0 = run('08-vocab-up/v16384'); d1 = run('09-dropout/v16384-drop0.1')
    chart(d1, d0, '#9 v16384 dropout 0.1', '09-dropout/chart.png', blabel='#8 v16384 dropout 0')
    L += ['## #9 dropout 0.1 on v16384 (otherwise identical, 1 h; nats/char)', '',
          '| run | val 1M at 15/30/45/60 min | full val | tok/s | steps | full val - train(last 10%) |', '|---|---|---|---|---|---|']
    for n, r in (('v16384 dropout 0', d0), ('v16384 dropout 0.1', d1)):
        v1 = ' / '.join(f"{x['val_1m']:.4f}" for x in r['v1m']) or '-'; fv = r['fin'][0]['val_full10pct'] if r['fin'] else None
        o = overfit(r); gap = f"{fv - o[0]:+.3f}" if (o and fv) else '-'
        L.append(f"| {n} | {v1} | {'-' if fv is None else f'{fv:.4f}'} | {r['tps'] or 0:,.0f} | {r['steps'] or '-'} | {gap} |")
    L += ['', 'Note: with dropout the train-batch loss is measured with dropout on, which raises it and shrinks the apparent gap; the gaps are not comparable.', '']
if os.path.exists('09-dropout/NOTES.md'): L += [open('09-dropout/NOTES.md').read(), '']
if os.path.exists('10-long/v16384-const-long.csv'):
    import subprocess; subprocess.run(['python3', '10-long/report10.py'])
    L += ['## #10 long constant-LR run: when does v16384 d256-L6 overfit? (nats/char)', '',
          'Constant LR 1e-3 / Muon 0.01 after a 36 s warmup, no decay. Val-1M, full 10M val and recent train loss every 30 min of training time. 12 h wall cap; early stop if the last 3 full-val evals are all > best + 0.003. Chart: `10-long/chart.png`.', '']
    if os.path.exists('10-long/summary.md'): L += [open('10-long/summary.md').read(), '']
if os.path.exists('10-long/NOTES.md'): L += [open('10-long/NOTES.md').read(), '']
if os.path.exists('11-decay/v16384-decay-final.csv'):
    import subprocess; subprocess.run(['python3', '11-decay/report11.py'])
    L += ['## #11 short linear LR decay from the #10 checkpoint (nats/char)', '',
          'Resumed 10-long/v16384-const-ckpt-097608.f32 (weights only; full val 1.0605) and decayed LR linearly from peak (Muon 0.01 / AdamW 1e-3) to 0 over 720 s of training time (= the 20% WSD decay of the 1 h runs). Chart: `11-decay/chart.png`.', '',
          '| point | step | train s | val 1M | full val |', '|---|---|---|---|---|',
          '| start (#10 ckpt) | 0 | 0 | 0.9376 | 1.0605 |', '| midpoint | 1018 | 360 | 0.9206 | 1.0388 |', '| end | 1966 | 720 | 0.8982 | **1.0155** |', '']
if os.path.exists('11-decay/NOTES.md'): L += [open('11-decay/NOTES.md').read(), '']
if os.path.exists('12-decay-from-earlier/v16384-decay-5h-final.csv'):
    import subprocess; subprocess.run(['python3', '12-decay-from-earlier/report12.py'])
    L += ['## #12 the same 720 s linear decay started from earlier #10 checkpoints (nats/char)', '',
          'Same settings as #11 (init= + sched=lin, peak to 0 over 720 s of training time, seed 11); final weights saved. Chart: `12-decay-from-earlier/chart.png`.', '',
          '| decay start | #10 full val there | midpoint full val | final full val | vs 1 h WSD 1.0480 | full - train (end) |', '|---|---|---|---|---|---|',
          '| 2 h (step 20,410) | 1.0815 | 1.0578 | 1.0332 | -0.0148 | +0.083 |',
          '| 5 h (step 50,960) | 1.0679 | 1.0440 | 1.0209 | -0.0271 | +0.081 |',
          '| 10 h (step 97,608, #11) | 1.0605 | 1.0388 | **1.0155** | -0.0325 | +0.086 |', '']
if os.path.exists('12-decay-from-earlier/NOTES.md'): L += [open('12-decay-from-earlier/NOTES.md').read(), '']
if os.path.exists('13-cyclic/v16384-cyc-r1-final.csv'):
    import subprocess; subprocess.run(['python3', '13-cyclic/report13.py'])
    L += ['## #13 cyclic warm restarts: 5 x (2 h constant + 720 s linear decay) in one 11 h run (nats/char)', '',
          'Interrupted by a box VM restart at 6.0 training h and resumed once from ckpt 056223 (weights only; optimizer moments reset). Chart: `13-cyclic/chart.png`.', '',
          '| cycle end | training h | step | full val | branch decay (#11/#12) | vs 1 h WSD 1.0480 | full - train |', '|---|---|---|---|---|---|---|',
          '| 1 | 2.2 | 20,907 | 1.0317 | 2 h: 1.0332 | -0.0163 | +0.058 |',
          '| 2 | 4.4 | 41,833 | 1.0227 | 5 h: 1.0209 | -0.0253 | +0.063 |',
          '| 3 (after resume) | 6.6 | 61,500 | 1.0212 | | -0.0268 | +0.056 |',
          '| 4 | 8.8 | 81,059 | 1.0164 | | -0.0316 | +0.067 |',
          '| 5 (final) | 11.0 | 100,604 | **1.0152** | 10 h: 1.0155 | -0.0328 | +0.078 |', '']
if os.path.exists('13-cyclic/NOTES.md'): L += [open('13-cyclic/NOTES.md').read(), '']
if os.path.exists('05-precision/NOTES.md'): L += [open('05-precision/NOTES.md').read(), '']
if os.path.exists('04-bpe/NOTES.md'): L += [open('04-bpe/NOTES.md').read(), '']
if os.path.exists('03-ctx/NOTES.md'): L += [open('03-ctx/NOTES.md').read(), '']
if os.path.exists('02-muon/NOTES.md'): L += [open('02-muon/NOTES.md').read(), '']
if os.path.exists('01-lr/NOTES.md'): L += [open('01-lr/NOTES.md').read(), '']
for name, r in (('baseline', base), ('#1', lr1)):
    if r and r['sample']: L += [f'### Sample ({name})', '```', r['sample'][:700], '```', '']
open('REPORT.md', 'w').write('\n'.join(L))
print('\n'.join(L[:16]))
