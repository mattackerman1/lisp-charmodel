# gpt_torch.py (huge/) -- uint8 data loading + fullval-targets= cap on periodic val; otherwise identical to big/gpt_torch.py.
# gpt_torch.py -- PyTorch twin of gpt-big.lisp (same architecture, init, hyperparameters, data split, eval protocol, timing rules).
# Pre-LN GPT: x += Drop(Proj(Attn(LN1 x))); x += Drop(FC2(GELU_tanh(FC(LN2 x)))); final LN; head tied to wte (no bias).
# Dropout on embeddings (wte+wpe), attention probs, attn-proj output, MLP output. float32, CPU.
import sys, time, math, csv, torch, torch.nn as nn, torch.nn.functional as F
args = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a)
opt = lambda k, d: args.get(k, d)
torch.manual_seed(int(opt('seed', '1'))); torch.set_num_threads(int(opt('threads', '8')))
D, NL, NH, TT, BT = int(opt('d', 128)), int(opt('layers', 4)), int(opt('heads', 4)), int(opt('ctx', 64)), int(opt('batch', 32))
LR0, WD, PD, BUDGET = float(opt('lr', 1e-3)), float(opt('wd', 0.1)), float(opt('dropout', 0.1)), float(opt('secs', 480))
FEVERY, TAG, WARM = float(opt('fullval-every', 0)), opt('tag', 'torch'), 0.02
import numpy as np
raw = np.frombuffer(open(opt('data', 'corpus.txt'), 'rb').read(), dtype=np.uint8)   # ASCII corpus
codes = np.unique(raw); chars = [chr(c) for c in codes]; V = len(chars); c2i = {c: i for i, c in enumerate(chars)}
lut = np.zeros(256, dtype=np.uint8); lut[codes] = np.arange(V, dtype=np.uint8)
data = torch.from_numpy(lut[raw].copy()); N = len(data); VSTART = int(math.floor(0.9 * N)); HALF = TT // 2   # uint8; .long() per batch
FVT = int(opt('fullval-targets', 10**12))

class Block(nn.Module):
    def __init__(s):
        super().__init__()
        s.ln1, s.ln2 = nn.LayerNorm(D, eps=1e-5), nn.LayerNorm(D, eps=1e-5)
        s.qkv, s.proj, s.fc, s.fc2 = nn.Linear(D, 3 * D), nn.Linear(D, D), nn.Linear(D, 4 * D), nn.Linear(4 * D, D)
    def forward(s, x):
        B, T, _ = x.shape
        q, k, v = s.qkv(s.ln1(x)).split(D, dim=2)
        q, k, v = [z.view(B, T, NH, D // NH).transpose(1, 2) for z in (q, k, v)]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=PD if s.training else 0.0)  # dropout on attn probs
        x = x + F.dropout(s.proj(y.transpose(1, 2).reshape(B, T, D)), PD, s.training)
        return x + F.dropout(s.fc2(F.gelu(s.fc(s.ln2(x)), approximate='tanh')), PD, s.training)

class GPT(nn.Module):
    def __init__(s):
        super().__init__()
        s.wte, s.wpe = nn.Parameter(torch.empty(V, D)), nn.Parameter(torch.empty(TT, D))
        s.blocks = nn.ModuleList(Block() for _ in range(NL)); s.lnf = nn.LayerNorm(D, eps=1e-5)
        rstd = 0.02 / math.sqrt(2 * NL)                       # same init scales as gpt.lisp
        for n, p in s.named_parameters():
            if n in ('wte', 'wpe') or n.endswith(('qkv.weight', 'fc.weight')): nn.init.normal_(p, 0, 0.02)
            elif n.endswith(('proj.weight', 'fc2.weight')): nn.init.normal_(p, 0, rstd)
            elif 'ln' in n and n.endswith('weight'): nn.init.ones_(p)
            else: nn.init.zeros_(p)
    def forward(s, idx):
        x = F.dropout(s.wte[idx] + s.wpe[:idx.shape[1]], PD, s.training)
        for b in s.blocks: x = b(x)
        return s.lnf(x) @ s.wte.t()

m = GPT(); nparams = sum(p.numel() for p in m.parameters())
dec = [p for n, p in m.named_parameters() if n in ('wte', 'wpe') or n.endswith(('qkv.weight', 'proj.weight', 'fc.weight', 'fc2.weight'))]
nodec = [p for n, p in m.named_parameters() if not (n in ('wte', 'wpe') or n.endswith(('qkv.weight', 'proj.weight', 'fc.weight', 'fc2.weight')))]
optim = torch.optim.AdamW([{'params': dec, 'weight_decay': WD}, {'params': nodec, 'weight_decay': 0.0}], lr=LR0, betas=(0.9, 0.99), eps=1e-8)
def lr_at(fr): return LR0 * max(0.01, fr / WARM) if fr < WARM else 0.1 * LR0 + 0.9 * LR0 * 0.5 * (1 + math.cos(math.pi * min(1.0, (fr - WARM) / (1 - WARM))))
ar = torch.arange(TT)
def val_loss(ntargets):
    starts, s = [], VSTART - 1 - HALF
    while s + HALF + 1 < VSTART + ntargets and s + TT + 1 <= N: starts.append(s); s += HALF
    m.eval(); tot, cnt = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(starts), BT):
            st = torch.tensor(starts[i:i + BT])[:, None] + ar
            lg = m(data[st].long())[:, HALF:]; tg = data[st + 1][:, HALF:].long()
            tot += F.cross_entropy(lg.reshape(-1, V), tg.reshape(-1), reduction='sum').item(); cnt += tg.numel()
    m.train(); return tot / cnt, cnt
print(f"dropout={PD} (emb, attn probs, attn/MLP residual outputs; off at eval)")
print(f"config: PyTorch {torch.__version__} GPT d={D} layers={NL} heads={NH} ctx={TT} vocab={V} params={nparams} | batch {BT} seqs ({BT*TT} tokens), threads={torch.get_num_threads()}, float32")
print(f"AdamW lr={LR0} betas 0.9/0.99 wd={WD} clip 1.0, warmup 2% then cosine to lr/10 over {BUDGET:.0f}s of training time (eval excluded) | data {opt('data','corpus.txt')} | train chars {VSTART}, val chars {N-VSTART} (last 10%)", flush=True)
hist, fhist, step, evs, t0, fnext = [], [], 0, 0.0, time.time(), FEVERY
tsecs = lambda: time.time() - t0 - evs
def wcsv(fn, hdr, rows):
    with open(fn, 'w') as f: f.write(hdr + '\n'); [f.write(','.join(map(str, r)) + '\n') for r in rows]
m.train()
while (el := tsecs()) < BUDGET:
    for g in optim.param_groups: g['lr'] = lr_at(el / BUDGET)
    st = torch.randint(0, VSTART - TT - 1, (BT,))[:, None] + ar
    loss = F.cross_entropy(m(data[st].long()).reshape(-1, V), data[st + 1].long().reshape(-1))
    optim.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); optim.step(); step += 1
    if step % 200 == 0:
        e0 = time.time(); vl, _ = val_loss(32000); evs += time.time() - e0
        print(f"step {step:6d}  t={tsecs():7.1f}s  wall={time.time()-t0:7.1f}s  lr={lr_at(tsecs()/BUDGET):.6f}  train(last batch)={loss.item():.4f}  val(subset)={vl:.4f}", flush=True)
        hist.append((step, round(tsecs(), 3), loss.item(), vl)); wcsv(f"{TAG}.csv", "step,seconds,train_batch,val_subset", hist)
    if FEVERY > 0 and tsecs() >= fnext and tsecs() < BUDGET:
        fnext += FEVERY; ts = tsecs(); e0 = time.time(); vf, _ = val_loss(min(N, FVT)); evs += time.time() - e0
        print(f"FULLVAL step {step}  t={ts:.1f}s  val(full{'' if FVT >= N else ', first %d targets' % FVT})={vf:.4f}", flush=True); fhist.append((step, round(ts, 3), vf)); wcsv(f"{TAG}-full.csv", "step,train_seconds,val_full", fhist)
s = tsecs()
print(f"trained {step} steps in {s:.1f}s training time (wall {time.time()-t0:.1f}s, eval {evs:.1f}s): {step/s:.2f} steps/s, {step*BT*TT/s:.0f} tokens/s (training time)", flush=True)
vf, cnt = val_loss(N); print(f"FINAL: val(full fixed val, {cnt} targets scored)={vf:.4f}"); fhist.append((step, round(s, 3), vf)); wcsv(f"{TAG}-full.csv", "step,train_seconds,val_full", fhist)
torch.save(m.state_dict(), f"{TAG}.pt")
m.eval(); ctx = [c2i['\n']]; out = []
with torch.no_grad():
    for _ in range(300):
        p = F.softmax(m(torch.tensor([ctx[-TT:]]))[0, -1], -1).double() ** 1.25
        c = torch.multinomial(p / p.sum(), 1).item(); ctx.append(c); out.append(chars[c])
print("\n=== sample (T=0.8) ===\n" + ''.join(out))
