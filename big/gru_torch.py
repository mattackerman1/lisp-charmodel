# gru_torch.py -- PyTorch twin of gru-big.lisp (GRU H=288, non-recurrent dropout, Adam + warmup/cosine, TBPTT T=32 x 32 streams).
# Same equations/gate order (z, r, n), same init, same dropout placement (on the 3H input projection Ein[x] and on h before
# the output layer, never on the recurrence), same data streams, same stream-carried eval, same timing rules (eval excluded).
import sys, time, math, torch, torch.nn.functional as F
args = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a); opt = lambda k, d: args.get(k, d)
torch.manual_seed(int(opt('seed', '1'))); torch.set_num_threads(int(opt('threads', '8')))
H, LR0, PD, BUDGET = int(opt('h', 288)), float(opt('lr', 2e-3)), float(opt('dropout', 0.25)), float(opt('secs', 480))
FEVERY, TAG, WARM, BT, TT = float(opt('fullval-every', 0)), opt('tag', 'grutorch'), 0.02, 32, 32
text = open(opt('data', 'corpus.txt'), encoding='utf-8').read()
chars = sorted(set(text)); V = len(chars); c2i = {c: i for i, c in enumerate(chars)}
data = torch.tensor([c2i[c] for c in text], dtype=torch.long); N = len(data); VSTART = int(math.floor(0.9 * N))
SLEN, VLEN, G = VSTART // BT, (N - VSTART) // BT, 3 * H
s = 1 / math.sqrt(H)
Ein = (torch.randn(V, G) * 0.1).requires_grad_(); bx = torch.zeros(G, requires_grad=True)
Wh = (torch.randn(H, G) * s).requires_grad_(); bh = torch.zeros(G, requires_grad=True)
Why = (torch.randn(H, V) * s).requires_grad_(); by = torch.zeros(V, requires_grad=True)
params = [Ein, bx, Wh, bh, Why, by]; nparams = sum(p.numel() for p in params)
optim = torch.optim.Adam(params, lr=LR0, betas=(0.9, 0.999), eps=1e-8)
def lr_at(fr): return LR0 * max(0.01, fr / WARM) if fr < WARM else 0.1 * LR0 + 0.9 * LR0 * 0.5 * (1 + math.cos(math.pi * min(1.0, (fr - WARM) / (1 - WARM))))
def chunk(x, h, train):
    """x: (T,B) ids. Returns logits (T,B,V) and final h."""
    xg = Ein[x]
    if train: xg = F.dropout(xg, PD, True)              # dropout on the input projection Ein[x] (3H per position)
    xg = xg + bx; hs = []
    for t in range(x.shape[0]):
        hg = h @ Wh + bh
        z = torch.sigmoid(xg[t, :, :H] + hg[:, :H]); r = torch.sigmoid(xg[t, :, H:2*H] + hg[:, H:2*H])
        n = torch.tanh(xg[t, :, 2*H:] + r * hg[:, 2*H:]); h = (1 - z) * n + z * h; hs.append(h)
    Hs = torch.stack(hs)
    if train: Hs = F.dropout(Hs, PD, True)              # dropout on h before the output layer (recurrence uses undropped h)
    return Hs @ Why + by, h
ar = torch.arange(TT)
def stream_loss(base, length):
    st = base + torch.arange(BT) * length; h = torch.zeros(BT, H); tot, cnt = 0.0, 0
    with torch.no_grad():
        for p in range(0, length - 1, TT):
            nt = min(TT, length - 1 - p); idx = (st[None, :] + p + torch.arange(nt)[:, None])
            lg, h = chunk(data[idx], h, False)
            tot += F.cross_entropy(lg.reshape(-1, V), data[idx + 1].reshape(-1), reduction='sum').item(); cnt += idx.numel()
    return tot / cnt
def wcsv(fn, hdr, rows):
    with open(fn, 'w') as f: f.write(hdr + '\n'); [f.write(','.join(map(str, r)) + '\n') for r in rows]
print(f"config: PyTorch {torch.__version__} float32 threads={torch.get_num_threads()} | GRU H={H} vocab={V} params={nparams} | B=32 T=32 Adam lr {LR0} warmup+cosine clip 1 | budget {BUDGET:.0f}s training time (eval excluded)")
print(f"data {opt('data','corpus.txt')} | train chars {VSTART}, val chars {N-VSTART} | non-recurrent dropout={PD} | full val every {FEVERY}s", flush=True)
hist, fhist, step, evs, t0, fnext, pos = [], [], 0, 0.0, time.time(), FEVERY, 0
tsecs = lambda: time.time() - t0 - evs
h = torch.zeros(BT, H); starts = torch.arange(BT) * SLEN
while (el := tsecs()) < BUDGET:
    if pos + TT + 1 > SLEN: pos = 0; h = torch.zeros(BT, H)
    for g in optim.param_groups: g['lr'] = lr_at(el / BUDGET)
    idx = starts[None, :] + pos + ar[:, None]
    lg, h = chunk(data[idx], h, True)
    loss = F.cross_entropy(lg.reshape(-1, V), data[idx + 1].reshape(-1))
    optim.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); optim.step()
    h = h.detach(); pos += TT; step += 1
    if step % 200 == 0:
        e0 = time.time(); vl = stream_loss(VSTART, 1000); evs += time.time() - e0
        print(f"step {step:6d}  t={tsecs():7.1f}s  wall={time.time()-t0:7.1f}s  lr={lr_at(tsecs()/BUDGET):.6f}  train(last batch)={loss.item():.4f}  val(subset)={vl:.4f}", flush=True)
        hist.append((step, round(tsecs(), 3), loss.item(), vl))
        if step % 2000 == 0: wcsv(f"{TAG}.csv", "step,seconds,train_batch,val_subset", hist)
    if FEVERY > 0 and tsecs() >= fnext and tsecs() < BUDGET:
        fnext += FEVERY; ts = tsecs(); e0 = time.time(); vf = stream_loss(VSTART, VLEN); evs += time.time() - e0
        print(f"FULLVAL step {step}  t={ts:.1f}s  val(full)={vf:.4f}", flush=True); fhist.append((step, round(ts, 3), vf)); wcsv(f"{TAG}-full.csv", "step,train_seconds,val_full", fhist)
s_ = tsecs()
print(f"trained {step} steps in {s_:.1f}s training time (wall {time.time()-t0:.1f}s, eval {evs:.1f}s): {step/s_:.2f} steps/s, {step*BT*TT/s_:.0f} tokens/s (training time)", flush=True)
vf = stream_loss(VSTART, VLEN); print(f"FINAL: val(full fixed val, 32 streams x {VLEN}, state carried)={vf:.4f}  train(32x3000 subset)={stream_loss(0, 3000):.4f}", flush=True)
fhist.append((step, round(s_, 3), vf)); wcsv(f"{TAG}-full.csv", "step,train_seconds,val_full", fhist); wcsv(f"{TAG}.csv", "step,seconds,train_batch,val_subset", hist)
torch.save({'params': params}, f"{TAG}.pt")
c, h, out = c2i['\n'], torch.zeros(1, H), []
with torch.no_grad():
    for _ in range(300):
        lg, h = chunk(torch.tensor([[c]]), h, False); p = F.softmax(lg[0, 0], -1).double() ** 1.25
        c = torch.multinomial(p / p.sum(), 1).item(); out.append(chars[c])
print("\n=== sample (T=0.8) ===\n" + ''.join(out))
