# PyTorch replica of ../charmodel.lisp (ctx 8, emb 16, hidden 200 tanh, float64).
# Usage: python charmodel_torch.py time SECONDS [threads]   |   python charmodel_torch.py steps N [threads]
import sys, time, math, torch, torch.nn as nn, torch.nn.functional as F
mode, limit = sys.argv[1], float(sys.argv[2])
threads = int(sys.argv[3]) if len(sys.argv) > 3 else 1
torch.set_num_threads(threads); torch.manual_seed(42)
CTX, EMB, HID, BS = 8, 16, 200, 32

# Data: first 300k chars, sorted vocab, 90/10 split (identical to the Lisp code)
text = open("../input.txt", encoding="utf-8").read()[:300000]
chars = sorted(set(text)); V = len(chars); c2i = {c: i for i, c in enumerate(chars)}
data = torch.tensor([c2i[c] for c in text]); N = len(data); split = int(0.9 * N)
# Context matrix: row p = previous CTX ids before position p, newline-padded at the start
padded = torch.cat([torch.full((CTX,), c2i["\n"]), data])
ctxs = torch.stack([padded[k:k + N] for k in range(CTX)], 1)

class MLP(nn.Module):
    def __init__(s):
        super().__init__()
        s.C = nn.Parameter(torch.randn(V, EMB, dtype=torch.float64))                          # N(0,1)
        s.W1 = nn.Parameter(torch.randn(CTX * EMB, HID, dtype=torch.float64) / math.sqrt(CTX * EMB))
        s.b1 = nn.Parameter(torch.zeros(HID, dtype=torch.float64))
        s.W2 = nn.Parameter(torch.randn(HID, V, dtype=torch.float64) * 0.1 / math.sqrt(HID))
        s.b2 = nn.Parameter(torch.zeros(V, dtype=torch.float64))
    def forward(s, idx):  # idx [B, CTX] -> logits [B, V]
        return torch.tanh(s.C[idx].reshape(len(idx), -1) @ s.W1 + s.b1) @ s.W2 + s.b2
m = MLP(); params = sum(p.numel() for p in m.parameters())
opt = torch.optim.SGD(m.parameters(), lr=0.1)

def eval_loss(a, b, stride=1):
    with torch.no_grad():
        pos = torch.arange(a, b, stride); return F.cross_entropy(m(ctxs[pos]), data[pos]).item()

print(f"PyTorch {torch.__version__} threads={threads} vocab={V} train={split} val={N-split} params={params} mode={mode} {limit:g}", flush=True)
t0 = time.perf_counter(); step = 0; run = 0.0; nrun = 0
while True:
    el = time.perf_counter() - t0
    frac = el / limit if mode == "time" else step / limit
    if frac >= 1: break
    for g in opt.param_groups: g["lr"] = 0.1 if frac < 0.75 else 0.01      # drop for last 25%
    pos = torch.randint(1, split, (BS,))
    loss = F.cross_entropy(m(ctxs[pos]), data[pos])
    opt.zero_grad(); loss.backward(); opt.step()
    step += 1; run += loss.item(); nrun += 1
    if step % 2000 == 0:
        print(f"step {step:6d}  t={time.perf_counter()-t0:6.1f}s  batch-avg={run/nrun:.4f}  train={eval_loss(0, split, 9):.4f}  val={eval_loss(split, N):.4f}", flush=True)
        run = 0.0; nrun = 0
secs = time.perf_counter() - t0
print(f"trained {step} steps in {secs:.1f}s ({step/secs:.0f} steps/s); final train={eval_loss(0, split):.4f} val={eval_loss(split, N):.4f} (full train eval)")

# Sample 300 chars at T=0.8, starting from newline context
g = torch.Generator().manual_seed(0); ctx = [c2i["\n"]] * CTX; out = []
with torch.no_grad():
    for _ in range(300):
        p = F.softmax(m(torch.tensor([ctx])) / 0.8, -1)[0]
        c = torch.multinomial(p, 1, generator=g).item(); out.append(chars[c]); ctx = ctx[1:] + [c]
print("\n=== PyTorch MLP sample (T=0.8) ===\n" + "".join(out))
