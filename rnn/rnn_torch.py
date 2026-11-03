# PyTorch replica of rnn.lisp: vanilla char RNN, same data/streams/TBPTT/Adam/clip/init/eval.
# Usage: python rnn_torch.py SECONDS [H] [threads]
import sys, time, math, torch, torch.nn.functional as F
secs_budget = float(sys.argv[1]); H = int(sys.argv[2]) if len(sys.argv) > 2 else 256
threads = int(sys.argv[3]) if len(sys.argv) > 3 else 1
torch.set_num_threads(threads); torch.manual_seed(1); dt = torch.float64

text = open("../input.txt", encoding="utf-8").read()[:300000]
chars = sorted(set(text)); V = len(chars); c2i = {c: i for i, c in enumerate(chars)}
data = torch.tensor([c2i[c] for c in text]); N = len(data); split = int(0.9 * N)
B, T = 32, 32
slen = split // B; train_streams = data[:B * slen].view(B, slen)            # B contiguous streams
VB = 32; vlen = (N - split) // VB; val_streams = data[split:split + VB * vlen].view(VB, vlen)

E = (0.1 * torch.randn(V, H, dtype=dt)).requires_grad_()
Whh = (torch.randn(H, H, dtype=dt) / math.sqrt(H)).requires_grad_()
bh = torch.zeros(H, dtype=dt, requires_grad=True)
Why = (torch.randn(H, V, dtype=dt) / math.sqrt(H)).requires_grad_()
by = torch.zeros(V, dtype=dt, requires_grad=True)
params = [E, Whh, bh, Why, by]
opt = torch.optim.Adam(params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8)

def run(xs, h):  # xs [b, t] ids -> logits [b, t, V], final h
    outs = []
    for s in range(xs.shape[1]):
        h = torch.tanh(E[xs[:, s]] + h @ Whh + bh); outs.append(h)
    return torch.stack(outs, 1) @ Why + by, h

def stream_loss(streams):  # carry h through each stream, score all but first char
    with torch.no_grad():
        logits, _ = run(streams[:, :-1], torch.zeros(streams.shape[0], H, dtype=dt))
        return F.cross_entropy(logits.reshape(-1, V), streams[:, 1:].reshape(-1)).item()

print(f"PyTorch {torch.__version__} threads={threads} vanilla RNN: vocab={V} hidden={H} params={sum(p.numel() for p in params)}, "
      f"B={B} T={T}, Adam lr 2e-3, clip 1.0, budget {secs_budget:g}s", flush=True)
t0 = time.perf_counter(); step = 0; pos = 0; run_loss = 0.0; nrun = 0
h = torch.zeros(B, H, dtype=dt)
while time.perf_counter() - t0 < secs_budget:
    if pos + T + 1 > slen: pos = 0; h = torch.zeros(B, H, dtype=dt)          # new epoch
    x = train_streams[:, pos:pos + T]; y = train_streams[:, pos + 1:pos + T + 1]
    logits, hT = run(x, h.detach())
    loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
    h = hT; pos += T; step += 1; run_loss += loss.item(); nrun += 1
    if step % 100 == 0:
        print(f"step {step:5d}  t={time.perf_counter()-t0:6.1f}s  train(batch avg)={run_loss/nrun:.4f}  val={stream_loss(val_streams):.4f}", flush=True)
        run_loss = 0.0; nrun = 0
el = time.perf_counter() - t0
print(f"trained {step} steps in {el:.1f}s ({step/el:.1f} steps/s, {step*B*T/el:.0f} chars/s)")
print(f"final: train={stream_loss(train_streams):.4f} (32 streams over full train split) val={stream_loss(val_streams):.4f}")
g = torch.Generator().manual_seed(0); c = c2i["\n"]; hs = torch.zeros(1, H, dtype=dt); out = []
with torch.no_grad():
    for _ in range(300):
        hs = torch.tanh(E[c] + hs @ Whh + bh)
        c = torch.multinomial(F.softmax((hs @ Why + by)[0] / 0.8, -1), 1, generator=g).item(); out.append(chars[c])
print("\n=== PyTorch RNN sample (T=0.8) ===\n" + "".join(out))
