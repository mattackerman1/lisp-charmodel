BASE="tokens=08-vocab-up/tokens16384.u16 vocab=08-vocab-up/vocab16384.hex vstart=23505859 fullval-targets=238949 d=256 layers=6 heads=8 ctx=64 batch=64 threads=8 dropout=0 lr=1e-3 optim=muon mlr=0.01 ew=f32"
PROXY="$BASE sched=wsd warm=0.01 decay=0.2 lrmin=0 steps=2500"
DECAY="$BASE sched=lin warm=0 init=10-long/v16384-const-ckpt-020410.f32 steps=1868"
