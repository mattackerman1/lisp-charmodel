# #18 args. V16/V32 = model+data args per vocab (BASE from 17-method/common.sh is V16). CPT = train chars/token.
M="d=256 layers=6 heads=8 ctx=64 batch=64 threads=8 dropout=0 lr=1e-3 optim=muon mlr=0.01 ew=f32"
V16="tokens=08-vocab-up/tokens16384.u16 vocab=08-vocab-up/vocab16384.hex vstart=23505859 fullval-targets=238949 $M"; CPT16=3.82458
V32="tokens=08-vocab-up/tokens32768.u16 vocab=08-vocab-up/vocab32768.hex vstart=22432474 fullval-targets=229892 $M"; CPT32=4.00758
SCH="sched=wsd warm=0.01 decay=0.2 lrmin=0"
