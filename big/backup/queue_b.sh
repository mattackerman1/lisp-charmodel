#!/bin/bash
# Phase B: Lisp GRU then PyTorch GRU, 3600 s training time each, sequential, 8 threads, nice -n 10.
cd /home/box/lisp-charmodel/big
chk() { if pgrep -f "science-team|[Ee]xperimenter" >/dev/null; then echo "$(date '+%F %T %Z') WARNING science/experimenter job running; using 4 threads" >> queue_b.log; echo 4; else echo 8; fi; }
T=$(chk); echo "$(date '+%F %T %Z') start GRU-LISP threads=$T load=$(cut -d' ' -f1-3 /proc/loadavg)" >> queue_b.log
grep -q "=== sample" gru-lisp.log 2>/dev/null || FT=single MM=blas nice -n 10 sbcl --script gru-big.lisp data=corpus.txt blas-threads=1 h=288 secs=3600 sched=cosine dropout=0.25 fullval-every=600 threads=$T tag=gru-lisp > gru-lisp.log 2>&1
echo "$(date '+%F %T %Z') end GRU-LISP" >> queue_b.log
T=$(chk); echo "$(date '+%F %T %Z') start GRU-TORCH threads=$T load=$(cut -d' ' -f1-3 /proc/loadavg)" >> queue_b.log
grep -q "=== sample" gru-torch.log 2>/dev/null || nice -n 10 /home/box/venvs/torch/bin/python gru_torch.py data=corpus.txt h=288 secs=3600 dropout=0.25 fullval-every=600 threads=$T tag=gru-torch > gru-torch.log 2>&1
echo "$(date '+%F %T %Z') end GRU-TORCH" >> queue_b.log
echo "$(date '+%F %T %Z') QUEUE DONE" >> queue_b.log
