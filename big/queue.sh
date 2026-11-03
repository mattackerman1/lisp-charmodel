#!/bin/bash
# Phase A: Lisp then PyTorch transformer, 3600 s training time each, sequential, 8 threads, nice -n 10.
cd /home/box/lisp-charmodel/big
chk() { if pgrep -f "science-team|[Ee]xperimenter" >/dev/null; then echo "$(date '+%F %T %Z') WARNING science/experimenter job running; using 4 threads" >> queue.log; echo 4; else echo 8; fi; }
T=$(chk); echo "$(date '+%F %T %Z') start LISP threads=$T load=$(cut -d' ' -f1-3 /proc/loadavg)" >> queue.log
grep -q "=== sample" lisp.log 2>/dev/null || FT=single nice -n 10 sbcl --script gpt-big.lisp --run data=corpus.txt secs=3600 dropout=0.1 fullval-every=600 threads=$T tag=lisp > lisp.log 2>&1
echo "$(date '+%F %T %Z') end LISP" >> queue.log
T=$(chk); echo "$(date '+%F %T %Z') start TORCH threads=$T load=$(cut -d' ' -f1-3 /proc/loadavg)" >> queue.log
grep -q "=== sample" torch.log 2>/dev/null || nice -n 10 /home/box/venvs/torch/bin/python gpt_torch.py data=corpus.txt secs=3600 dropout=0.1 fullval-every=600 threads=$T tag=torch > torch.log 2>&1
echo "$(date '+%F %T %Z') end TORCH" >> queue.log
echo "$(date '+%F %T %Z') QUEUE DONE" >> queue.log
